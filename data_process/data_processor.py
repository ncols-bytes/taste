# Base on: https://github.com/sunlab-osu/TURL/blob/release_ongoing/data_loader/CT_Wiki_data_loaders.py

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import json
from tqdm import tqdm
import itertools
from multiprocessing import Pool
from functools import partial
from data_process.histogram_helper import *

from model.transformers import BertTokenizer


class DataProcessor(Dataset):
    def process_single_table_metadata(self, pgTitle, secTitle, caption, headers):

        tokenized_pgTitle = self.tokenizer.encode(pgTitle, max_length=self.max_title_length, add_special_tokens=False)
        tokenized_meta = tokenized_pgTitle + \
                            self.tokenizer.encode(secTitle, max_length=self.max_title_length, add_special_tokens=False)
        if caption != secTitle:
            tokenized_meta += self.tokenizer.encode(caption, max_length=self.max_title_length, add_special_tokens=False)
        tokenized_headers = [self.tokenizer.encode(z, max_length=self.max_header_length, add_special_tokens=False) for z
                                in headers]
        input_tok = []
        input_tok_pos = []
        input_tok_type = []
        tokenized_meta_length = len(tokenized_meta)
        input_tok += tokenized_meta
        input_tok_pos += list(range(tokenized_meta_length))
        input_tok_type += [0] * tokenized_meta_length
        tokenized_headers_length = [len(z) for z in tokenized_headers]
        input_tok += list(itertools.chain(*tokenized_headers))
        input_tok_pos += list(itertools.chain(*[list(range(z)) for z in tokenized_headers_length]))
        input_tok_type += [1] * sum(tokenized_headers_length)

        # create column header mask
        start_i = 0
        header_span = {}
        column_header_mask = np.zeros([len(headers), len(input_tok)], dtype=int)
        for j in range(len(headers)):
            header_span[j] = (start_i, start_i + tokenized_headers_length[j])
            column_header_mask[j,
            tokenized_meta_length + header_span[j][0]:tokenized_meta_length + header_span[j][1]] = 1
            start_i += tokenized_headers_length[j]
        # create input mask
        input_tok_tok_mask = np.ones([len(input_tok), len(input_tok)], dtype=int)

        return np.array(input_tok), np.array(input_tok_type), np.array(input_tok_pos),input_tok_tok_mask, len(input_tok), \
                column_header_mask, len(headers), tokenized_meta_length, tokenized_headers_length
    
    def process_single_table_entity_data(self, entities, col_num, tokenized_meta_length,
                                tokenized_headers_length, uncertain_cols=None):
        entities = [z for column in entities for z in column[:self.max_row]]

        input_ent = []
        input_ent_text = []
        input_ent_type = []
        column_en_map = {}
        row_en_map = {}

        if uncertain_cols != None:
            reduced_entities = []
            for entity in entities:
                col_idx = entity[0][1]
                if col_idx in uncertain_cols:
                    reduced_entities.append(entity)
            entities = reduced_entities

        for e_i, (index, cell) in enumerate(entities):
            entity, entity_text = cell
            tokenized_ent_text = self.tokenizer.encode(entity_text, max_length=self.max_cell_length,
                                                        add_special_tokens=False)
            input_ent.append(entity)
            input_ent_text.append(tokenized_ent_text)
            input_ent_type.append(4)
            if index[1] not in column_en_map:
                column_en_map[index[1]] = [e_i]
            else:
                column_en_map[index[1]].append(e_i)
            if index[0] not in row_en_map:
                row_en_map[index[0]] = [e_i]
            else:
                row_en_map[index[0]].append(e_i)

        # create column entity mask
        column_entity_mask = np.zeros([col_num, len(input_ent_text)], dtype=int)
        for j in range(col_num):
            if j not in column_en_map:
                continue
            for e_i_1 in column_en_map[j]:
                column_entity_mask[j, e_i_1] = 1

        # create input mask
        header_ent_mask = np.ones([sum(tokenized_headers_length), len(input_ent_text)], dtype=int)
        ent_header_mask = np.transpose(header_ent_mask)

        ent_meta_mask = np.ones([len(input_ent_text), tokenized_meta_length], dtype=int)

        ent_ent_mask = np.eye(len(input_ent_text), dtype=int)
        for _, e_is in column_en_map.items():
            for e_i_1 in e_is:
                for e_i_2 in e_is:
                    ent_ent_mask[e_i_1, e_i_2] = 1
        input_ent_mask = [np.concatenate([ent_meta_mask, ent_header_mask], axis=1), ent_ent_mask]
        
        input_ent_cell_length = [len(x) if len(x) != 0 else 1 for x in input_ent_text]
        if len(input_ent_cell_length) == 0:
            max_cell_length = 0
        else:
            max_cell_length = max(input_ent_cell_length)
        input_ent_text_padded = np.zeros([len(input_ent_text), max_cell_length], dtype=int)
        for i, x in enumerate(input_ent_text):
            input_ent_text_padded[i, :len(x)] = x
        input_ent_len = len(input_ent_text)

        return input_ent_text_padded, input_ent_cell_length, np.array(input_ent_type), \
                (np.array(input_ent_mask[0]), np.array(input_ent_mask[1])), column_entity_mask, input_ent_len

    def process_single_table_labels(self, type_annotations):
        labels = np.zeros([len(type_annotations), self.type_num], dtype=int)
        for j, types in enumerate(type_annotations):
            has_type = False
            for t in types:
                if t in self.type_vocab:
                    labels[j, self.type_vocab[t]] = 1
                    has_type = True
            if not has_type:
                labels[j, len(self.type_vocab)] = 1 # backgroud type (type:null)
        return labels

    def process_single_table(self, input_data):
        table_id, pgTitle, pgEnt, secTitle, caption, headers, entities, type_annotations = input_data

        input_tok, input_tok_type, input_tok_pos, input_tok_tok_mask, input_tok_len, column_header_mask, \
                col_num, tokenized_meta_length, tokenized_headers_length \
                = self.process_single_table_metadata(pgTitle, secTitle, caption, headers)

        histogram = HistogramHelper().gen_histogram_from_entities(entities)

        input_ent_text_padded, input_ent_cell_length, input_ent_type,  input_ent_mask, \
                column_entity_mask, input_ent_len \
                = self.process_single_table_entity_data(entities, col_num, tokenized_meta_length,
                                    tokenized_headers_length)

        labels = self.process_single_table_labels(type_annotations)
        
        return [table_id,np.array(input_tok),np.array(input_tok_type),np.array(input_tok_pos),np.array(input_tok_tok_mask),len(input_tok), \
                    input_ent_text_padded,input_ent_cell_length,np.array(input_ent_type),(np.array(input_ent_mask[0]),np.array(input_ent_mask[1])),input_ent_len, \
                    column_header_mask,column_entity_mask,labels,len(type_annotations),histogram]

    def _read_from_json_and_preprocess(self, dataset_path):
        print(f'loading {dataset_path}')
        with open(dataset_path, "r") as f:
            cols = json.load(f)

        print(f'processing {dataset_path}')
        pool = Pool(processes=1) # you can increase processes if your RAM is sufficient
        processed_cols = list(tqdm(pool.imap(partial(self.process_single_table), cols, chunksize=1000),total=len(cols)))
        pool.close()

        return processed_cols

    def __init__(self, dataset_path, type_vocab, src="train", max_row=10, max_input_tok=500, max_length = [50, 10, 10], tokenizer = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_input_tok = max_input_tok
        self.max_title_length = max_length[0]
        self.max_header_length = max_length[1]
        self.max_cell_length = max_length[2]
        self.max_row = max_row

        if src == 'train' or src == 'dev':
            self.type_vocab = type_vocab
            self.type_num = len(self.type_vocab) + 1 # An extra one for type:null
            self.data = self._read_from_json_and_preprocess(dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DataCollator:
    def __init__(self, tokenizer, is_train=True):
        self.tokenizer = tokenizer
        self.is_train = is_train

    def collate_metadata(self, batch_size, max_input_tok_length, max_input_col_num, \
                         batch_input_tok, batch_input_tok_type, batch_input_tok_pos, batch_input_tok_tok_mask, batch_input_tok_length, \
                         batch_column_header_mask, batch_col_num, batch_histogram):
        
        batch_input_tok_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_type_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_pos_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_mask_padded = np.zeros([batch_size, max_input_tok_length, max_input_tok_length], dtype=int)

        batch_column_header_mask_padded = np.zeros([batch_size, max_input_col_num, max_input_tok_length], dtype=int)

        batch_histogram_padded = None
        if batch_histogram is not None:
            batch_histogram_padded = np.zeros([batch_size, max_input_col_num, batch_histogram[0].shape[-1]], dtype=float)

        for i, (tok_l, col_num) in enumerate(zip(batch_input_tok_length, batch_col_num)):
            batch_input_tok_padded[i, :tok_l] = batch_input_tok[i]
            batch_input_tok_type_padded[i, :tok_l] = batch_input_tok_type[i]
            batch_input_tok_pos_padded[i, :tok_l] = batch_input_tok_pos[i]
            batch_input_tok_mask_padded[i, :tok_l, :tok_l] = batch_input_tok_tok_mask[i]

            batch_column_header_mask_padded[i, :col_num, :tok_l] = batch_column_header_mask[i]
            batch_column_header_mask_padded[i, col_num:, 0] = 1

            if batch_histogram is not None:
                batch_histogram_padded[i, :col_num] = batch_histogram[i]

        batch_input_tok_padded = torch.LongTensor(batch_input_tok_padded)
        batch_input_tok_type_padded = torch.LongTensor(batch_input_tok_type_padded)
        batch_input_tok_pos_padded = torch.LongTensor(batch_input_tok_pos_padded)
        batch_input_tok_mask_padded = torch.LongTensor(batch_input_tok_mask_padded)

        batch_column_header_mask_padded = torch.FloatTensor(batch_column_header_mask_padded)

        if batch_histogram is not None:
            batch_histogram_padded = torch.FloatTensor(batch_histogram_padded)

        return batch_input_tok_padded, batch_input_tok_type_padded, batch_input_tok_pos_padded, batch_input_tok_mask_padded, \
            batch_column_header_mask_padded, batch_histogram_padded

    def collate_entity_data(self, batch_size, max_input_tok_length, max_input_ent_length, max_input_cell_length, max_input_col_num, \
                            batch_input_ent_text, batch_input_ent_cell_length, batch_input_ent_type, batch_input_ent_mask, batch_input_ent_length, \
                            batch_column_entity_mask, batch_input_tok_length, batch_col_num):
        
        batch_input_ent_text_padded = np.zeros([batch_size, max_input_ent_length, max_input_cell_length], dtype=int)
        batch_input_ent_text_length = np.ones([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_type_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_mask_padded = np.zeros(
            [batch_size, max_input_ent_length, max_input_tok_length + max_input_ent_length], dtype=int)

        batch_column_entity_mask_padded = np.zeros([batch_size, max_input_col_num, max_input_ent_length], dtype=int)

        for i, (tok_l, ent_l, col_num) in enumerate(zip(batch_input_tok_length, batch_input_ent_length, batch_col_num)):
            batch_input_ent_text_padded[i, :ent_l, :batch_input_ent_text[i].shape[-1]] = batch_input_ent_text[i]
            batch_input_ent_text_length[i, :ent_l] = batch_input_ent_cell_length[i]
            batch_input_ent_type_padded[i, :ent_l] = batch_input_ent_type[i]
            batch_input_ent_mask_padded[i, :ent_l, :tok_l] = batch_input_ent_mask[i][0]
            batch_input_ent_mask_padded[i, :ent_l, max_input_tok_length:max_input_tok_length + ent_l] = \
            batch_input_ent_mask[i][1]
            batch_column_entity_mask_padded[i, :col_num, :ent_l] = batch_column_entity_mask[i]
            batch_column_entity_mask_padded[i, col_num:, 0] = 1

        batch_input_ent_text_padded = torch.LongTensor(batch_input_ent_text_padded)
        batch_input_ent_text_length = torch.LongTensor(batch_input_ent_text_length)
        batch_input_ent_type_padded = torch.LongTensor(batch_input_ent_type_padded)
        batch_input_ent_mask_padded = torch.LongTensor(batch_input_ent_mask_padded)

        batch_column_entity_mask_padded = torch.FloatTensor(batch_column_entity_mask_padded)

        return batch_input_ent_text_padded, batch_input_ent_text_length, batch_input_ent_type_padded, batch_input_ent_mask_padded, \
            batch_column_entity_mask_padded

    def collate_label(self, batch_size, max_input_col_num, batch_labels, batch_col_num):
        batch_labels_mask = np.zeros([batch_size, max_input_col_num], dtype=int)
        batch_labels_padded = np.zeros([batch_size, max_input_col_num, batch_labels[0].shape[-1]], dtype=int)

        for i, col_num in enumerate(batch_col_num):
            batch_labels_mask[i, :col_num] = batch_labels[i].sum(1)!=0
            batch_labels_padded[i, :col_num] = batch_labels[i]

        batch_labels_mask = torch.FloatTensor(batch_labels_mask)
        batch_labels_padded = torch.FloatTensor(batch_labels_padded)

        return batch_labels_mask, batch_labels_padded

    def __call__(self, raw_batch):
        batch_table_id, batch_input_tok, batch_input_tok_type, batch_input_tok_pos, batch_input_tok_tok_mask, batch_input_tok_length, \
            batch_input_ent_text, batch_input_ent_cell_length, batch_input_ent_type, batch_input_ent_mask, batch_input_ent_length, \
            batch_column_header_mask, batch_column_entity_mask, batch_labels, batch_col_num, batch_histogram = zip(*raw_batch)
        
        batch_size = len(batch_table_id)
        max_input_tok_length = max(batch_input_tok_length)
        max_input_col_num = max(batch_col_num)

        batch_input_tok_padded, batch_input_tok_type_padded, batch_input_tok_pos_padded, batch_input_tok_mask_padded, \
            batch_column_header_mask_padded, batch_histogram_padded \
            = self.collate_metadata(batch_size, max_input_tok_length, max_input_col_num, batch_input_tok, batch_input_tok_type, \
                                    batch_input_tok_pos, batch_input_tok_tok_mask, batch_input_tok_length, \
                                    batch_column_header_mask, batch_col_num, batch_histogram)

        max_input_cell_length = max([z.shape[-1] for z in batch_input_ent_text])
        max_input_ent_length = max(batch_input_ent_length)

        batch_input_ent_text_padded, batch_input_ent_text_length, batch_input_ent_type_padded, batch_input_ent_mask_padded, \
            batch_column_entity_mask_padded \
            = self.collate_entity_data(batch_size, max_input_tok_length, max_input_ent_length, max_input_cell_length, max_input_col_num, \
                                         batch_input_ent_text, batch_input_ent_cell_length, batch_input_ent_type, batch_input_ent_mask, batch_input_ent_length, \
                                        batch_column_entity_mask, batch_input_tok_length, batch_col_num)
        
        batch_labels_mask, batch_labels_padded = self.collate_label(batch_size, max_input_col_num, batch_labels, batch_col_num)

        return batch_table_id, batch_input_tok_padded, batch_input_tok_type_padded, batch_input_tok_pos_padded, batch_input_tok_mask_padded, \
                batch_input_ent_text_padded, batch_input_ent_text_length, batch_input_ent_type_padded, batch_input_ent_mask_padded, \
                batch_column_entity_mask_padded, batch_column_header_mask_padded, batch_labels_mask, batch_labels_padded, batch_histogram_padded


class WikiDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        is_train = True,
        num_workers=0,
        sampler=None,
    ):
        self.shuffle = shuffle
        if sampler is not None:
            self.shuffle = False

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.is_train = is_train
        self.collate_fn = DataCollator(dataset.tokenizer, is_train=self.is_train)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
            "sampler": sampler
        }
        super().__init__(**self.init_kwargs)
