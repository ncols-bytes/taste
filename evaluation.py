import json
import time
import concurrent.futures
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import torch
import numpy as np
import json
import argparse
from type_vocab.vocab_util import *
from data_process.data_processor import *
from data_process.mysql_table_loader import *
from model.configuration import TableConfig
from model.model import AsymmetricDoubleTower
import time
import threading


class GlobalVars():
    device = None
    data_processor = None
    data_collator = None
    model = None
    type_vocab = None
    use_hist_feature = None
    mysql_table_loader = None
    mysql_table_loader2 = None
    table_2_tags = None
    histogram_map = None
    tables = None

    threshold_alpha = 0.1
    threshold_beta = 0.9

    col_cnt = 0
    col_need_p2_cnt = 0

    Y_pred = []
    Y_true = []

global_vars = GlobalVars


class Scheduler():
    def __init__(self) -> None:
        self.pool1_1 = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pool1_2 = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pool2_1 = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pool2_2 = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.disable_pool =False

    def set_disable_pool(self):
        self.disable_pool = True

    def run_phase1_stage1(self, task_id):
        if not self.disable_pool:
            self.pool1_1.submit(phase1_stage1, task_id)
        else:
            phase1_stage1(task_id)

    def run_phase1_stage2(self, task_id):
        if not self.disable_pool:
            self.pool1_2.submit(phase1_stage2, task_id)
        else:
            phase1_stage2(task_id)
        
    def run_phase2_stage1(self, task_id):
        if not self.disable_pool:
            self.pool2_1.submit(phase2_stage1, task_id)
        else:
            phase2_stage1(task_id)
    
    def run_phase2_stage2(self, task_id):
        if not self.disable_pool:
            self.pool2_2.submit(phase2_stage2, task_id)
        else:
            phase2_stage2(task_id)

scheduler = Scheduler()

data_cache = {}
finished_event = threading.Event()


def phase1_stage1(table_idx):
    table_name = global_vars.tables[table_idx][0]

    use_hist_feature = global_vars.use_hist_feature
    # Phase1: get metadata
    table_id, pgTitle, pgEnt, secTitle, caption, headers, histogram = global_vars.mysql_table_loader.get_metadata(table_name,
                                                                                                        global_vars.histogram_map, use_hist_feature)
    # Phase1: start filter
    input_tok, input_tok_type, input_tok_pos, input_tok_tok_mask, input_tok_len, column_header_mask, \
        col_num, tokenized_meta_length, tokenized_headers_length \
        = global_vars.data_processor.process_single_table_metadata(pgTitle, secTitle, caption, headers)

    global_vars.col_cnt += col_num

    max_input_tok_length = input_tok_len
    max_input_col_num = col_num
    batch_size = 1

    input_tok, input_tok_type, input_tok_pos, input_tok_mask, column_header_mask, \
        histogram = global_vars.data_collator.collate_metadata(batch_size, max_input_tok_length, max_input_col_num, [input_tok],
                                                    [input_tok_type], \
                                                    [input_tok_pos], [input_tok_tok_mask], [input_tok_len], \
                                                    [column_header_mask], [col_num], [histogram] if use_hist_feature else None)

    table_data = {}
    table_data["input_tok"] = input_tok
    table_data["input_tok_type"] = input_tok_type
    table_data["input_tok_pos"] = input_tok_pos
    table_data["input_tok_mask"] = input_tok_mask
    table_data["column_header_mask"] = column_header_mask
    table_data["histogram"] = histogram
    table_data["table_id"] = table_id
    table_data["col_num"] = col_num
    table_data["tokenized_meta_length"] = tokenized_meta_length
    table_data["tokenized_headers_length"] = tokenized_headers_length
    table_data["input_tok_len"] = input_tok_len
    table_data["table_name"] = table_name

    data_cache[table_idx] = table_data

    scheduler.run_phase1_stage2(table_idx)


def phase1_stage2(table_idx):
    table_data = data_cache[table_idx]
    input_tok = table_data["input_tok"]
    input_tok_type = table_data["input_tok_type"]
    input_tok_pos = table_data["input_tok_pos"]
    input_tok_mask = table_data["input_tok_mask"]
    column_header_mask = table_data["column_header_mask"]
    histogram = table_data["histogram"]
    col_num = table_data["col_num"]

    device = global_vars.device
    input_tok = input_tok.to(device)
    input_tok_type = input_tok_type.to(device)
    input_tok_pos = input_tok_pos.to(device)
    input_tok_mask = input_tok_mask.to(device)
    column_header_mask = column_header_mask.to(device)
    if global_vars.use_hist_feature:
        histogram = histogram.to(device)
    input_tok_mask = input_tok_mask[:, :, :input_tok_mask.shape[1]]

    admitted_col_types = [[] for i in range(col_num)]

    with torch.no_grad():
        outputs = global_vars.model(input_tok, input_tok_type, input_tok_pos, input_tok_mask, None, None, None, None, None, column_header_mask, None, None, histogram, table_idx)

    prediction_scores = outputs[0]
    prediction_scores = torch.sigmoid(prediction_scores.view(-1, len(global_vars.type_vocab) + 1)) # An extra one for backgroud type (type:null)
    prediction_scores = prediction_scores[:col_num, :-1] # exclude backgroud type (type:null)

    uncertain_types = (prediction_scores >= global_vars.threshold_alpha) & (prediction_scores <= global_vars.threshold_beta)
    uncertain_col_flags = torch.any(uncertain_types, dim=1)

    certain_scores = prediction_scores * ~uncertain_col_flags.view(-1, 1)
    admitted_types = certain_scores > global_vars.threshold_beta
    admitted_type_indices = torch.nonzero(admitted_types).tolist()
    for [col_idx, tag_idx] in admitted_type_indices:
        admitted_col_types[col_idx].append(tag_idx)

    uncertain_cols = torch.where(uncertain_col_flags)[0].tolist()

    table_data["uncertain_cols"] = uncertain_cols
    table_data["admitted_col_types"] = admitted_col_types
    table_data["input_tok"] = input_tok
    table_data["input_tok_type"] = input_tok_type
    table_data["input_tok_pos"] = input_tok_pos
    table_data["input_tok_mask"] = input_tok_mask
    table_data["column_header_mask"] = column_header_mask
    table_data["histogram"] = histogram
    data_cache[table_idx] = table_data

    scheduler.run_phase2_stage1(table_idx)


def phase2_stage1(table_idx):
    table_data = data_cache[table_idx]
    uncertain_cols = table_data["uncertain_cols"]
    col_num = table_data["col_num"]
    input_tok_len = table_data["input_tok_len"]
    tokenized_meta_length = table_data["tokenized_meta_length"]
    tokenized_headers_length = table_data["tokenized_headers_length"]
    table_name = table_data["table_name"]
    
    if len(uncertain_cols) > 0:
        global_vars.col_need_p2_cnt += len(uncertain_cols)

        entities = global_vars.mysql_table_loader2.get_entity_data(table_name, col_num, uncertain_cols)

        input_ent_text, input_ent_cell_length, input_ent_type, input_ent_mask, \
            column_entity_mask, input_ent_len \
            = global_vars.data_processor.process_single_table_entity_data(entities, col_num,
                                                                tokenized_meta_length,
                                                                tokenized_headers_length, uncertain_cols)
        
        max_input_tok_length = input_tok_len
        max_input_col_num = col_num
        max_input_ent_length = 10 * max_input_col_num + 1
        max_input_cell_length = 10
        batch_size = 1

        input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask, \
            column_entity_mask \
            = global_vars.data_collator.collate_entity_data(batch_size, max_input_tok_length, max_input_ent_length,
                                                max_input_cell_length, max_input_col_num, \
                                                [input_ent_text], [input_ent_cell_length],
                                                [input_ent_type], [input_ent_mask], [input_ent_len], \
                                                [column_entity_mask], [input_tok_len], [col_num])

        table_data["input_ent_text_length"] = input_ent_text_length
        table_data["input_ent_text"] = input_ent_text
        table_data["input_ent_type"] = input_ent_type
        table_data["input_ent_mask"] = input_ent_mask
        table_data["column_entity_mask"] = column_entity_mask
        data_cache[table_idx] = table_data

    scheduler.run_phase2_stage2(table_idx)


def phase2_stage2(table_idx):
    table_data = data_cache[table_idx]

    uncertain_cols = table_data["uncertain_cols"]
    col_num = table_data["col_num"]
    admitted_col_types = table_data["admitted_col_types"]
    table_id = table_data["table_id"]

    if len(uncertain_cols) > 0:
        input_tok = table_data["input_tok"]
        histogram = table_data["histogram"]
        column_header_mask = table_data["column_header_mask"]
        input_tok_pos = table_data["input_tok_pos"]
        input_tok_type = table_data["input_tok_type"]
        input_tok_mask = table_data["input_tok_mask"]
        input_ent_text = table_data["input_ent_text"]
        input_ent_text_length = table_data["input_ent_text_length"]
        input_ent_type = table_data["input_ent_type"]
        input_ent_mask = table_data["input_ent_mask"]
        column_entity_mask = table_data["column_entity_mask"]

        device = global_vars.device
        input_tok_mask = input_tok_mask.to(device)
        input_ent_text = input_ent_text.to(device)
        input_ent_text_length = input_ent_text_length.to(device)
        input_ent_type = input_ent_type.to(device)
        input_ent_mask = input_ent_mask.to(device)
        column_entity_mask = column_entity_mask.to(device)

        with torch.no_grad():
            outputs = global_vars.model(input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
                input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask, column_entity_mask, column_header_mask, None, None, histogram, table_idx)
            prediction_scores = outputs[1]
            prediction_scores = torch.sigmoid(prediction_scores.view(-1, len(global_vars.type_vocab) + 1)) # An extra one for backgroud type (type:null)
            prediction_scores = prediction_scores[:col_num, :-1] # exclude backgroud type (type:null)

            prediction_labels = prediction_scores > 0.5
            label_indices = torch.nonzero(prediction_labels).tolist()
            for [col_idx, tag_idx] in label_indices:
                if col_idx in uncertain_cols:
                    admitted_col_types[col_idx].append(tag_idx)

    for col_idx in range(col_num):
        y_pred = [False] * len(global_vars.type_vocab)
        y_true = [False] * len(global_vars.type_vocab)

        for tag_idx in admitted_col_types[col_idx]:
            y_pred[tag_idx] = True
        global_vars.Y_pred.append(y_pred)

        for tag in global_vars.table_2_tags[table_id][col_idx]:
            if tag in global_vars.type_vocab:
                y_true[global_vars.type_vocab[tag]] = True
        global_vars.Y_true.append(y_true)

    print(table_idx + 1, '/', len(global_vars.tables), end = "\r")
    if table_idx == len(global_vars.tables) - 1:
        finished_event.set()


def load_model(model_class, config_name, checkpoint_path, type_vocab, use_hist_feature, device):
    config_class = TableConfig
    config = config_class.from_pretrained(config_name)
    config.class_num = len(type_vocab) + 1 # An extra one for backgroud type (type:null)
    config.use_histogram_feature = use_hist_feature

    model = model_class(config, is_simple=True)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mysql_host", default=None, type=str, required=True)
    parser.add_argument("--mysql_port", default=3306, type=int, required=False)
    parser.add_argument("--mysql_user", default=None, type=str, required=True)
    parser.add_argument("--mysql_password", default=None, type=str, required=True)
    parser.add_argument("--eval_database", default=None, type=str, required=True)
    parser.add_argument("--model_dir", default=None, type=str, required=True)
    parser.add_argument('--use_histogram_feature', action='store_true')
    parser.add_argument("--test_dataset", default=None, type=str, required=True)
    parser.add_argument("--type_vocab", default=None, type=str, required=True)
    parser.add_argument("--threshold_alpha", default=0.1, type=float, required=False)
    parser.add_argument("--threshold_beta", default=0.9, type=float, required=False)
    parser.add_argument('--enable_pipeline', action='store_true')
    args = parser.parse_args()

    if not args.enable_pipeline:
        scheduler.set_disable_pool()
        
    global_vars.threshold_alpha = args.threshold_alpha
    global_vars.threshold_beta = args.threshold_beta

    # load model
    device = torch.device('cuda')
    hybrid_model_config = 'model/hybrid_model_config.json'
    type_vocab = load_type_vocab(args.type_vocab)
    model_path = args.model_dir + '/pytorch_model.bin'
    model = load_model(AsymmetricDoubleTower, hybrid_model_config, model_path, type_vocab, args.use_histogram_feature, device)
    print('load model finished.')

    data_processor = DataProcessor(None, type_vocab=None, max_input_tok=500, src="test", max_length=[50, 10, 10])
    data_collator = DataCollator(data_processor.tokenizer, is_train=False)

    mysql_table_loader = MysqlTableLoader(args.mysql_host, args.mysql_port, args.mysql_user, args.mysql_password,args.eval_database)
    mysql_table_loader2 = MysqlTableLoader(args.mysql_host, args.mysql_port, args.mysql_user, args.mysql_password, args.eval_database)
    mysql_table_loader.connect()
    mysql_table_loader2.connect()
    tables = mysql_table_loader.list_all_tables()

    # load label from original json
    table_2_tags = {}
    with open(args.test_dataset, "r") as fcc_file:
        fcc_data = json.load(fcc_file)
        for table_idx in range(len(fcc_data)):
            table_id = fcc_data[table_idx][0]
            annotations = fcc_data[table_idx][7]
            table_2_tags[str(table_id)] = annotations
    
    print(f'Evaluating {args.eval_database}...')
    evaluation_time_start = time.time()
    
    # preload histograms
    histogram_map = None
    if args.use_histogram_feature:
        histograms = mysql_table_loader.get_histograms()
        histogram_map = HistogramHelper().reformat_mysql_histograms(histograms)

    # set global args
    global_vars.col_cnt = 0
    global_vars.col_need_p2_cnt = 0
    global_vars.Y_pred = []
    global_vars.Y_true = []
    global_vars.device = device
    global_vars.data_processor = data_processor
    global_vars.data_collator = data_collator
    global_vars.model = model
    global_vars.type_vocab = type_vocab
    global_vars.use_hist_feature = args.use_histogram_feature
    global_vars.mysql_table_loader = mysql_table_loader
    global_vars.mysql_table_loader2 = mysql_table_loader2
    global_vars.table_2_tags = table_2_tags
    global_vars.histogram_map = histogram_map
    global_vars.tables = tables
    
    for table_idx in range(len(tables)):
        scheduler.run_phase1_stage1(table_idx)

    finished_event.wait()
    total_time = time.time() - evaluation_time_start

    print()
    print("============== Execution Time =============")
    print(f"Total time (s): {total_time}")
    print()

    print("======== Intrusiveness to Database ========")
    print("All tables cnt:", len(tables))
    print("All columns cnt:", global_vars.col_cnt)
    print("Scanned columns cnt:", global_vars.col_need_p2_cnt)
    print("Ratio of scanned columns:", global_vars.col_need_p2_cnt / global_vars.col_cnt)
    print()

    Y_true = np.array(global_vars.Y_true)
    Y_pred = np.array(global_vars.Y_pred)
    precision = precision_score(Y_true, Y_pred, average='micro')
    recall = recall_score(Y_true, Y_pred, average='micro')
    f1 = f1_score(Y_true, Y_pred, average='micro')

    print("================  Accuracy ================")
    print("Micro-Precision:", precision)
    print("Micro-Recall:", recall)
    print("Micro-F1:", f1)
    print()

    mysql_table_loader.disconnect()
    mysql_table_loader2.disconnect()


if __name__ == "__main__":
    main()

