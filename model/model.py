# coding=utf-8
# Base on: https://github.com/sunlab-osu/TURL/blob/release_ongoing/model/model.py

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
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
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F

from model.transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from model.transformers.configuration_bert import BertConfig
from model.transformers.file_utils import add_start_docstrings
from model.transformers.modeling_bert import *
import torch.nn.functional as F
from data_process.histogram_helper import *

import pdb

logger = logging.getLogger(__name__)

class TableEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(TableEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.ent_embeddings = nn.Embedding(config.ent_vocab_size, config.hidden_size, padding_idx=0).cpu()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def load_pretrained(self, checkpoint, is_bert=True):
        state_dict = self.state_dict()
        if is_bert:
            state_dict['LayerNorm.weight'] = checkpoint['bert.embeddings.LayerNorm.weight']
            state_dict['LayerNorm.bias'] = checkpoint['bert.embeddings.LayerNorm.bias']
            state_dict['word_embeddings.weight'] = checkpoint['bert.embeddings.word_embeddings.weight']
            state_dict['position_embeddings.weight'] = checkpoint['bert.embeddings.position_embeddings.weight']
            new_type_size = state_dict['type_embeddings.weight'].shape[0]
            state_dict['type_embeddings.weight'] = checkpoint['bert.embeddings.token_type_embeddings.weight'][0].repeat(new_type_size).view(new_type_size, -1)
        else:
            for key in state_dict:
                state_dict[key] = checkpoint['table.embeddings.'+key]
        self.load_state_dict(state_dict)

    def forward(self, input_tok, input_tok_type, input_tok_pos, input_ent = None, input_ent_type = None, ent_candidates = None):
        input_tok_embeds = self.word_embeddings(input_tok)
        input_tok_pos_embeds = self.position_embeddings(input_tok_pos)
        input_tok_type_embeds = self.type_embeddings(input_tok_type)

        tok_embeddings = input_tok_embeds + input_tok_pos_embeds + input_tok_type_embeds
        tok_embeddings = self.LayerNorm(tok_embeddings)
        tok_embeddings = self.dropout(tok_embeddings)

        ent_embeddings = None
        if input_ent is not None:
            input_ent_embeds = self.ent_embeddings(input_ent)
        if input_ent_type is not None:
            input_ent_type_embeds = self.type_embeddings(input_ent_type)
            ent_embeddings = input_ent_embeds + input_ent_type_embeds

        if ent_embeddings is not None:
            ent_embeddings = self.LayerNorm(ent_embeddings)
            ent_embeddings = self.dropout(ent_embeddings)

        if ent_candidates is not None:
            ent_candidates_embeddings = self.ent_embeddings(ent_candidates)
        else:
            ent_candidates_embeddings = None

        return tok_embeddings, ent_embeddings, ent_candidates_embeddings

class TableHeaderEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(TableHeaderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0,sparse=True)
        self.header_embeddings = nn.Embedding(config.header_vocab_size, config.hidden_size, padding_idx=0,sparse=True)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def load_pretrained(self, checkpoint, is_bert=True):
        state_dict = self.state_dict()
        if is_bert:
            state_dict['LayerNorm.weight'] = checkpoint['bert.embeddings.LayerNorm.weight']
            state_dict['LayerNorm.bias'] = checkpoint['bert.embeddings.LayerNorm.bias']
            state_dict['word_embeddings.weight'] = checkpoint['bert.embeddings.word_embeddings.weight']
            state_dict['position_embeddings.weight'] = checkpoint['bert.embeddings.position_embeddings.weight']
            new_type_size = state_dict['type_embeddings.weight'].shape[0]
            state_dict['type_embeddings.weight'] = checkpoint['bert.embeddings.token_type_embeddings.weight'][0].repeat(new_type_size).view(new_type_size, -1)
        else:
            state_dict['LayerNorm.weight'] = checkpoint['table.embeddings.LayerNorm.weight']
            state_dict['LayerNorm.bias'] = checkpoint['table.embeddings.LayerNorm.bias']
            state_dict['word_embeddings.weight'] = checkpoint['table.embeddings.word_embeddings.weight']
            state_dict['position_embeddings.weight'] = checkpoint['table.embeddings.position_embeddings.weight']
            state_dict['type_embeddings.weight'] = checkpoint['table.embeddings.type_embeddings.weight']
        self.load_state_dict(state_dict)

    def forward(self, input_tok, input_tok_type, input_tok_pos, input_header, input_header_type):
        input_tok_embeds = self.word_embeddings(input_tok)
        input_tok_pos_embeds = self.position_embeddings(input_tok_pos)
        input_tok_type_embeds = self.type_embeddings(input_tok_type)

        tok_embeddings = input_tok_embeds + input_tok_pos_embeds + input_tok_type_embeds
        tok_embeddings = self.LayerNorm(tok_embeddings)
        tok_embeddings = self.dropout(tok_embeddings)

        header_embeddings = self.header_embeddings(input_header)
        input_header_type_embeds = self.type_embeddings(input_header_type)
        header_embeddings += input_header_type_embeds

        if header_embeddings is not None:
            header_embeddings = self.LayerNorm(header_embeddings)
            header_embeddings = self.dropout(header_embeddings)

        return tok_embeddings, header_embeddings

class TableHybridEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(TableHybridEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0,sparse=False)
        self.ent_embeddings = nn.Embedding(config.ent_vocab_size, config.hidden_size, padding_idx=0,sparse=False)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.ent_mask_embedding = nn.Embedding(4, config.hidden_size, padding_idx=0)

        self.fusion = nn.Linear(2*config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def load_pretrained(self, checkpoint, is_bert=True):
        state_dict = self.state_dict()
        if is_bert:
            state_dict['LayerNorm.weight'] = checkpoint['bert.embeddings.LayerNorm.weight']
            state_dict['LayerNorm.bias'] =  ['bert.embeddings.LayerNorm.bias']
            state_dict['word_embeddings.weight'] = checkpoint['bert.embeddings.word_embeddings.weight']
            state_dict['position_embeddings.weight'] = checkpoint['bert.embeddings.position_embeddings.weight']
            new_type_size = state_dict['type_embeddings.weight'].shape[0]
            state_dict['type_embeddings.weight'] = checkpoint['bert.embeddings.token_type_embeddings.weight'][0].repeat(new_type_size).view(new_type_size, -1)
        else:
            for key in state_dict:
                state_dict[key] = checkpoint['table.embeddings.'+key]
        self.load_state_dict(state_dict)

    def forward(self, input_tok = None, input_tok_type = None, input_tok_pos = None, input_ent_tok = None, input_ent_tok_length = None, input_ent_mask_type = None, input_ent = None, input_ent_type = None, ent_candidates = None):
        tok_embeddings = None
        if input_tok is not None:
            input_tok_embeds = self.word_embeddings(input_tok)
            input_tok_pos_embeds = self.position_embeddings(input_tok_pos)
            input_tok_type_embeds = self.type_embeddings(input_tok_type)

            tok_embeddings = input_tok_embeds + input_tok_pos_embeds + input_tok_type_embeds
            tok_embeddings = self.LayerNorm(tok_embeddings)
            tok_embeddings = self.dropout(tok_embeddings)

        if input_ent is None and input_ent_tok is None:
            return tok_embeddings, None, None

        ent_embeddings = None
        if input_ent_tok is not None:
            input_ent_tok_embeds = self.word_embeddings(input_ent_tok)
            input_ent_tok_embeds = input_ent_tok_embeds.sum(dim=-2)
            input_ent_tok_embeds = input_ent_tok_embeds/input_ent_tok_length[:,:,None]
            if input_ent_mask_type is not None:
                input_ent_mask_embeds = self.ent_mask_embedding(input_ent_mask_type)
                input_ent_tok_embeds = torch.where((input_ent_mask_type!=0)[:,:,None], input_ent_mask_embeds, input_ent_tok_embeds)
        if input_ent is not None:
                # if input_ent.is_cuda:
                #     input_ent_embeds = self.ent_embeddings(input_ent.cpu()).cuda()
                # else:
            input_ent_embeds = self.ent_embeddings(input_ent)
            if input_ent_tok is None:
                input_ent_tok_embeds = torch.zeros_like(input_ent_embeds)
        else:
            input_ent_embeds = torch.zeros_like(input_ent_tok_embeds)
        ent_embeddings = self.fusion(torch.cat([input_ent_embeds, input_ent_tok_embeds], dim=-1))
        ent_embeddings = self.transform_act_fn(ent_embeddings)
        ent_embeddings = self.LayerNorm(ent_embeddings)
        ent_embeddings = self.dropout(ent_embeddings)

        if input_ent_type is not None:
            input_ent_type_embeds = self.type_embeddings(input_ent_type)
            ent_embeddings += input_ent_type_embeds

        if ent_embeddings is not None:
            ent_embeddings = self.LayerNorm(ent_embeddings)
            ent_embeddings = self.dropout(ent_embeddings)

        if ent_candidates is not None:
            ent_candidates_embeddings = self.ent_embeddings(ent_candidates)
        else:
            ent_candidates_embeddings = None

        return tok_embeddings, ent_embeddings, ent_candidates_embeddings

class TableELEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(TableELEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0,sparse=True)
        self.ent_type_embeddings = nn.Embedding(config.ent_type_vocab_size, config.hidden_size, padding_idx=0,sparse=True)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def load_pretrained(self, checkpoint, is_bert=True):
        state_dict = self.state_dict()
        if is_bert:
            state_dict['LayerNorm.weight'] = checkpoint['bert.embeddings.LayerNorm.weight']
            state_dict['LayerNorm.bias'] = checkpoint['bert.embeddings.LayerNorm.bias']
            state_dict['word_embeddings.weight'] = checkpoint['bert.embeddings.word_embeddings.weight']
        else:
            for key in state_dict:
                if 'table.embeddings.'+key in checkpoint:
                    state_dict[key] = checkpoint['table.embeddings.'+key]
        self.load_state_dict(state_dict)

    def forward(self, cand_name=None, cand_name_length=None, cand_description=None, cand_description_length=None, cand_type=None, cand_type_length=None):
        cand_embeddings = []
        if cand_name is not None:
            cand_name_embeds = self.word_embeddings(cand_name)
            cand_name_embeds = cand_name_embeds.sum(dim=-2)
            cand_name_embeds = cand_name_embeds/cand_name_length[:,:,None]
            cand_name_embeds = self.LayerNorm(cand_name_embeds)
            cand_name_embeds = self.dropout(cand_name_embeds)
            cand_embeddings.append(cand_name_embeds)

        if cand_description is not None:
            cand_description_embeds = self.word_embeddings(cand_description)
            cand_description_embeds = cand_description_embeds.sum(dim=-2)
            cand_description_embeds = cand_description_embeds/cand_description_length[:,:,None]
            cand_description_embeds = self.LayerNorm(cand_description_embeds)
            cand_description_embeds = self.dropout(cand_description_embeds)
            cand_embeddings.append(cand_description_embeds)

        if cand_type is not None:
            cand_type_embeds = self.ent_type_embeddings(cand_type)
            cand_type_embeds = cand_type_embeds.sum(dim=-2)
            cand_type_embeds = cand_type_embeds/cand_type_length[:,:,None]
            cand_type_embeds = self.LayerNorm(cand_type_embeds)
            cand_type_embeds = self.dropout(cand_type_embeds)
            cand_embeddings.append(cand_type_embeds)

        cand_embeddings = torch.cat(cand_embeddings, dim=-1)

        return cand_embeddings


class TableLayerSimple(nn.Module):
    def __init__(self, config):
        super(TableLayerSimple, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.tok_hidden_states_cache = {}
        
    def forward(self, tok_hidden_states=None, tok_attention_mask=None, ent_hidden_states=None, ent_attention_mask=None, predict_id=None):
        tok_outputs, ent_outputs = (None, None), (None, None)
        # if predict_id not in self.tok_hidden_states_cache:
        if tok_hidden_states is not None: # train or first predict
            tok_self_attention_outputs = self.attention(tok_hidden_states, encoder_hidden_states=tok_hidden_states, encoder_attention_mask=tok_attention_mask)
            tok_attention_output = tok_self_attention_outputs[0]
            tok_outputs = tok_self_attention_outputs[1:]
            tok_intermediate_output = self.intermediate(tok_attention_output)
            tok_layer_output = self.output(tok_intermediate_output, tok_attention_output)
            tok_outputs = (tok_layer_output,) + tok_outputs
            self.tok_hidden_states_cache[predict_id] = tok_hidden_states
        else:
            tok_hidden_states = self.tok_hidden_states_cache[predict_id]

        if ent_hidden_states is not None:
            ent_self_attention_outputs = self.attention(ent_hidden_states, encoder_hidden_states=torch.cat([tok_hidden_states, ent_hidden_states], dim=1), encoder_attention_mask=ent_attention_mask)
            ent_attention_output = ent_self_attention_outputs[0]
            ent_outputs = ent_self_attention_outputs[1:]
            ent_intermediate_output = self.intermediate(ent_attention_output)
            ent_layer_output = self.output(ent_intermediate_output, ent_attention_output)
            ent_outputs = (ent_layer_output,) + ent_outputs

        return tok_outputs, ent_outputs


class TableEncoderSimple(nn.Module):
    def __init__(self, config):
        super(TableEncoderSimple, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([TableLayerSimple(config) for _ in range(config.num_hidden_layers)])

    def load_pretrained(self, checkpoint, is_bert=True):
        state_dict = self.state_dict()
        if is_bert:
            for x in state_dict:
                state_dict[x] = checkpoint['bert.encoder.'+x]
                print('load %s <- %s'%(x, 'bert.encoder.'+x))
        else:
            for x in state_dict:
                state_dict[x] = checkpoint['table.encoder.'+x]
        self.load_state_dict(state_dict)

    def forward(self, tok_hidden_states=None, tok_attention_mask=None, ent_hidden_states=None, ent_attention_mask=None, predict_id=None):
        tok_all_hidden_states = ()
        tok_all_attentions = ()
        ent_all_hidden_states = ()
        ent_all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                tok_all_hidden_states = tok_all_hidden_states + (tok_hidden_states,)
                ent_all_hidden_states = ent_all_hidden_states + (ent_hidden_states,)

            tok_layer_outputs, ent_layer_outputs = layer_module(tok_hidden_states, tok_attention_mask, ent_hidden_states, ent_attention_mask, predict_id)
            tok_hidden_states = tok_layer_outputs[0]
            ent_hidden_states = ent_layer_outputs[0]

            if self.output_attentions:
                tok_all_attentions = tok_all_attentions + (tok_layer_outputs[1],)
                ent_all_attentions = ent_all_attentions + (ent_layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            tok_all_hidden_states = tok_all_hidden_states + (tok_hidden_states,)
            ent_all_hidden_states = ent_all_hidden_states + (ent_hidden_states,)

        tok_outputs = (tok_hidden_states,)
        ent_outputs = (ent_hidden_states,)
        if self.output_hidden_states:
            tok_outputs = tok_outputs + (tok_all_hidden_states,)
            ent_outputs = ent_outputs + (ent_all_hidden_states,)
        if self.output_attentions:
            tok_outputs = tok_outputs + (tok_all_attentions,)
            ent_outputs = ent_outputs + (ent_all_attentions,)
        return tok_outputs, ent_outputs  # last-layer hidden state, (all hidden states), (all attentions)

class TableLMSubPredictionHead(nn.Module):
    """
    only make prediction for a subset of candidates
    """
    def __init__(self, config, output_dim=None, use_bias=True):
        super(TableLMSubPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config, output_dim=output_dim)
        if use_bias:
            self.bias = nn.Embedding.from_pretrained(torch.zeros(config.ent_vocab_size, 1), freeze=False)
        else:
            self.bias = None

    def forward(self, hidden_states, candidates, candidates_embeddings, return_hidden=False):
        hidden_states = self.transform(hidden_states)
        scores = torch.matmul(hidden_states, torch.transpose(candidates_embeddings,1,2))
        if self.bias is not None:
            scores += torch.transpose(self.bias(candidates),1,2)
        if return_hidden:
            return (scores,hidden_states)
        else:
            return scores


class TableMLMHead(nn.Module):
    def __init__(self, config):
        super(TableMLMHead, self).__init__()
        self.tok_predictions = BertLMPredictionHead(config)
        self.ent_predictions = TableLMSubPredictionHead(config)

    def load_pretrained(self, checkpoint):
        state_dict = self.state_dict()
        for x in state_dict:
            if x.find('tok_predictions')!=-1:
                state_dict[x] = checkpoint['cls.'+x[4:]]
                print('load %s <- %s'%(x, 'cls.'+x[4:]))
            elif x.find('bias')==-1:
                state_dict[x] = checkpoint['cls.'+x[4:]]
                print('load %s <- %s'%(x, 'cls.'+x[4:]))
        self.load_state_dict(state_dict)

    def forward(self, tok_sequence_output, ent_sequence_output, ent_candidates, ent_candidates_embeddings):
        tok_prediction_scores = self.tok_predictions(tok_sequence_output)
        ent_prediction_scores = self.ent_predictions(ent_sequence_output, ent_candidates, ent_candidates_embeddings)
        return tok_prediction_scores, ent_prediction_scores

class TableModel(BertPreTrainedModel):
    def __init__(self, config, is_simple=False):
        super(TableModel, self).__init__(config)
        self.is_simple = is_simple
        self.config = config

        self.embeddings = TableEmbeddings(config)
        self.encoder = TableEncoderSimple(config)

        self.init_weights()

    def load_pretrained(self, checkpoint, is_bert=True):
        self.embeddings.load_pretrained(checkpoint, is_bert=is_bert)
        self.encoder.load_pretrained(checkpoint, is_bert=is_bert)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, word_embedding_matrix, ent_embedding_matrix):
        assert self.embeddings.word_embeddings.weight.shape == word_embedding_matrix.shape
        assert self.embeddings.ent_embeddings.weight.shape == ent_embedding_matrix.shape
        self.embeddings.word_embeddings.weight.data = word_embedding_matrix
        self.embeddings.ent_embeddings.weight.data = ent_embedding_matrix

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_tok = None, input_tok_type = None, input_tok_pos = None, input_tok_mask = None,
                input_ent = None, input_ent_type = None, input_ent_mask = None, ent_candidates = None):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_input_tok_mask, extended_input_ent_mask = None, None
        if input_tok_mask is not None:
            extended_input_tok_mask = input_tok_mask[:, None, :, :]
            extended_input_tok_mask = extended_input_tok_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_input_tok_mask = (1.0 - extended_input_tok_mask) * -10000.0
        if input_ent_mask is not None:
            extended_input_ent_mask = input_ent_mask[:, None, :, :]
            extended_input_ent_mask = extended_input_ent_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_input_ent_mask = (1.0 - extended_input_ent_mask) * -10000.0

        tok_embedding_output, ent_embedding_output, ent_candidates_embeddings = self.embeddings(input_tok, input_tok_type, input_tok_pos, input_ent, input_ent_type, ent_candidates) #disgard ent_pos since they are all 0
        tok_encoder_outputs, ent_encoder_outputs = self.encoder(tok_embedding_output, extended_input_tok_mask, ent_embedding_output, extended_input_ent_mask)
        tok_sequence_output = tok_encoder_outputs[0]
        ent_sequence_output = ent_encoder_outputs[0]

        tok_outputs = (tok_sequence_output, ) + tok_encoder_outputs[1:]  # add hidden_states and attentions if they are here
        ent_outputs = (ent_sequence_output, ) + ent_encoder_outputs[1:]
        return tok_outputs, ent_outputs, ent_candidates_embeddings  # sequence_output, (hidden_states), (attentions)

class HybridTableModel(BertPreTrainedModel):
    def __init__(self, config, is_simple=False):
        super(HybridTableModel, self).__init__(config)
        self.is_simple = is_simple
        self.config = config

        self.embeddings = TableHybridEmbeddings(config)
        self.encoder = TableEncoderSimple(config)

        self.init_weights()

        self.tok_outputs_cache = {}

    def load_pretrained(self, checkpoint, is_bert=True):
        self.embeddings.load_pretrained(checkpoint, is_bert=is_bert)
        self.encoder.load_pretrained(checkpoint, is_bert=is_bert)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, word_embedding_matrix, ent_embedding_matrix):
        assert self.embeddings.word_embeddings.weight.shape == word_embedding_matrix.shape
        assert self.embeddings.ent_embeddings.weight.shape == ent_embedding_matrix.shape
        self.embeddings.word_embeddings.weight.data = word_embedding_matrix
        self.embeddings.ent_embeddings.weight.data = ent_embedding_matrix

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_tok = None, input_tok_type = None, input_tok_pos = None, input_tok_mask = None,
                input_ent_tok = None, input_ent_tok_length = None, input_ent_mask_type = None,
                input_ent = None, input_ent_type = None, input_ent_mask = None, ent_candidates = None, predict_id = None):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        if predict_id is None or predict_id not in self.tok_outputs_cache: # train or first predict
            extended_input_tok_mask = input_tok_mask[:, None, :, :]
            extended_input_tok_mask = extended_input_tok_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_input_tok_mask = (1.0 - extended_input_tok_mask) * -10000.0
        else:
            tok_outputs = self.tok_outputs_cache[predict_id]
            input_tok = None
            extended_input_tok_mask = None

        extended_input_ent_mask = None
        if input_ent_mask is not None:
            extended_input_ent_mask = input_ent_mask[:, None, :, :]
            extended_input_ent_mask = extended_input_ent_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_input_ent_mask = (1.0 - extended_input_ent_mask) * -10000.0

        tok_embedding_output, ent_embedding_output, ent_candidates_embeddings = self.embeddings(input_tok, input_tok_type, input_tok_pos, input_ent_tok, input_ent_tok_length, input_ent_mask_type, input_ent, input_ent_type, ent_candidates) #disgard ent_pos since they are all 0
        tok_encoder_outputs, ent_encoder_outputs = self.encoder(tok_embedding_output, extended_input_tok_mask, ent_embedding_output, extended_input_ent_mask, predict_id)
        
        if predict_id is None or predict_id not in self.tok_outputs_cache: # train or first predict
            tok_sequence_output = tok_encoder_outputs[0]
            tok_outputs = (tok_sequence_output, ) + tok_encoder_outputs[1:]  # add hidden_states and attentions if they are here
            if predict_id is not None: # first predict
                self.tok_outputs_cache[predict_id] = tok_outputs
        else:
            del self.tok_outputs_cache[predict_id]

        ent_sequence_output = ent_encoder_outputs[0]
        ent_outputs = (ent_sequence_output, ) + ent_encoder_outputs[1:] # add hidden_states and attentions if they are here

        return tok_outputs, ent_outputs, ent_candidates_embeddings  # sequence_output, (hidden_states), (attentions)

class TableHeaderModel(BertPreTrainedModel):
    def __init__(self, config, is_simple=False):
        super(TableHeaderModel, self).__init__(config)
        self.is_simple = is_simple
        self.config = config

        self.embeddings = TableHeaderEmbeddings(config)
        self.encoder = TableEncoderSimple(config)

        self.init_weights()

    def load_pretrained(self, checkpoint, is_bert=True):
        self.embeddings.load_pretrained(checkpoint, is_bert=is_bert)
        self.encoder.load_pretrained(checkpoint, is_bert=is_bert)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, word_embedding_matrix, header_embedding_matrix):
        assert self.embeddings.word_embeddings.weight.shape == word_embedding_matrix.shape
        assert self.embeddings.header_embeddings.weight.shape == header_embedding_matrix.shape
        self.embeddings.word_embeddings.weight.data = word_embedding_matrix
        self.embeddings.header_embeddings.weight.data = header_embedding_matrix

    def set_header_embeddings(self, header_embedding_matrix):
        assert self.embeddings.header_embeddings.weight.shape == header_embedding_matrix.shape
        self.embeddings.header_embeddings.weight.data = header_embedding_matrix

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_tok, input_tok_type, input_tok_pos,
                input_header, input_header_type, input_mask):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # pdb.set_trace()
        extended_input_mask = input_mask[:, None, :, :]
        extended_input_mask = extended_input_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_input_mask = (1.0 - extended_input_mask) * -10000.0

        tok_embedding_output, header_embedding_output = self.embeddings(input_tok, input_tok_type, input_tok_pos, input_header, input_header_type) #disgard header_pos since they are all 0
        tok_encoder_outputs, _ = self.encoder(torch.cat([tok_embedding_output, header_embedding_output], dim=1), extended_input_mask)
        tok_sequence_output = tok_encoder_outputs[0]

        tok_outputs = (tok_sequence_output, ) + tok_encoder_outputs[1:]  # add hidden_states and attheaderions if they are here
        return tok_outputs  # sequence_output, (hidden_states), (attentions)

class TableMaskedLM(BertPreTrainedModel):
    def __init__(self, config, is_simple=False):
        super(TableMaskedLM, self).__init__(config)

        self.table = TableModel(config, is_simple)
        self.cls = TableMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.tok_predictions.decoder

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        tok_output_embeddings = self.get_output_embeddings()
        tok_input_embeddings = self.table.get_input_embeddings()
        if tok_output_embeddings is not None:
            self._tie_or_clone_weights(tok_output_embeddings, tok_input_embeddings)

    def load_pretrained(self, checkpoint):
        self.table.load_pretrained(checkpoint)
        self.cls.load_pretrained(checkpoint)

    def forward(self, input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                input_ent, input_ent_type, input_ent_mask, ent_candidates,
                tok_masked_lm_labels, ent_masked_lm_labels, exclusive_ent_mask=None):
        tok_outputs, ent_outputs, ent_candidates_embeddings = self.table(input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent, input_ent_type, input_ent_mask, ent_candidates)

        tok_sequence_output = tok_outputs[0]
        ent_sequence_output = ent_outputs[0]
        tok_prediction_scores, ent_prediction_scores = self.cls(tok_sequence_output, ent_sequence_output, ent_candidates, ent_candidates_embeddings)

        tok_outputs = (tok_prediction_scores,) + tok_outputs[1:]  # Add hidden states and attention if they are here
        ent_outputs = (ent_prediction_scores,) + ent_outputs[1:]

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        # pdb.set_trace()
        if tok_masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            tok_masked_lm_loss = loss_fct(tok_prediction_scores.view(-1, self.config.vocab_size), tok_masked_lm_labels.view(-1))
            tok_outputs = (tok_masked_lm_loss,) + tok_outputs
        if ent_masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            if exclusive_ent_mask is not None:
                ent_prediction_scores.scatter_add_(2, exclusive_ent_mask, (1.0 - (exclusive_ent_mask>=1000).float()) * -10000.0)
            ent_masked_lm_loss = loss_fct(ent_prediction_scores.view(-1, self.config.max_entity_candidate), ent_masked_lm_labels.view(-1))
            ent_outputs = (ent_masked_lm_loss,) + ent_outputs
        # pdb.set_trace()
        return tok_outputs, ent_outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)

class HybridTableMaskedLM(BertPreTrainedModel):
    def __init__(self, config, is_simple=False):
        super(HybridTableMaskedLM, self).__init__(config)

        self.table = HybridTableModel(config, is_simple)
        self.cls = TableMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.tok_predictions.decoder

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        tok_output_embeddings = self.get_output_embeddings()
        tok_input_embeddings = self.table.get_input_embeddings()
        if tok_output_embeddings is not None:
            self._tie_or_clone_weights(tok_output_embeddings, tok_input_embeddings)

    def load_pretrained(self, checkpoint):
        self.table.load_pretrained(checkpoint)
        self.cls.load_pretrained(checkpoint)

    def forward(self, input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                input_ent_tok, input_ent_tok_length, input_ent_mask_type,
                input_ent, input_ent_type, input_ent_mask, ent_candidates,
                tok_masked_lm_labels=None, ent_masked_lm_labels=None, exclusive_ent_mask=None):
        tok_outputs, ent_outputs, ent_candidates_embeddings = self.table(input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent_tok, input_ent_tok_length, input_ent_mask_type, input_ent, input_ent_type, input_ent_mask, ent_candidates)

        tok_sequence_output = tok_outputs[0]
        ent_sequence_output = ent_outputs[0]
        tok_prediction_scores, ent_prediction_scores = self.cls(tok_sequence_output, ent_sequence_output, ent_candidates, ent_candidates_embeddings)

        tok_outputs = (tok_prediction_scores,) + tok_outputs  # Add hidden states and attention if they are here
        ent_outputs = (ent_prediction_scores,) + ent_outputs

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        # pdb.set_trace()
        if tok_masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            tok_masked_lm_loss = loss_fct(tok_prediction_scores.view(-1, self.config.vocab_size), tok_masked_lm_labels.view(-1))
            tok_outputs = (tok_masked_lm_loss,) + tok_outputs
        if ent_masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            if exclusive_ent_mask is not None:
                ent_prediction_scores.scatter_add_(2, exclusive_ent_mask, (1.0 - (exclusive_ent_mask>=1000).float()) * -10000.0)
            ent_prediction_scores += (ent_candidates[:,None,:]==0).float()*-10000.0
            ent_masked_lm_loss = loss_fct(ent_prediction_scores.view(-1, self.config.max_entity_candidate), ent_masked_lm_labels.view(-1))
            ent_outputs = (ent_masked_lm_loss,) + ent_outputs
        # pdb.set_trace()
        return tok_outputs, ent_outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Original implementation from: https://github.com/Mikoto10032/AutomaticWeightedLoss
    
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class AsymmetricDoubleTower(BertPreTrainedModel):
    def __init__(self, config, is_simple=False):
        super(AsymmetricDoubleTower, self).__init__(config)

        self.table = HybridTableModel(config, is_simple)
        self.table.embeddings.ent_embeddings.weight.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.use_histogram_feature = config.use_histogram_feature
        if config.use_histogram_feature:
            self.fc1 = nn.Linear(config.hidden_size + HISTOGRAM_LEN + 1, 500, bias=True)
            self.cls1 = nn.Linear(500, config.class_num, bias=True)

            self.fc2 = nn.Linear(2*config.hidden_size + HISTOGRAM_LEN + 1, 1000, bias=True)
            self.cls2 = nn.Linear(1000, config.class_num, bias=True)
        else:
            self.cls1 = nn.Linear(config.hidden_size, config.class_num, bias=True)
            self.cls2 = nn.Linear(2*config.hidden_size, config.class_num, bias=True)

        self.loss_fct1 = BCEWithLogitsLoss(reduction='none')
        self.loss_fct2 = BCEWithLogitsLoss(reduction='none')
        
        self.awl = AutomaticWeightedLoss(2)

        self.init_weights()

    def load_pretrained(self, checkpoint):
        self.table.load_pretrained(checkpoint,is_bert=False)

    def forward(self, input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                input_ent_tok, input_ent_tok_length, input_ent_type, input_ent_mask,
                column_entity_mask, column_header_mask, labels_mask, labels,
                histogram, predict_id=None):

        tok_outputs, ent_outputs, _ = self.table(input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent_tok, input_ent_tok_length, None, None, input_ent_type, input_ent_mask, None, predict_id)
        tok_sequence_output = self.dropout(tok_outputs[0])
        tok_col_output = torch.matmul(column_header_mask, tok_sequence_output)
        tok_col_output /= column_header_mask.sum(dim=-1,keepdim=True).clamp(1.0,9999.0)
        if self.use_histogram_feature:
            logits1 = torch.relu(self.fc1(torch.cat([tok_col_output, histogram], dim=-1)))
            logits1 = self.dropout(logits1)
            logits1 = self.cls1(logits1)
        else:
            logits1 = self.cls1(tok_col_output)
        
        logits2 = None
        if input_ent_tok is not None:
            ent_sequence_output = self.dropout(ent_outputs[0])
            ent_col_output = torch.matmul(column_entity_mask, ent_sequence_output)
            ent_col_output /= column_entity_mask.sum(dim=-1,keepdim=True).clamp(1.0,9999.0)
            if self.use_histogram_feature:
                logits2 = torch.relu(self.fc2(torch.cat([tok_col_output, ent_col_output, histogram], dim=-1)))
                logits2 = self.dropout(logits2)
                logits2 = self.cls2(logits2)
            else:
                logits2 = self.cls2(torch.cat([tok_col_output, ent_col_output], dim=-1))

        outputs = (logits1, logits2,) + ent_outputs + tok_outputs
        if labels is not None:
            CT_loss = self.awl(self.loss_fct1(logits1, labels), self.loss_fct2(logits2, labels))
            CT_loss = torch.sum(CT_loss.mean(dim=-1)*labels_mask)/labels_mask.sum()

            outputs = (CT_loss,) + outputs
        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)
