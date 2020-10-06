from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

try:
    from model import BertForSequentialSentenceSelector
except:
    from model import BertForSequentialSentenceSelector

import json

import sys
sys.path.append('../')
from config.para_support_fact_config import SP_CONFIG

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)




class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, q, a, t, c, sf):

        self.guid = guid
        self.question = q
        self.answer = a
        self.titles = t
        self.context = c
        self.supporting_facts = sf


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_masks, segment_ids, target_ids, output_masks, num_sents, num_sfs, ex_index):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.target_ids = target_ids
        self.output_masks = output_masks
        self.num_sents = num_sents
        self.num_sfs = num_sfs

        self.ex_index = ex_index


class DataProcessor:

    def get_train_examples(self, file_name):
        return self.create_examples(json.load(open(file_name, 'r')))

    def create_examples(self, jsn):
        examples = []
        max_sent_num = 0
        for data in jsn:
            guid = data['q_id']
            question = data['question']
            titles = data['titles']
            context = data['context']  # {title: [s1, s2, ...]}
            # {title: [index1, index2, ...]}
            supporting_facts = data['supporting_facts']

            max_sent_num = max(max_sent_num, sum(
                [len(context[title]) for title in context]))

            examples.append(InputExample(
                guid, question, data['answer'], titles, context, supporting_facts))

        return examples


def convert_examples_to_features(examples, max_seq_length, max_sent_num, max_sf_num, tokenizer, train=False):
    """Loads a data file into a list of `InputBatch`s."""

    DUMMY = [0] * max_seq_length
    DUMMY_ = [0.0] * max_sent_num
    features = []
    logger.info('#### Constructing features... ####')
    for (ex_index, example) in enumerate(tqdm(examples, desc='Example')):

        tokens_q = tokenizer.tokenize(
            'Q: {} A: {}'.format(example.question, example.answer))
        tokens_q = ['[CLS]'] + tokens_q + ['[SEP]']

        input_ids = []
        input_masks = []
        segment_ids = []

        for title in example.titles:
            sents = example.context[title]
            for (i, s) in enumerate(sents):

                if len(input_ids) == max_sent_num:
                    break

                tokens_s = tokenizer.tokenize(
                    s)[:max_seq_length-len(tokens_q)-1]
                tokens_s = tokens_s + ['[SEP]']

                padding = [0] * (max_seq_length -
                                 len(tokens_s) - len(tokens_q))

                input_ids_ = tokenizer.convert_tokens_to_ids(
                    tokens_q + tokens_s)
                input_masks_ = [1] * len(input_ids_)
                segment_ids_ = [0] * len(tokens_q) + [1] * len(tokens_s)

                input_ids_ += padding
                input_ids.append(input_ids_)

                input_masks_ += padding
                input_masks.append(input_masks_)

                segment_ids_ += padding
                segment_ids.append(segment_ids_)

                assert len(input_ids_) == max_seq_length
                assert len(input_masks_) == max_seq_length
                assert len(segment_ids_) == max_seq_length

        target_ids = []
        target_offset = 0

        for title in example.titles:
            ### chen begin: add if key in the dict
            ### reason: after select two parapraghs: may be the choosed title not in the gold support fact title....
            '''
            eg. selected titles: 0:"Peggy Seeger" 1:"June Miller"
            but gold support fact title: peggy seeger, ewan MacColl
            you can found that June Miller not in the support facts, example source: 4th example (3 if start from 0)
            '''
            if title in example.supporting_facts:
                sfs = example.supporting_facts[title]
                for i in sfs:
                    if i < len(example.context[title]) and i+target_offset < len(input_ids):
                        target_ids.append(i+target_offset)
                    else:
                        logger.warning('')
                        logger.warning('Invalid annotation: {}'.format(sfs))
                        logger.warning('Invalid annotation: {}'.format(
                            example.context[title]))

                target_offset += len(example.context[title])
            ### chen end 

        assert len(input_ids) <= max_sent_num
        assert len(target_ids) <= max_sf_num

        num_sents = len(input_ids)
        num_sfs = len(target_ids)

        output_masks = [([1.0] * len(input_ids) + [0.0] * (max_sent_num -
                                                           len(input_ids) + 1)) for _ in range(max_sent_num + 2)]

        if train:

            for i in range(len(target_ids)):
                for j in range(len(target_ids)):
                    if i == j:
                        continue

                    output_masks[i][target_ids[j]] = 0.0

            for i in range(len(output_masks)):
                if i >= num_sfs+1:
                    for j in range(len(output_masks[i])):
                        output_masks[i][j] = 0.0

        else:
            for i in range(len(input_ids)):
                output_masks[i+1][i] = 0.0

        target_ids += [0] * (max_sf_num - len(target_ids))

        padding = [DUMMY] * (max_sent_num - len(input_ids))
        input_ids += padding
        input_masks += padding
        segment_ids += padding

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_masks=input_masks,
                          segment_ids=segment_ids,
                          target_ids=target_ids,
                          output_masks=output_masks,
                          num_sents=num_sents,
                          num_sfs=num_sfs,
                          ex_index=ex_index))

    logger.info('Done!')

    return features


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def main():
    sp_cfg = SP_CONFIG()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in sp_cfg.visable_gpus)
    
    cpu = torch.device('cpu')

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not sp_cfg.no_cuda else "cpu")

    # device = torch.device("cuda:1")

    n_gpu = torch.cuda.device_count()
    # n_gpu = 1
    print('------------------------------------->:' , device, n_gpu)

    sp_cfg.train_batch_size = int(
        sp_cfg.train_batch_size / sp_cfg.gradient_accumulation_steps)

    random.seed(sp_cfg.seed)
    np.random.seed(sp_cfg.seed)
    torch.manual_seed(sp_cfg.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(sp_cfg.seed)

    if os.path.exists(sp_cfg.output_dir) and os.listdir(sp_cfg.output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(sp_cfg.output_dir))
    os.makedirs(sp_cfg.output_dir, exist_ok=True)

    processor = DataProcessor()

    # Prepare model
    if sp_cfg.bert_model != 'bert-large-uncased-whole-word-masking':
        tokenizer = BertTokenizer.from_pretrained(
            sp_cfg.bert_model, do_lower_case=sp_cfg.do_lower_case)

        model = BertForSequentialSentenceSelector.from_pretrained(sp_cfg.bert_model,
                                                                  cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
    else:
        model = BertForSequentialSentenceSelector.from_pretrained('bert-large-uncased',
                                                                  cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        from utils import get_bert_model_from_pytorch_transformers

        state_dict, vocab_file = get_bert_model_from_pytorch_transformers(
            sp_cfg.bert_model)
        model.bert.load_state_dict(state_dict)
        tokenizer = BertTokenizer.from_pretrained(
            vocab_file, do_lower_case=sp_cfg.do_lower_case)

        logger.info(
            'The {} model is successfully loaded!'.format(sp_cfg.bert_model))

    model.to(device)
    print('----------------------------------->:', model)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    POSITIVE = 1.0
    NEGATIVE = 0.0

    # Load training examples
    train_examples = None
    num_train_steps = None
    train_examples = processor.get_train_examples(sp_cfg.train_file_path)
    train_features = convert_examples_to_features(
        train_examples, sp_cfg.max_seq_length, sp_cfg.max_sent_num, sp_cfg.max_sf_num, tokenizer, train=True)

    num_train_steps = int(
        len(train_features) / sp_cfg.train_batch_size / sp_cfg.gradient_accumulation_steps * sp_cfg.num_train_epochs)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=sp_cfg.learning_rate,
                         warmup=sp_cfg.warmup_proportion,
                         t_total=t_total,
                         max_grad_norm=1.0)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", sp_cfg.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long)
    all_input_masks = torch.tensor(
        [f.input_masks for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor(
        [f.target_ids for f in train_features], dtype=torch.long)
    all_output_masks = torch.tensor(
        [f.output_masks for f in train_features], dtype=torch.float)
    all_num_sents = torch.tensor(
        [f.num_sents for f in train_features], dtype=torch.long)
    all_num_sfs = torch.tensor(
        [f.num_sfs for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids,
                               all_input_masks,
                               all_segment_ids,
                               all_target_ids,
                               all_output_masks,
                               all_num_sents,
                               all_num_sfs)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=sp_cfg.train_batch_size)

    model.train()
    epc = 0
    for _ in trange(int(sp_cfg.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            input_masks = batch[1]
            batch_max_len = input_masks.sum(dim=2).max().item()

            target_ids = batch[3]

            num_sents = batch[5]
            batch_max_sent_num = num_sents.max().item()

            num_sfs = batch[6]
            batch_max_sf_num = num_sfs.max().item()

            output_masks_cpu = (batch[4])[
                :, :batch_max_sf_num+1, :batch_max_sent_num+1]

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, segment_ids, _, output_masks, __, ___ = batch
            B = input_ids.size(0)

            input_ids = input_ids[:, :batch_max_sent_num, :batch_max_len]
            input_masks = input_masks[:, :batch_max_sent_num, :batch_max_len]
            segment_ids = segment_ids[:, :batch_max_sent_num, :batch_max_len]
            target_ids = target_ids[:, :batch_max_sf_num]
            # 1 for EOE
            output_masks = output_masks[:,
                                        :batch_max_sf_num+1, :batch_max_sent_num+1]

            target = torch.FloatTensor(output_masks.size()).fill_(
                NEGATIVE)  # (B, NUM_STEPS, |S|+1) <- 1 for EOE
            for i in range(B):
                output_masks[i, :num_sfs[i]+1, -1] = 1.0  # for EOE
                target[i, num_sfs[i], -1].fill_(POSITIVE)

                for j in range(num_sfs[i].item()):
                    target[i, j, target_ids[i, j]].fill_(POSITIVE)
            target = target.to(device)

            loss = model(input_ids, segment_ids, input_masks,
                         output_masks, target, target_ids, batch_max_sf_num)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if sp_cfg.gradient_accumulation_steps > 1:
                loss = loss / sp_cfg.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            nb_tr_examples += B
            nb_tr_steps += 1
            if (step + 1) % sp_cfg.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = sp_cfg.learning_rate * \
                    warmup_linear(global_step/t_total, sp_cfg.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(
            sp_cfg.output_dir, "pytorch_model_"+str(epc+1)+".bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        epc += 1


if __name__ == "__main__":
    main()
