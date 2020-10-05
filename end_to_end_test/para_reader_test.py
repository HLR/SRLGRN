from __future__ import absolute_import, division, print_function

import argparse
import collections
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer



if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from test_config import TEST_CONFIG

import sys
sys.path.append('../')
from para_reader.model import BertForQuestionAnsweringConfidence
from para_reader.rc_utils import convert_examples_to_features_yes_no, read_squad_examples, write_predictions_yes_no_no_empty_answer


logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch_logits"])




def main():
    reader_cfg = TEST_CONFIG()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in reader_cfg.visable_gpus)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not reader_cfg.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    
    model_state_dict = torch.load(reader_cfg.para_reader_ckpt)
    model = BertForQuestionAnsweringConfidence.from_pretrained(reader_cfg.para_reader_path, state_dict=model_state_dict)
    tokenizer = BertTokenizer.from_pretrained(reader_cfg.para_reader_path+'vocab.txt', do_lower_case=reader_cfg.do_lower_case)
    
    model.to(device)
    
    eval_examples = read_squad_examples(
        input_file=reader_cfg.para_reader_predict_input_file, is_training=False, version_2_with_negative=reader_cfg.version_2_with_negative,
        max_answer_len=reader_cfg.max_answer_length, skip_negatives=reader_cfg.skip_negatives)
    eval_features = convert_examples_to_features_yes_no(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=reader_cfg.para_reader_max_seq_length,
        doc_stride=reader_cfg.doc_stride,
        max_query_length=reader_cfg.para_reader_max_query_length,
        is_training=False)

    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", reader_cfg.predict_batch_size)

    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(
        all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=reader_cfg.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    # for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=reader_cfg.local_rank not in [-1, 0]):
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, batch_switch_logits = model(
                input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            switch_logits = batch_switch_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                            start_logits=start_logits,
                                            end_logits=end_logits,
                                            switch_logits=switch_logits))
    output_prediction_file = reader_cfg.final_output_ans_file
    output_nbest_file = os.path.join(
        reader_cfg.final_output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(
        reader_cfg.final_output_dir, "null_odds.json")
    write_predictions_yes_no_no_empty_answer(eval_examples, eval_features, all_results,
                                                reader_cfg.n_best_size, reader_cfg.max_answer_length,
                                                reader_cfg.do_lower_case, output_prediction_file,
                                                output_nbest_file, output_null_log_odds_file, reader_cfg.verbose_logging,
                                                reader_cfg.version_2_with_negative, reader_cfg.null_score_diff_threshold,
                                                reader_cfg.no_masking)
    print('finish the test!')


if __name__ == "__main__":
    main()