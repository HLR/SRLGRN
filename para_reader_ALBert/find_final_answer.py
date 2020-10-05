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

from utils import get_bert_model_from_pytorch_transformers

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from model import BertForQuestionAnsweringConfidence

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import AlbertTokenizer

from rc_utils import convert_examples_to_features_yes_no, read_squad_examples, write_predictions_yes_no_no_empty_answer

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import sys
sys.path.append('../')
from config.para_reader_config import READER_CONFIG

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch_logits"])




def main():
    reader_cfg = READER_CONFIG()
    CONFIG_NAME = 'config.json'
    WEIGHTS_NAME = 'pytorch_model.bin'

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in reader_cfg.visable_gpus)

    if reader_cfg.local_rank == -1 or reader_cfg.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not reader_cfg.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(reader_cfg.local_rank)
        device = torch.device("cuda", reader_cfg.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if reader_cfg.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(reader_cfg.local_rank != -1), reader_cfg.fp16))

    if reader_cfg.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            reader_cfg.gradient_accumulation_steps))

    reader_cfg.train_batch_size = reader_cfg.train_batch_size // reader_cfg.gradient_accumulation_steps

    random.seed(reader_cfg.seed)
    np.random.seed(reader_cfg.seed)
    torch.manual_seed(reader_cfg.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(reader_cfg.seed)

    if not reader_cfg.do_train and not reader_cfg.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if reader_cfg.do_train:
        if not reader_cfg.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if reader_cfg.do_predict:
        if not reader_cfg.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if not os.path.exists(reader_cfg.output_dir):
        os.makedirs(reader_cfg.output_dir)

    if reader_cfg.do_predict and (reader_cfg.local_rank == -1 or torch.distributed.get_rank() == 0):

        load_res = pickle.load(open( os.path.join(reader_cfg.output_dir, "chen_results_predictions.pkl"), "rb" ))
        # logger.info("***** Running final prediction *****")
        # logger.info("  example of pickle load logit res = %s", load_res[0])

        load_eval_examples = pickle.load(open( os.path.join(reader_cfg.output_dir, "chen_eval_examples.pkl"), "rb" ))
        logger.info("  example of pickle load load_eval_examples = %s", load_eval_examples[0])

        load_eval_features = pickle.load(open( os.path.join(reader_cfg.output_dir, "chen_eval_features.pkl"), "rb" ))
        logger.info("  example of pickle load load_eval_features = %s", load_eval_features[0])
        

        output_prediction_file = os.path.join(
            reader_cfg.output_dir, "predictions.json")
        output_nbest_file = os.path.join(
            reader_cfg.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(
            reader_cfg.output_dir, "null_odds.json")
        # write_predictions_yes_no_no_empty_answer(eval_examples, eval_features, load_res,
        #                                          reader_cfg.n_best_size, reader_cfg.max_answer_length,
        #                                          reader_cfg.do_lower_case, output_prediction_file,
        #                                          output_nbest_file, output_null_log_odds_file, reader_cfg.verbose_logging,
        #                                          reader_cfg.version_2_with_negative, reader_cfg.null_score_diff_threshold,
        #                                          reader_cfg.no_masking)
        write_predictions_yes_no_no_empty_answer(load_eval_examples, load_eval_features, load_res,
                                                 reader_cfg.n_best_size, reader_cfg.max_answer_length,
                                                 reader_cfg.do_lower_case, output_prediction_file,
                                                 output_nbest_file, output_null_log_odds_file, reader_cfg.verbose_logging,
                                                 reader_cfg.version_2_with_negative, reader_cfg.null_score_diff_threshold,
                                                 reader_cfg.no_masking)


if __name__ == "__main__":
    main()