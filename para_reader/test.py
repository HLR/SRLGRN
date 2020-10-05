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

from model import BertForQuestionAnsweringConfidence

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

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

    if reader_cfg.do_predict:
        if not reader_cfg.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    # if os.path.exists(reader_cfg.output_dir) and os.listdir(reader_cfg.output_dir) and reader_cfg.do_train:
    #     raise ValueError(
    #         "Output directory () already exists and is not empty.")
    # if not os.path.exists(reader_cfg.output_dir):
    #     os.makedirs(reader_cfg.output_dir)

    
    model_state_dict = torch.load(reader_cfg.load_ckpt_path)
    model = BertForQuestionAnsweringConfidence.from_pretrained(reader_cfg.output_dir, state_dict=model_state_dict)
    tokenizer = BertTokenizer.from_pretrained(reader_cfg.output_dir+'vocab.txt', do_lower_case=reader_cfg.do_lower_case)

    # for name, param in model.named_parameters():
    #     print(name) # name, param.data

    
    # import sys
    # sys.exit()
    
    model.to(device)
    if reader_cfg.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)



    if reader_cfg.do_predict and (reader_cfg.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_squad_examples(
            input_file=reader_cfg.predict_file, is_training=False, version_2_with_negative=reader_cfg.version_2_with_negative,
            max_answer_len=reader_cfg.max_answer_length, skip_negatives=reader_cfg.skip_negatives)
        eval_features = convert_examples_to_features_yes_no(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=reader_cfg.max_seq_length,
            doc_stride=reader_cfg.doc_stride,
            max_query_length=reader_cfg.max_query_length,
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
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=reader_cfg.local_rank not in [-1, 0]):
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
        output_prediction_file = os.path.join(
            reader_cfg.output_dir, "predictions.json")
        output_nbest_file = os.path.join(
            reader_cfg.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(
            reader_cfg.output_dir, "null_odds.json")
        write_predictions_yes_no_no_empty_answer(eval_examples, eval_features, all_results,
                                                 reader_cfg.n_best_size, reader_cfg.max_answer_length,
                                                 reader_cfg.do_lower_case, output_prediction_file,
                                                 output_nbest_file, output_null_log_odds_file, reader_cfg.verbose_logging,
                                                 reader_cfg.version_2_with_negative, reader_cfg.null_score_diff_threshold,
                                                 reader_cfg.no_masking)


if __name__ == "__main__":
    main()