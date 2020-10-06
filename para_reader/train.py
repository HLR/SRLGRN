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

    if os.path.exists(reader_cfg.output_dir) and os.listdir(reader_cfg.output_dir) and reader_cfg.do_train:
        raise ValueError(
            "Output directory () already exists and is not empty.")
    if not os.path.exists(reader_cfg.output_dir):
        os.makedirs(reader_cfg.output_dir)

    train_examples = None
    num_train_optimization_steps = None
    if reader_cfg.do_train:
        train_examples = read_squad_examples(
            input_file=reader_cfg.train_file, is_training=True, version_2_with_negative=reader_cfg.version_2_with_negative,
            max_answer_len=reader_cfg.max_answer_len, skip_negatives=reader_cfg.skip_negatives)
        print('---------->len(train_examples), ', len(train_examples))
        print(train_examples[0])
        num_train_optimization_steps = int(
            len(train_examples) / reader_cfg.train_batch_size / reader_cfg.gradient_accumulation_steps) * reader_cfg.num_train_epochs
        if reader_cfg.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model and tokenizer
    if reader_cfg.bert_model != 'bert-large-uncased-whole-word-masking':
        tokenizer = BertTokenizer.from_pretrained(
            reader_cfg.bert_model, do_lower_case=reader_cfg.do_lower_case)

        model = BertForQuestionAnsweringConfidence.from_pretrained(reader_cfg.bert_model,
                                                                   cache_dir=os.path.join(
                                                                       str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(reader_cfg.local_rank)),
                                                                   num_labels=4,
                                                                   no_masking=reader_cfg.no_masking,
                                                                   lambda_scale=reader_cfg.lambda_scale)
    else:
        model = BertForQuestionAnsweringConfidence.from_pretrained('bert-large-uncased',
                                                                   cache_dir=os.path.join(
                                                                       str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(reader_cfg.local_rank)),
                                                                   num_labels=4,
                                                                   no_masking=reader_cfg.no_masking,
                                                                   lambda_scale=reader_cfg.lambda_scale)

        ####  Chen: the model has some problem in here
        from utils import get_bert_model_from_pytorch_transformers
        state_dict, vocab_file = get_bert_model_from_pytorch_transformers(
            reader_cfg.bert_model)
        model.bert.load_state_dict(state_dict)
        tokenizer = BertTokenizer.from_pretrained(
            vocab_file, do_lower_case=reader_cfg.do_lower_case)

        print(model)
        import sys
        sys.exit()

        print('The', reader_cfg.bert_model, 'model is successfully loaded!', flush=True)

    if reader_cfg.fp16:
        model.half()
    
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

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if reader_cfg.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=reader_cfg.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if reader_cfg.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=reader_cfg.loss_scale)
    else:
        if reader_cfg.do_train:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=reader_cfg.learning_rate,
                                 warmup=reader_cfg.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    global_step = 0
    if reader_cfg.do_train:
        # print('-----------------------------------chen come to here!!!')
        cached_train_features_file = reader_cfg.train_file + '_{0}_{1}_{2}_{3}'.format(
            list(filter(None, reader_cfg.bert_model.split('/'))).pop(), str(reader_cfg.max_seq_length), str(reader_cfg.doc_stride), str(reader_cfg.max_query_length))
        print('-----------------------------------chen file: ', cached_train_features_file)
        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features_yes_no(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=reader_cfg.max_seq_length,
                doc_stride=reader_cfg.doc_stride,
                max_query_length=reader_cfg.max_query_length,
                is_training=True)
            if reader_cfg.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info(
                    "  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)

        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", reader_cfg.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor(
            [f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor(
            [f.end_position for f in train_features], dtype=torch.long)
        all_switches = torch.tensor(
            [f.switch for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions, all_switches)
        if reader_cfg.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=reader_cfg.train_batch_size)

        if reader_cfg.save_gran is not None:
            save_chunk, save_start = reader_cfg.save_gran.split(',')
            save_chunk = num_train_optimization_steps // int(save_chunk)
            save_start = int(save_start)

        model.train()
        for _ in trange(int(reader_cfg.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=reader_cfg.local_rank not in [-1, 0])):
                if n_gpu == 1:
                    # multi-gpu does scattering it-self
                    batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, start_positions, end_positions, switches = batch
                loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                             start_positions=start_positions, end_positions=end_positions, switch_list=switches)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if reader_cfg.gradient_accumulation_steps > 1:
                    loss = loss / reader_cfg.gradient_accumulation_steps

                if reader_cfg.fp16:
                    optimizer.backward(loss)
                else:
                    loss.mean().backward()
                if (step + 1) % reader_cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if reader_cfg.save_gran is not None:
                        if (global_step % save_chunk == 0) and (global_step // save_chunk >= save_start):
                            logger.info('Saving a checkpoint....')
                            output_dir_per_epoch = os.path.join(
                                reader_cfg.output_dir, str(global_step) + 'steps')
                            os.makedirs(output_dir_per_epoch)

                            # Save a trained model, configuration and tokenizer
                            model_to_save = model.module if hasattr(
                                model, 'module') else model  # Only save the model it-self

                            # If we save using the predefined names, we can load using
                            # `from_pretrained`
                            output_model_file = os.path.join(
                                output_dir_per_epoch, WEIGHTS_NAME)
                            output_config_file = os.path.join(
                                output_dir_per_epoch, CONFIG_NAME)

                            torch.save(model_to_save.state_dict(),
                                       output_model_file)
                            model_to_save.config.to_json_file(
                                output_config_file)
                            tokenizer.save_vocabulary(output_dir_per_epoch)
                            logger.info('Done')

    # if reader_cfg.do_train and (reader_cfg.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Save a trained model, configuration and tokenizer
    #     model_to_save = model.module if hasattr(
    #         model, 'module') else model  # Only save the model it-self

    #     # If we save using the predefined names, we can load using
    #     # `from_pretrained`
    #     output_model_file = os.path.join(reader_cfg.output_dir, WEIGHTS_NAME)
    #     output_config_file = os.path.join(reader_cfg.output_dir, CONFIG_NAME)

    #     torch.save(model_to_save.state_dict(), output_model_file)
    #     model_to_save.config.to_json_file(output_config_file)
    #     tokenizer.save_vocabulary(reader_cfg.output_dir)

    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = BertForQuestionAnsweringConfidence.from_pretrained(
    #         reader_cfg.output_dir,  num_labels=4, no_masking=reader_cfg.no_masking)
    #     tokenizer = BertTokenizer.from_pretrained(
    #         reader_cfg.output_dir, do_lower_case=reader_cfg.do_lower_case)

    if reader_cfg.do_train is False and reader_cfg.do_predict is True:
        model = BertForQuestionAnsweringConfidence.from_pretrained(
            reader_cfg.output_dir,  num_labels=4, no_masking=reader_cfg.no_masking)
        tokenizer = BertTokenizer.from_pretrained(
            reader_cfg.output_dir, do_lower_case=reader_cfg.do_lower_case)

    elif reader_cfg.do_train is True and reader_cfg.do_predict is True:
        model = BertForQuestionAnsweringConfidence.from_pretrained(
            reader_cfg.output_dir,  num_labels=4, no_masking=reader_cfg.no_masking)
        tokenizer = BertTokenizer.from_pretrained(
            reader_cfg.output_dir, do_lower_case=reader_cfg.do_lower_case)

    else:
        model = BertForQuestionAnsweringConfidence.from_pretrained(
            reader_cfg.bert_model,  num_labels=4, no_masking=reader_cfg.no_masking, lambda_scale=reader_cfg.lambda_scale)

    model.to(device)

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