from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import random
import pandas
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import sys
sys.path.append('../')
from config.para_sele_config import PARA_CONFIG
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):

    def get_train_examples(self, data_dir, ith_para_sele, second_step):
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "hotpot_ss_train.csv")))
        # train_path = os.path.join(data_dir, "hotpot_ss_train.csv")
        ### chen begin
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "hotpot_train_v1.1.json")))
        train_path = os.path.join(data_dir, "hotpot_train_v1.1.json")
        ### chen end
        return self._create_examples(
            # pandas.read_csv(train_path), set_type='train')
            ### chen begin
            pandas.read_json(train_path), set_type='train', 
            ith_para_sele=ith_para_sele,
            second_step = second_step)
            ### chen end

    def get_dev_examples(self, data_dir, ith_para_sele, second_step):
        # dev_path = os.path.join(data_dir, "hotpot_ss_dev.csv")
        ### chen begin
        dev_path = os.path.join(data_dir, "hotpot_dev_distractor_v1.json")
        ### chen end
        return self._create_examples(
            # pandas.read_csv(dev_path), set_type='dev')
            ### chen begin
            pandas.read_json(dev_path), set_type='dev',
            ith_para_sele=ith_para_sele,
            second_step = second_step)
            ### chen end

    def get_labels(self):
        return [False, True]


    '''
    _id                                          5a7a06935542990198eaf050
    answer                                              Arthur's Magazine
    context             [[Radio City (Indian radio station), [Radio Ci...
    level                                                          medium
    question            Which magazine was started first Arthur's Maga...
    supporting_facts       [[Arthur's Magazine, 0], [First for Women, 0]]
    type                                                       comparison
    Name: 0, dtype: object
    '''
    def _create_examples(self, df, set_type, ith_para_sele, second_step):
        examples = []
        count = 0
        # print('<<<<<<<<<<<<<<size: ',df.count)
        for (cur_id, row) in df.iterrows(): ### sometimes, the number of the context is less than 10 paragraphs, so I add "wrong answer" after the empty paragraphs
            ### chen begin
            for i in range(10):
                if i > len(row['context'])-1:
                    guid = "%s-%s-%s-%s" % (set_type, cur_id, ith_para_sele, i)
                    if ith_para_sele == 0:
                        text_a = row['question']
                    else:
                        text_a = row['question'] + second_step[count]
                    text_b = '{} {}'.format('wrong answer', 'wrong answer') ## 0: title 1: paragraph content
                    label = 0
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                else:
                    guid = "%s-%s-%s-%s" % (set_type, cur_id, ith_para_sele, i)
                    if ith_para_sele == 0:
                        text_a = row['question']
                    else:
                        text_a = row['question'] + ' ' + second_step[count]
                    # text_b = '{} {}'.format(row['context'], row['title'])
                    text_b = '{} {}'.format(row['context'][i][0], ''.join(row['context'][i][1])) ## 0: title 1: paragraph content
                    # label = row['label']
                    if row['context'][i][0] == row['supporting_facts'][ith_para_sele][0] and ith_para_sele == 0: ## if context_i title == supporting_facts_ith_para_sele. since we have 2 times of para_sele, so we choose different ith_para_sele
                        label = 1
                        second_step.append(row['context'][i][0]+' '+''.join(row['context'][i][1]))
                    elif row['context'][i][0] == row['supporting_facts'][ith_para_sele][0] and ith_para_sele != 0:
                        label = 1
                    else:
                        label = 0
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            count += 1
        ### chen end
        return (examples, second_step)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, verbose=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # Feature ids
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a ) + 2) + [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Mask and Paddings
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5 and verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def evaluate(do_pred=False, pred_path=None):
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", para_cfg.eval_batch_size)
    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=para_cfg.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluation"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        predictions.append(logits)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'global_step': global_step}

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        # writer.write("%s = %s\n" % (key, str(result[key])))

    if do_pred and pred_path is not None:
        logger.info("***** Writting Predictions ******")
        logits0 = np.concatenate(predictions, axis=0)[:, 0]
        logits1 = np.concatenate(predictions, axis=0)[:, 1]
        ground_truth = [fea.label_id for fea in eval_features]
        pandas.DataFrame({'logits0': logits0, 'logits1': logits1, 'label': ground_truth}).to_csv(pred_path)
    return eval_loss, eval_accuracy


if __name__ == "__main__":
    para_cfg = PARA_CONFIG()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in para_cfg.visable_gpus)

    # Set GPU Issue
    device = torch.device("cuda" if torch.cuda.is_available() and not para_cfg.no_cuda else "cpu")
    para_cfg.train_batch_size = int(para_cfg.train_batch_size / para_cfg.gradient_accumulation_steps)


    # Set Seeds
    random.seed(para_cfg.seed)
    np.random.seed(para_cfg.seed)
    torch.manual_seed(para_cfg.seed)

    os.makedirs(para_cfg.output_dir, exist_ok=True)
    os.makedirs(para_cfg.ckpt_dir, exist_ok=True)

    num_labels = 2
    processor = DataProcessor()
    label_list = processor.get_labels()

    # Prepare Tokenizer
    tokenizer = BertTokenizer.from_pretrained(para_cfg.bert_model, do_lower_case=para_cfg.do_lower_case)
    logger.info("finish tokenizer!")

    # Prepare Model
    model = BertForSequenceClassification.from_pretrained(para_cfg.bert_model, num_labels=num_labels)
    for param in model.bert.parameters():
        param.requires_grad = False
    model.to(device)
    logger.info("finish load model!")

    # Prepare Optimizer
    train_examples, train_second_step = [], []
    num_train_steps = []
    if para_cfg.do_train:
        ## chen add ith_para_sele
        for ith_time in range(para_cfg.para_sele_times):
            # train_examples += (processor.get_train_examples(para_cfg.data_dir, ith_time))
            train_tuple = processor.get_train_examples(para_cfg.data_dir, ith_time, train_second_step)
            train_examples +=  train_tuple[0]
            train_second_step = train_tuple[1]

        num_train_steps = int(
            len(train_examples) / para_cfg.train_batch_size / para_cfg.gradient_accumulation_steps * para_cfg.num_train_epochs)



    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=para_cfg.learning_rate,
                         warmup=para_cfg.warmup_proportion,
                         t_total=t_total)

    # Training
    ### load dev data sample
    eval_examples, eval_second_step = [], []
    for ith_time in range(para_cfg.para_sele_times):
        eval_tuple = processor.get_dev_examples(para_cfg.data_dir, ith_time, eval_second_step)
        eval_examples +=  eval_tuple[0]
        eval_second_step = eval_tuple[1]
    eval_features = convert_examples_to_features(
        eval_examples, label_list, para_cfg.max_seq_length, tokenizer, verbose=False)

    global_step = 0
    if para_cfg.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, para_cfg.max_seq_length, tokenizer, verbose=True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", para_cfg.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=para_cfg.train_batch_size)

        model.train()
        for epc in trange(int(para_cfg.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids.to(device), segment_ids.to(device), input_mask.to(device), label_ids.to(device))
                if para_cfg.gradient_accumulation_steps > 1:
                    loss = loss / para_cfg.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % para_cfg.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = para_cfg.learning_rate * warmup_linear(global_step / t_total, para_cfg.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(para_cfg.ckpt_dir, "{}_epoch{}_ckpt.bin".format(para_cfg.name, epc))
            output_prediction_file = os.path.join(para_cfg.output_dir, "{}_epoch{}_pred.csv".format(para_cfg.name, epc))
            torch.save(model_to_save.state_dict(), output_model_file)
            evaluate(do_pred=True, pred_path=output_prediction_file)
    # Load a trained model that you have fine-tuned
    # model_state_dict = torch.load(output_model_file)
    # model = BertForSequenceClassification.from_pretrained(para_cfg.bert_model, state_dict=model_state_dict)
    # model.cuda()

