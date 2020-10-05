from model import BertForSequentialSentenceSelector


from pytorch_pretrained_bert.tokenization import BertTokenizer

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from tqdm import tqdm
import json
import os

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



import sys
sys.path.append('../')
from config.para_support_fact_config import SP_CONFIG


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





class SequentialSentenceSelector:
    def __init__(self,
                 args,
                 device):

        # if args.sequential_sentence_selector_path is None:
        #     return None
        
        print('initializing SequentialSentenceSelector...', flush=True)
        # self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)    
        # model_state_dict = torch.load(args.load_ckpt_path)
        # self.model = BertForSequentialSentenceSelector.from_pretrained(args.bert_model, state_dict=model_state_dict)

        # Prepare model
        if args.bert_model != 'bert-large-uncased-whole-word-masking':
            self.tokenizer = BertTokenizer.from_pretrained(
                args.bert_model, do_lower_case=args.do_lower_case)

            model_state_dict = torch.load(args.load_ckpt_path)
            self.model = BertForSequentialSentenceSelector.from_pretrained(args.bert_model, state_dict=model_state_dict)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-large-uncased', do_lower_case=args.do_lower_case)
            model_state_dict = torch.load(args.load_ckpt_path)
            self.model = BertForSequentialSentenceSelector.from_pretrained('bert-large-uncased', state_dict=model_state_dict)

        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.processor = DataProcessor()
        print('Done!', flush=True)

    # def convert_reader_output(self,
    #                           reader_output,
    #                           tfidf_retriever):
    #     new_output = []

    #     for data in reader_output:
    #         entry = {}
    #         entry['q_id'] = data['q_id']
    #         entry['question'] = data['question']
    #         entry['answer'] = data['answer']
    #         entry['titles'] = data['context']
    #         entry['context'] = tfidf_retriever.load_abstract_para_text(entry['titles'], keep_sentence_split = True)
    #         entry['supporting_facts'] = {t: [] for t in entry['titles']}
    #         new_output.append(entry)

    #     return new_output
        
    def predict(self,
                args):

        eval_examples = self.processor.get_train_examples(args.predict_file_path)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, args.max_sent_num, args.max_sf_num, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_masks = torch.tensor([f.input_masks for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_output_masks = torch.tensor([f.output_masks for f in eval_features], dtype=torch.float)
        all_num_sents = torch.tensor([f.num_sents for f in eval_features], dtype=torch.long)
        all_num_sfs = torch.tensor([f.num_sfs for f in eval_features], dtype=torch.long)
        all_ex_indices = torch.tensor([f.ex_index for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids,
                                  all_input_masks,
                                  all_segment_ids,
                                  all_output_masks,
                                  all_num_sents,
                                  all_num_sfs,
                                  all_ex_indices)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        pred_output = {}

        for input_ids, input_masks, segment_ids, output_masks, num_sents, num_sfs, ex_indices in tqdm(eval_dataloader, desc="Evaluating"):
            batch_max_len = input_masks.sum(dim = 2).max().item()
            batch_max_sent_num = num_sents.max().item()
            batch_max_sf_num = num_sfs.max().item()

            input_ids = input_ids[:, :batch_max_sent_num, :batch_max_len]
            input_masks = input_masks[:, :batch_max_sent_num, :batch_max_len]
            segment_ids = segment_ids[:, :batch_max_sent_num, :batch_max_len]
            output_masks = output_masks[:, :batch_max_sent_num+2, :batch_max_sent_num+1]

            output_masks[:, 1:, -1] = 1.0 # Ignore EOE in the first step

            input_ids = input_ids.to(self.device)
            input_masks = input_masks.to(self.device)
            segment_ids = segment_ids.to(self.device)
            output_masks = output_masks.to(self.device)

            examples = [eval_examples[ex_indices[i].item()] for i in range(input_ids.size(0))]

            with torch.no_grad():
                pred, prob, topk_pred, topk_prob = self.model.beam_search(input_ids, segment_ids, input_masks, output_masks, max_num_steps = args.max_sf_num+1, examples = examples, beam = args.beam_size)

            for i in range(len(pred)):
                e = examples[i]

                sfs = {}
                for p in pred[i]:
                    offset = 0
                    for j in range(len(e.titles)):
                        if p >= offset and p < offset+len(e.context[e.titles[j]]):
                            if e.titles[j] not in sfs:
                                sfs[e.titles[j]] = [[p-offset, e.context[e.titles[j]][p-offset]]]
                            else:
                                sfs[e.titles[j]].append([p-offset, e.context[e.titles[j]][p-offset]])
                            break
                        offset += len(e.context[e.titles[j]])

                # Hack
                for title in e.titles:
                    if title not in sfs and len(sfs) < 2:
                        sfs[title] = [0]


                # output = {}
                # output['q_id'] = e.guid
                # output['supporting facts'] = sfs
                # pred_output.append(output)
                '''
                chen begin
                '''
                tmp_res = []
                for k, v in sfs.items():
                    for v_li in v:
                        tmp_res.append([k, v_li[0]])
                pred_output[e.guid] = tmp_res
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        with open(args.output_dir + args.output_file, "w") as writer:
            writer.write(json.dumps(pred_output, indent=4) + "\n")

        return pred_output

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

    predict_func = SequentialSentenceSelector(sp_cfg, device)
    print('finish 1')
    predict_res = predict_func.predict(sp_cfg)
    print('finish 2')
    print(predict_res)


if __name__ == "__main__":
    main()



