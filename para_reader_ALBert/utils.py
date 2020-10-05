import os
import json
import requests


from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertModel, BertTokenizer)

from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import AlbertModel, AlbertConfig, AlbertTokenizer

import sys
sys.path.append('../')
from config.para_reader_config import READER_CONFIG

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
}

def get_bert_model_from_pytorch_transformers(model_name):
    reader_cfg = READER_CONFIG()
    model_name = reader_cfg.bert_model
    config_class, model_class, tokenizer_class = MODEL_CLASSES['albert']
    config = config_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, from_tf=bool('.ckpt' in model_name), config=config)

    # tokenizer = tokenizer_class.from_pretrained(model_name)

    return model.state_dict(), None