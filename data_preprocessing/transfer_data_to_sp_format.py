import json
import sys
sys.path.append('../')
from config.data_preprocessing_config import DATA_PROCESSING_CONFIG

'''
The goal of this .py file is to change the hotpotqa data format to support face data format.
The next step is support face predicting task through graph sentence/entity embedding + BERT token embedding
'''
def read_json(fpath):
    data = json.load(open(fpath, 'r'))
    print(f'loading data from:{fpath}')
    # keys = list(data[0].keys())
    # print(keys)
    return data

### keys in dict: ['supporting_facts', 'level', 'question', 'context', 'answer', '_id', 'type']
### supporting fact format: ['supporting_facts', 'level', 'question', 'context', 'answer', 'q_id', 'titles', 'type']
def generate_data_sele_para_hotpot_to_sp(sele_para_json, raw_json):
    for i in range(len(raw_json)): ### dicts in list:
        cur_id = raw_json[i]['_id']


        count = 0
        if cur_id in sele_para_json.keys(): ### in the raw text, support facts and contexts are list, and there is no key of titles.
            tmp_title_and_context = sele_para_json[cur_id][0: 2] ### all titles and contexts, and choose the first two.
            titles = []
            contexts = dict()
            for li in tmp_title_and_context:
                titles.append(li[0])
                contexts[li[0]] = li[1]

            supporting_facts = dict()
            for j in range(len(raw_json[i]['supporting_facts'])):
                key = raw_json[i]['supporting_facts'][j][0] ## 0 is key, 1 is value
                value = raw_json[i]['supporting_facts'][j][1]
                if key in supporting_facts.keys():
                    tmp = supporting_facts[key]
                    tmp.append(value)
                    supporting_facts[key] = tmp
                else:
                    supporting_facts[key] = [value]
            
            ### finally combine together: 
            ### ['supporting_facts', 'level', 'question', 'context', 'answer', '_id', 'type']
            ### supporting fact format: ['supporting_facts', 'level', 'question', 'context', 'answer', 'q_id', 'titles', 'type']
            ### remove some unnecessary keys
            del raw_json[i]['context']
            del raw_json[i]['_id']
            del raw_json[i]['supporting_facts']
            ### add some necessary keys
            raw_json[i]['q_id'] = cur_id
            raw_json[i]['titles'] = titles
            raw_json[i]['context'] = contexts
            raw_json[i]['supporting_facts'] = supporting_facts

        else:
            count+=1
    print('the count of missing id: ', count)

    return raw_json

def store_to_json_file(squad_text, out_file):
    with open(out_file, 'w') as f:
        json.dump(squad_text, f)


if __name__ == '__main__':
    ### dev turing
    # json_store_file_loc = '/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_dev_sp.json'
    # sele_para_res = read_json("/tank/space/chen_zheng/data/hotpotqa/rangers/data/dev_selected_paras.json")
    # raw_res = read_json("/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_dev_distractor_v1.json")

    ### train turing
    # json_store_file_loc = '/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_train_sp.json'
    # sele_para_res = read_json("/tank/space/chen_zheng/data/hotpotqa/rangers/data/train_selected_paras.json")
    # raw_res = read_json("/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_train_v1.1.json")

    ### dev avicenna
    # json_store_file_loc = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/hotpot_dev_sp.json'
    # sele_para_res = read_json("/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/dev_selected_paras.json")
    # raw_res = read_json("/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/hotpot_dev_distractor_v1.json")

    ### train avicenna
    # json_store_file_loc = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/hotpot_train_sp.json'
    # sele_para_res = read_json("/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/train_selected_paras.json")
    # raw_res = read_json("/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/hotpot_train_v1.1.json")

    ### config
    config = DATA_PROCESSING_CONFIG()
    json_store_file_loc = config.to_sp_json_store_file_loc
    sele_para_res = read_json(config.generated_selected_paras_loc)
    raw_res = read_json(config.raw_data_loc)

    final_raw_sp_text = generate_data_sele_para_hotpot_to_sp(sele_para_res, raw_res)
    store_to_json_file(final_raw_sp_text, json_store_file_loc)
    print('finish')