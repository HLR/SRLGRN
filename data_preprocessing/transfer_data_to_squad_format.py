import json
import sys
sys.path.append('../')
from config.data_preprocessing_config import DATA_PROCESSING_CONFIG

'''
The goal of this .py file is to change the hotpotqa data format to SQUAD data format.
The next step is reading comprehension through graph sentence/entity embedding + BERT token embedding
'''
def read_json(fpath):
    data = json.load(open(fpath, 'r'))
    print(f'loading data from:{fpath}')
    # keys = list(data.keys())
    return data


''' SQUAD data sample key format
# res['paragraphs'] = [{'qas':[ { 'question': '', 
#                                 'is_impossible': 'false',  ## yes/no is true, otherwise false
#                                 "answers": [{"text": "", 
#                                             "answer_start": -1
#                                             }], 
#                             } ], 
#                     'context': ''
#                     }]
'''
def generate_data_raw_hotpot_to_squad(input_json):
    final_res = dict()
    final_res["version"] = "v2.0"
    final_res["data"] = []
    
    for ith_data_sample in input_json: ### dicts in list:
        res = dict()
        res['paragraphs'] = [{'qas':[ { 'question': ith_data_sample['question'], 
                                # 'is_impossible': 'false', 
                                'is_impossible': True, 
                                "answers": [{"text": ith_data_sample['answer'], 
                                            "answer_start": -1
                                            }], 
                                "id": ith_data_sample['_id'],
                                'supporting_facts': ith_data_sample['supporting_facts'],
                                'type': ith_data_sample['type'], 
                                'level': ith_data_sample['level']
                            } ], 
                    'context': ''
                    }]
        res['title'] = ''
        # print(res['paragraphs'])

        final_res["data"].append(res) ## append this data sample dict to list
    return final_res



def generate_data_sele_para_hotpot_to_squad(sele_para_json, raw_squad_json):
    for i in range(len(raw_squad_json['data'])): ### dicts in list:
        ### find id in data sample of raw_squad_json:   ith_data_sample['paragraphs'][0]['qas'][0]['id']
        cur_id = raw_squad_json['data'][i]['paragraphs'][0]['qas'][0]['id']
        count = 0
        if cur_id in sele_para_json.keys():
            tmp_title_and_context = sele_para_json[cur_id] 
            title = ''
            context = ''
            for li in tmp_title_and_context:
                ith_title = li[0]
                ith_context = ''.join(li[1])
                if title == '':
                    title += ith_title ## i think one day i need to change " " to "tab"
                else:
                    title += '\t' + ith_title ## i think one day i need to change " " to "tab"
                context += ith_context
            raw_squad_json['data'][i]['title'] = title
            raw_squad_json['data'][i]['paragraphs'][0]['context'] = context
            

            ## change the answer type and span answer
            cur_answer = raw_squad_json['data'][i]['paragraphs'][0]['qas'][0]['answers'][0]["text"]
            print('cur_answer: ', cur_answer)
            if cur_answer != 'yes' and cur_answer != 'no':
                index = context.find(cur_answer)
                # print('index position: ', index)
                raw_squad_json['data'][i]['paragraphs'][0]['qas'][0]['answers'][0]["answer_start"] = index
                raw_squad_json['data'][i]['paragraphs'][0]['qas'][0]["is_impossible"] = False
            else:
                raw_squad_json['data'][i]['paragraphs'][0]['qas'][0]['answers'][0]["answer_start"] = -1
                raw_squad_json['data'][i]['paragraphs'][0]['qas'][0]["is_impossible"] = True

    
        else:
            count+=1
    print('the count of missing id: ',count)

    return raw_squad_json

def store_to_json_file(squad_text, out_file):
    with open(out_file, 'w') as f:
        json.dump(squad_text, f)


if __name__ == '__main__':
    # ### dev
    # json_store_file_loc = '/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_dev_squad.json'
    # sele_para_res = read_json("/tank/space/chen_zheng/data/hotpotqa/rangers/data/dev_selected_paras.json")
    # raw_res = read_json("/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_dev_distractor_v1.json")

    # ### train
    # json_store_file_loc = '/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_train_squad.json'
    # sele_para_res = read_json("/tank/space/chen_zheng/data/hotpotqa/rangers/data/train_selected_paras.json")
    # raw_res = read_json("/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_train_v1.1.json")

    #### config:
    config = DATA_PROCESSING_CONFIG()
    json_store_file_loc = config.to_squad_json_store_file_loc
    sele_para_res = read_json(config.generated_selected_paras_loc)
    raw_res = read_json(config.raw_data_loc)

    raw_squad_text = generate_data_raw_hotpot_to_squad(raw_res)
    final_raw_squad_text = generate_data_sele_para_hotpot_to_squad(sele_para_res, raw_squad_text)
    store_to_json_file(final_raw_squad_text, json_store_file_loc)

    print('finish')