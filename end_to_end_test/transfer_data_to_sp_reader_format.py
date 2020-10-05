import json
import sys
from test_config import TEST_CONFIG

def read_json(fpath):
    data = json.load(open(fpath, 'r'))
    print(f'loading data from:{fpath}')
    return data

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
            # print('cur_answer: ', cur_answer)
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
    config = TEST_CONFIG()
    raw_res = read_json(config.test_input_file)
    sele_para_res = read_json(config.para_sele_predict_output_file)
    json_store_file_loc = config.para_reader_predict_input_file

    raw_squad_text = generate_data_raw_hotpot_to_squad(raw_res)
    final_raw_squad_text = generate_data_sele_para_hotpot_to_squad(sele_para_res, raw_squad_text)
    store_to_json_file(final_raw_squad_text, json_store_file_loc)

    json_store_file_loc = config.para_sp_predict_input_file
    final_raw_sp_text = generate_data_sele_para_hotpot_to_sp(sele_para_res, raw_res)
    store_to_json_file(final_raw_sp_text, json_store_file_loc)

    print('finish')