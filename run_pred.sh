
####################################################################################
# Goal: scripts to generate the final output answer of hotpotqa task.
# Author: Chen Zheng
# School: Michigan State University
####################################################################################


####################################################################################
# First step: paragraph selection model.
# Goal: extract two or three most relevant paragraphs.
# Selection: If predict_batch_size == 4, then the gpu memory is  MiB 
####################################################################################
# cd end_to_end_test/
# python para_sele_test.py
# cd ../

python para_sele/select_paras.py    \
 --input_path="input.json"    \
 --output_path="tmp_dir/intermediate_dir/para_sele_predict_file.json"  \
 --ckpt_path="ckpt/para_sele/para_select_model.bin" 

####################################################################################
# Second step: paragraph selection result ---- squad format.
#              paragraph selection result ---- support fact format.
####################################################################################
cd end_to_end_test/
python transfer_data_to_sp_reader_format.py

####################################################################################
# third step: generate span answer / yes / no;
#             generate support fact answer.
# Supporting fact: If predict_batch_size == 4, then the gpu memory is 7669 MiB 
# Answer: If predict_batch_size == 4, then the gpu memory is 3709 MiB 
####################################################################################

python para_sp_test.py  
python para_reader_test.py 


####################################################################################
# fourth step: generate the final output answer, which is a json file
# we can use this file to meature our performance
# Work folder: rangers/evaluate_accuracy
####################################################################################

cd ../
python end_to_end_test/merge_sp_and_reader_res.py tmp_dir/res/ans.json tmp_dir/res/sp.json pred.json

python ../rangers/evaluate_accuracy/hotpot_evaluate_v1.py pred.json input.json
