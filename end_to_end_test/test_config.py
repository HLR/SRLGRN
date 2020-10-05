### lambda machine
class TEST_CONFIG(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(TEST_CONFIG, self).__init__()

        ####################################################################
        ### Hyper-parameters settings.
        ####################################################################
        self.visable_gpus = 0,
        # self.para_sele_bert_model = 'bert-base-cased'
        self.para_sele_bert_model = 'bert-large-uncased'
        self.para_sp_bert_model = 'bert-large-uncased'      ## bert-base-cased, bert-base-uncased, bert-large-cased, bert-large-uncased
        self.para_reader_bert_model = 'bert-large-uncased'  ## albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2
        self.sele_do_lower_case = True #False
        self.do_lower_case = True                           ## Set this flag if you are using an uncased model. Whether to lower case the input text. True for uncased models, False for cased models.
        self.no_cuda = False                                ## Whether not to use CUDA when available
        self.seed = 42                                      ## random seed for initialization
        self.predict_batch_size = 16                         ## Total batch size for predictions.

        ####################################################################
        ### input_file, intermediate_file, and output_file settings.
        ####################################################################
        self.test_input_file = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/intermediate_dir/hotpot_dev_distractor_v1.json'
        self.intermediate_dir = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/intermediate_dir/'          ## store all of the intermediate files, such as para_sele_res, generated data preprocessing files, blabla...
        self.final_output_dir = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/res/'                                                     ## store the final prediction.json file.
        self.final_output_sp_file = self.final_output_dir + '/sp.json'
        self.final_output_ans_file = self.final_output_dir + '/ans.json'
        self.final_output_joint_file = self.final_output_dir + '/joint.json'
        self.para_sele_predict_output_file = self.intermediate_dir + 'para_sele_predict_file_3.json'                 ## selection-format json file for evaluation.
        self.para_sp_predict_input_file = self.intermediate_dir + 'para_sp_predict_input_file3.json'                ## supporting_fact-format json file for evaluation.
        self.para_reader_predict_input_file = self.intermediate_dir + 'para_reader_predict_input_file3.json'        ## reader-format json file for evaluation.

        ####################################################################
        ### pretrained ckpt settings.
        ####################################################################
        # self.para_sele_path = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/ckpt/para_sele/'
        # self.para_sele_ckpt = self.para_sele_path + 'para_select_model.bin'
        self.para_sele_path = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/ckpt/para_sele_large_5_17/'
        self.para_sele_ckpt = self.para_sele_path + 'Evaluation_epoch2_ckpt.bin'
        self.para_sp_path = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/ckpt/para_sp/'
        self.para_sp_ckpt = self.para_sp_path + 'pytorch_model.bin'
        self.para_reader_path = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/ckpt/para_reader/'
        self.para_reader_ckpt = self.para_reader_path + 'pytorch_model.bin'

        ####################################################################
        ### max length settings.
        ####################################################################
        self.para_sele_max_query_length = 64
        self.para_sp_max_query_length = 64 
        self.para_reader_max_query_length = 64 

        self.para_sele_max_seq_length = 256 # 128
        self.para_sp_max_seq_length = 256
        self.para_reader_max_seq_length = 384           ## The maximum total input sequence length after WordPiece tokenization. Sequences "
                                                        ## "longer than this will be truncated, and sequences shorter than this will be padded."

        ####################################################################
        ### para_sele specific settings.
        ####################################################################


        ####################################################################
        ### para_reader specific settings.
        ####################################################################
        self.doc_stride = 128               ## When splitting up a long document into chunks, how much stride to take between chunks.
        self.n_best_size = 20               ## The total number of n-best predictions to generate in the nbest_predictions.json output file.
        self.max_answer_length = 30         ## The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.
        self.verbose_logging = True         ## If true, all of the warnings related to data processing will be printed.
                                            ## A number of warnings are expected for a normal SQuAD evaluation.

        # self.version_2_with_negative = False ## If true, the SQuAD examples contain some that do not have an answer.
        self.version_2_with_negative = True ## If true, the SQuAD examples contain some that do not have an answer.
        self.null_score_diff_threshold = 0.0 ## If null_score - best_non_null is greater than the threshold predict null.
        self.no_masking = False             ## If true, we do not mask the span loss for no-answer examples.
        self.skip_negatives = False         ## If true, we skip negative examples during training; this is mainly for ablation.
        self.max_answer_len = 1000000       ## maximum length of answer tokens (might be set to 5 for Natural Questions!)
        self.lambda_scale = 1.0             ## If you would like to change the two losses, please change the lambda scale.

        ####################################################################
        ### para_sp specific settings.
        ####################################################################
        self.beam_size = 8
        self.max_sent_num = 30
        self.max_sf_num = 15
