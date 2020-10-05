class PARA_CONFIG(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(PARA_CONFIG, self).__init__()

        self.para_sele_times = 2 ## if choose 2 paragraphs , then it is 2 ## important: train and val use 2, and test use 1
                                 ## since the second paragraph's train and val based on the previous gold question + title1 + paragraph1, 
                                 ## but in test, all of the result don't have gold title and paragraph for first turn, then we need to predicate two times;
                                 ## 1. predict title1 and paragraph1 based on question, 
                                 ## 2. redicate title2 and paragraph2 based on question + predicated title1 + predicated title2
        self.temp_dir_bert = '/tank/space/chen_zheng/data/bert_pretrained_files/'


        # --input_path=${INPUT_FILE} \
        # --output_path=work_dir/${OUTPUT_DIR}/selected_paras.json \
        # --ckpt_path=work_dir/para_select_model.bin \
        # --split=${OUTPUT_DIR}
        ### use for select_paras begin
        self.input_path = '/tank/space/chen_zheng/data/hotpotqa/data/raw_data/hotpot_dev_distractor_v1.json'
        self.output_path = '/tank/space/chen_zheng/data/hotpotqa/data/ranger_output/para_sele/predictions/selected_paras.json'
        self.ckpt_path = '/tank/space/chen_zheng/data/hotpotqa/data/ranger_output/ckpt/para_sele_ckpt/second_time_training/Evaluation_epoch2_ckpt.bin'
        self.split = 'dev'
        self.bert_use_which_gpu = 2
        ### use for select_paras end

        ### use for training begin
        self.name = 'Evaluation'
        # self.data_dir = '/home/hlr/shared/data/chenzheng/data/hotpotqa/data/raw_data' ## The input data dir. Should contain the .tsv files (or other data files) for the task.
        # self.bert_model = 'bert-base-cased' ## bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.
        # self.output_dir = '/home/hlr/shared/data/chenzheng/data/hotpotqa/data/ranger_output/para_sele/predictions' ## The output directory where the model predictions and checkpoints will be written.
        # self.ckpt_dir = '/home/hlr/shared/data/chenzheng/data/hotpotqa/data/ranger_output/ckpt/para_sele_ckpt/second_time_training' ## The output directory where the model checkpoints will be written.
        self.data_dir = '/tank/space/chen_zheng/data/hotpotqa/data/raw_data' ## The input data dir. Should contain the .tsv files (or other data files) for the task.
        self.bert_model = 'bert-base-uncased' ## bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.
        self.output_dir = '/tank/space/chen_zheng/data/hotpotqa/data/ranger_output/para_sele/predictions' ## The output directory where the model predictions and checkpoints will be written.
        self.ckpt_dir = '/tank/space/chen_zheng/data/hotpotqa/data/ranger_output/ckpt/para_sele_ckpt/second_time_training' ## The output directory where the model checkpoints will be written.
        self.max_seq_length = 384
        self.do_train = True
        self.do_eval = False
        self.do_lower_case = False ## Set this flag if you are using an uncased model.
        self.train_batch_size = 8
        self.eval_batch_size = 16
        self.learning_rate = 5e-5
        self.num_train_epochs = 3.0 ## float
        self.warmup_proportion = 0.1 ## float
        self.no_cuda = False
        self.local_rank = -1 ## local_rank for distributed training on gpus
        self.seed = 42
        self.gradient_accumulation_steps = 1 ## Number of updates steps to accumulate before performing a backward/update pass.
        self.fp16 = False ## Whether to use 16-bit float precision instead of 32-bit
        self.loss_scale = 0 ## 
