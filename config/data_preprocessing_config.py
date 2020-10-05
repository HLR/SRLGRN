class DATA_PROCESSING_CONFIG(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(DATA_PROCESSING_CONFIG, self).__init__()
        '''
        transfer data to support fact format: always choose train / dev, choose one of them
        '''

        '''
        Turing machine:
        '''
        ### train
        # self.to_sp_json_store_file_loc = '/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_train_sp.json'
        # self.to_squad_json_store_file_loc = '/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_train_squad.json'
        # self.generated_selected_paras_loc = "/tank/space/chen_zheng/data/hotpotqa/rangers/data/train_selected_paras.json"
        # self.raw_data_loc = "/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_train_v1.json"

        # ### dev
        # self.to_sp_json_store_file_loc = '/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_dev_sp.json'
        # self.to_squad_json_store_file_loc = '/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_dev_squad.json'
        # self.generated_selected_paras_loc = "/tank/space/chen_zheng/data/hotpotqa/rangers/data/dev_selected_paras.json"
        # self.raw_data_loc = "/tank/space/chen_zheng/data/hotpotqa/rangers/data/hotpot_dev_distractor_v1.json"

        '''
        avicenna machine:
        '''
        # ### train
        # self.to_sp_json_store_file_loc = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/hotpot_train_sp.json'
        # self.to_squad_json_store_file_loc = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/hotpot_train_squad.json'
        # self.generated_selected_paras_loc = "/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/train_selected_paras.json"
        # self.raw_data_loc = "/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/hotpot_train_v1.json"
        ### dev
        self.to_sp_json_store_file_loc = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/hotpot_dev_sp.json'
        self.to_squad_json_store_file_loc = '/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/hotpot_dev_squad.json'
        self.generated_selected_paras_loc = "/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/dev_selected_paras.json"
        self.raw_data_loc = "/home/hlr/shared/data/chenzheng/data/hotpotqa/rangers/data/hotpot_dev_distractor_v1.json"
