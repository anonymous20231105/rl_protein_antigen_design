import random
import time

import gym
import numpy as np

from src import utils_read_fasta, utils_esmfold_plddt, test_mean_ddg


def generate_obs_from_str(obs_str, ac_table, obs_len):
    obs = []
    for i in obs_str:
        # print(self.ac_table.index(i))
        try:
            num = ac_table.index(i)
            obs = obs + [num]
            # obs.extend(obs_num)
        except ValueError:
            if i != " ":
                print("i: ", i)
                print("obs_str: ", obs_str)
    while len(obs) > obs_len:
        del obs[0]
    while len(obs) < obs_len:
        obs = [0] + obs
    return obs


def generate_one_hot_obs_from_str(obs_str, ac_table, obs_len):
    obs = []
    for i in obs_str:
        # print(self.ac_table.index(i))
        try:
            num = ac_table.index(i)
            obs_num = [-1] * len(ac_table)
            obs_num[num] = 1
            obs = obs + obs_num
            # obs.extend(obs_num)
        except ValueError:
            if i != " ":
                print("i: ", i)
                print("obs_str: ", obs_str)
    while len(obs) > obs_len * len(ac_table):
        del obs[0]
    while len(obs) < obs_len * len(ac_table):
        obs = [0] + obs
    return obs


class ProteinEnv:
    def __init__(self, obs_len=100, pred_rate=0.7, render_flag=False, min_length=50, max_length=150):
        self.ac_table = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "K", "M", "N", "O", "P", "Q", "R", "S",
                         "T", "U", "V", "W", "X", "Y", "Z"]
        high = np.array([1] * obs_len * len(self.ac_table))  # [len(self.ac_table) - 1] * obs_len
        low = np.array([0] * obs_len * len(self.ac_table))
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.multi_branch_obs_dim = [[obs_len, len(self.ac_table)]]

        self.action_space = gym.spaces.Discrete(len(self.ac_table))

        # load dataset
        self.true_seq = None
        data_mode = "dataset"  # dataset list
        if data_mode == "dataset":
            seq_datasets = utils_read_fasta.read_dataset_from_fasta()
        elif data_mode == "list":
            seq_datasets = ["MASMAKKDVIELEGTVSEALPNAMFKVKLENGHEILCHISGKLRMNFIRILEGDKVNVELSPYDLTRGRITWRKKLEHHHHHH"]
        else:
            raise
        random.shuffle(seq_datasets)

        # split dataset
        data_num = len(seq_datasets)
        print("dataset_num: ", data_num)
        if data_num > 1:
            test_num = max(int(data_num * 0.2), 1)
            train_num = data_num - test_num
            self.train_seq_dataset = seq_datasets[0: train_num]
            self.test_seq_dataset = seq_datasets[train_num: train_num + test_num]
        else:
            self.train_seq_dataset = seq_datasets
            self.test_seq_dataset = seq_datasets

        self.obs_len = obs_len
        self.pred_rate = pred_rate
        self.render_flag = render_flag
        self.true_length = None
        self.min_length = min_length
        self.max_length = max_length

        print("Creating esmfold model...")
        self.esm_model = utils_esmfold_plddt.create_model(chunk_size=2048)

    def obs_once(self, sequence):
        obs_str = sequence[-self.obs_len:]
        obs = generate_one_hot_obs_from_str(obs_str, self.ac_table, self.obs_len)
        return obs

    def select_data(self, dataset_mode):
        true_length = 999
        while true_length > self.max_length or true_length < self.min_length:
            if dataset_mode == "Train":
                if len(self.train_seq_dataset) > 1:
                    seq_num = random.randint(0, len(self.train_seq_dataset) - 1)
                else:
                    seq_num = 0
                self.true_seq = self.train_seq_dataset[seq_num]
            elif dataset_mode == "Test":
                if len(self.test_seq_dataset) > 1:
                    seq_num = random.randint(0, len(self.test_seq_dataset) - 1)
                else:
                    seq_num = 0
                self.true_seq = self.test_seq_dataset[seq_num]
            else:
                raise
            true_length = len(self.true_seq)
        self.true_length = true_length

    def reset(self, dataset_mode="Train"):
        self.select_data(dataset_mode)
        pred_len = int(self.true_length * self.pred_rate)
        assert pred_len <= self.true_length
        self.current_seq = self.true_seq[:-pred_len]
        obs = self.obs_once(self.current_seq)
        return obs

    def calc_reward(self, current_seq):
        reward_info = dict()
        if len(current_seq) >= self.true_length:
            # start_time = time.time()
            plddt_entire, plddt_specify, plddt_design, _ = utils_esmfold_plddt.esmfold_sequence_2_plddt(self.esm_model, current_seq, 1-self.pred_rate)
            # print("infer time: ", time.time()-start_time)

            # plddt_entire
            # (plddt_entire + plddt_specify) / 2
            # (plddt_entire * plddt_specify)**0.5
            # plddt_specify
            reward = plddt_specify  # -abs(plddt_specify - 80)  #  - plddt_design

            reward_info["plddt_specify"] = plddt_specify
            reward_info["plddt_design"] = plddt_design
            reward_info["plddt_entire"] = plddt_entire
        else:
            reward = 0
            reward_info["plddt_specify"] = None
            reward_info["plddt_design"] = None
            reward_info["plddt_entire"] = None
        return reward, reward_info

    def render(self):
        # cvui
        print("--------------------------------------------------------")
        print("self.action_str: ", self.action_str)
        print("self.current_seq: ", self.current_seq)
        print("self.true_seq:    ", self.true_seq)
        print("--------------------------------------------------------")
        pass

    def step(self, action):
        self.action_str = self.ac_table[action]
        self.current_seq = self.current_seq + self.action_str
        current_length = len(self.current_seq)

        obs = self.obs_once(self.current_seq)

        reward, reward_info = self.calc_reward(self.current_seq)

        if current_length >= self.true_length:
            done = True
        else:
            done = False

        extra_info = reward_info
        extra_info["true_length"] = self.true_length

        if self.render_flag:
            self.render()

        return obs, reward, done, extra_info


class ProteinEnvDouble(ProteinEnv):
    def __init__(self, obs_len=100, pred_rate=0.7, render_flag=False, min_length=50, max_length=150):
        super().__init__(obs_len, pred_rate, render_flag, min_length, max_length)
        high = np.array([1] * obs_len * 2 * len(self.ac_table))
        low = np.array([0] * obs_len * 2 * len(self.ac_table))
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.multi_branch_obs_dim = [[obs_len, len(self.ac_table)], [obs_len, len(self.ac_table)]]
        self.action_space = gym.spaces.Discrete(len(self.ac_table)*len(self.ac_table))

    def obs_once(self, sequence):
        obs_str = sequence[-self.obs_len:]
        obs_forward = generate_one_hot_obs_from_str(obs_str, self.ac_table, self.obs_len)

        sequence_rev = sequence[::-1]
        obs_str_rev = sequence_rev[-self.obs_len:]
        obs_rev = generate_one_hot_obs_from_str(obs_str_rev, self.ac_table, self.obs_len)

        obs = obs_forward + obs_rev

        return obs

    def reset(self, dataset_mode="Train"):
        self.select_data(dataset_mode)
        self.current_seq = self.true_seq[int(self.true_length*(self.pred_rate/2)):int(self.true_length*(1-self.pred_rate/2))]
        obs = self.obs_once(self.current_seq)
        return obs

    def calc_reward(self, current_seq):
        reward_info = dict()
        if len(current_seq) >= self.true_length:
            # start_time = time.time()
            plddt_entire, _, _, plddt_acid_list = utils_esmfold_plddt.esmfold_sequence_2_plddt(self.esm_model, current_seq, 1-self.pred_rate)
            assert plddt_entire is not None
            assert plddt_entire != 0

            start_index = int(self.pred_rate/2 * self.true_length)
            end_index = int(-self.pred_rate/2 * self.true_length)

            plddt_design1 = np.mean(plddt_acid_list[:start_index])
            plddt_specify = np.mean(plddt_acid_list[start_index:end_index])
            plddt_design2 = np.mean(plddt_acid_list[end_index:])

            # - (plddt_design1 + plddt_design2) / 2 / 2
            reward = plddt_entire  # -abs(plddt_specify - 80)  #  - plddt_design plddt_specify
            assert reward != 0

            reward_info["plddt_specify"] = plddt_specify
            reward_info["plddt_design"] = (plddt_design1 + plddt_design2) / 2
            reward_info["plddt_entire"] = plddt_entire
        else:
            reward = 0
            reward_info["plddt_specify"] = None
            reward_info["plddt_design"] = None
            reward_info["plddt_entire"] = None
        return reward, reward_info

    def step(self, action):
        action_forward = action % len(self.ac_table)
        action_backward = action // len(self.ac_table)

        self.action_str_forward = self.ac_table[action_forward]
        if self.action_str_forward == self.current_seq[-1]:  #  == self.current_seq[-2]
            action_forward = random.randint(0, len(self.ac_table)-1)
            self.action_str_forward = self.ac_table[action_forward]

        self.action_str_backward = self.ac_table[action_backward]
        if self.action_str_backward == self.current_seq[0]:  #  == self.current_seq[1]
            action_backward = random.randint(0, len(self.ac_table)-1)
            self.action_str_backward = self.ac_table[action_backward]

        self.current_seq = self.action_str_backward + self.current_seq + self.action_str_forward
        current_length = len(self.current_seq)

        obs = self.obs_once(self.current_seq)

        reward, reward_info = self.calc_reward(self.current_seq)

        if current_length >= self.true_length:
            done = True
        else:
            done = False

        extra_info = reward_info
        extra_info["true_length"] = self.true_length

        if self.render_flag:
            self.render()

        return obs, reward, done, extra_info

    def render(self):
        print("--------------------------------------------------------")
        print("self.action_str_forward: ", self.action_str_forward)
        print("self.action_str_backward: ", self.action_str_backward)
        print("self.current_seq: ", self.current_seq)
        print("self.true_seq:    ", self.true_seq)
        print("--------------------------------------------------------")
        pass


MEAN_RESULTS = ["AYRYNYEYDY",
                "AYLYDYYYDY",
                "ARYYYYGYDY",
                "ADYYYYYFYYYYSYDY",
                "YYYDGYDYYYYYTYAD",
                "AGDLYYYYDY",
                "AYYYYDYLDY",
                "YYYTYYYYDY",
                "AYFAYYYYYLY",
                "ADYYYYYYDYYDY",
                "AYYYYDY",
                "YYYDYYGYYYDYDF",
                "AYMVYYTSSYVDYYD",
                "SYYGYDY",
                "ASVRYTRYYYYDF",
                "AYRWSSGSSRASYDDSYYD",
                "AYTYRSSDYYSYYYLR",
                "SYYYYYYYYYGDY",
                "SSTYYSSYYSYTLMD",
                "AYYYYDY",
                "AYYYDDY",
                "AIGYLGY",
                "TYYYSDY",
                "AYRYNDY",
                "AYGDYYYYDY",
                "ARYRYYYYYGDD",
                "AYRYYYYYYNYDY",
                "VDRGYTYFYDYYDL",
                "AYYSDYYSYYYRYFDY",
                "AAYYYYIYRYDRYYDY",
                "AYYYYYYDYGDY",
                "AYYSFRGYYYDY",
                "AYYEYYGDY",
                "YRYDDYSGTSSDY",
                "TRGYYYGRYYYDY",
                "ARTEGDAFADSTYLYSYYYD",
                "YARDYSTTAYYYGTTSDRSDT",
                "ASFYYYYYYGY",
                "AGLYYDYYYDY",
                "AYYYYRYYTYYGDY",
                "AIYSDYYYDY",
                "ARRYYDYYRYYYYY",
                "YYSYYFYIYLYSDSDDY",
                "DAATSRSATGSEEGRRADASATR",
                "SESATASSYSTSTNYSYYRD",
                "AYGYDYYYDTYDYY",
                "AYYDSYYYYYDY",
                "AYYRYYGGAYRYDY",
                "AYYYRYYVDYDY",
                "AGYYYRLYDG",
                "AYYSYYYYDSYDDADY",
                "SYYDRSLYRYYVRYAD",
                "ALYRYAYDL",
                ]
MEAN_RESULTS_NPY = np.load("../data/raw/sabdab_cdrh3_mean_100.npy")


class ProteinEnvDDG:
    def obs_once(self, sequence):
        obs_str_current = sequence[-self.obs_len :]
        obs_current = generate_one_hot_obs_from_str(obs_str_current, self.ac_table, self.obs_len)

        obs_str_heavy = self.heavy_chain[-100 :]
        obs_heavy = generate_one_hot_obs_from_str(obs_str_heavy, self.ac_table, 100)

        obs_str_light = self.light_chain[-100 :]
        obs_light = generate_one_hot_obs_from_str(obs_str_light, self.ac_table, 100)

        obs_str_antig = self.antig_chain[-100 :]
        obs_antig = generate_one_hot_obs_from_str(obs_str_antig, self.ac_table, 100)

        obs_index = [0] * 53
        obs_index[self.origin_index] = 1

        # + obs_light + obs_antig  + obs_heavy  # , self.cdrh3_len-len(sequence)
        obs = obs_current + obs_index + [len(sequence)]  #

        return obs

    def reset(self, dataset_mode="Train", set_index=None):
        self.current_seq = ""
        if set_index is None:
            assert len(self.origin_cplx) == 53
            if dataset_mode == "Train":
                self.origin_index = random.randint(0, 42)
            elif dataset_mode == "Test":
                self.origin_index = random.randint(43, 52)
            else:
                raise
        else:
            self.origin_index = set_index
        self.origin_cplx_temp = self.origin_cplx[self.origin_index]
        self.id = self.origin_cplx_temp.get_id()

        cdrh3_pos = self.origin_cplx_temp.cdr_pos['CDR-H3']
        self.cdrh3_len = cdrh3_pos[1] - cdrh3_pos[0] + 1

        self.heavy_chain = self.origin_cplx_temp.peptides[self.origin_cplx_temp.heavy_chain].seq
        self.cdrh3_seq = self.heavy_chain[cdrh3_pos[0]:cdrh3_pos[1]+1]

        if self.origin_cplx_temp.light_chain != "":
            self.light_chain = self.origin_cplx_temp.peptides[self.origin_cplx_temp.light_chain].seq
        else:
            self.light_chain = " "
        if len(self.origin_cplx_temp.antigen_chains) > 0:
            self.antig_chain = self.origin_cplx_temp.peptides[self.origin_cplx_temp.antigen_chains[0]].seq
        else:
            self.antig_chain = " "

        # x = np.load("../data/raw/true_x/true_x_" + self.id + ".npy")
        # self.cdrh3_len = x.shape[0]

        obs = self.obs_once(self.current_seq)

        self.mean_result = MEAN_RESULTS_NPY[self.origin_index]

        return obs

    def __init__(self, obs_len=30, render_flag=False, dataset_num=None, plddt_flag=False):
        self.ac_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                         'Y']
        self.multi_branch_obs_dim = [[obs_len, len(self.ac_table)], 53, 1]
        mbd = self.multi_branch_obs_dim
        total_obs_dim = mbd[0][0] * mbd[0][1] + mbd[1] + mbd[2]
        high = np.array([999] * total_obs_dim)
        low = np.array([-999] * total_obs_dim)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.action_space = gym.spaces.Discrete(len(self.ac_table) + 1)  # + 2
        self.current_seq = ""
        self.obs_len = obs_len
        self.render_flag = render_flag

        mode = "111"
        test_set = "../data/interim/mean/summaries/skempi_all.json"
        dataset, _ = test_mean_ddg.prepare_efficient_mc_att(mode, test_set)
        self.origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]
        if dataset_num is not None:
            self.origin_cplx = self.origin_cplx[:dataset_num]
        for i in self.origin_cplx:
            print("i.get_id(): ", i.get_id())

        self.random_index = random.randint(0, 100000)

        print("Creating esmfold model...")
        if plddt_flag:
            self.esm_model = utils_esmfold_plddt.create_model(chunk_size=2048)
        self.plddt_flag = plddt_flag

        self.plddt_cdrh3_origin_list = [68.16221885521885,
                                        77.96496015712682,
                                        77.96496015712682,
                                        65.04844632034633,
                                        65.04844632034633,
                                        79.01781677890011,
                                        77.1187219416386,
                                        77.96496015712682,
                                        67.76192431457432,
                                        57.96360858585859,
                                        71.89540277777778,
                                        63.39446878121878,
                                        57.92785227272727,
                                        74.17284523809523,
                                        68.1876178150553,
                                        51.15042803030303,
                                        68.51552265512265,
                                        65.47987977994228,
                                        54.81000144300145,
                                        71.1246369047619,
                                        76.6767103174603,
                                        71.71945436507936,
                                        70.30186111111111,
                                        70.46230555555556,
                                        69.96860125060124,
                                        62.17395454545454,
                                        68.91574386724386,
                                        62.76550399600399,
                                        60.798908874458874,
                                        59.109222607022616,
                                        66.95702331759149,
                                        65.2726948051948,
                                        63.333914772727276,
                                        65.47987977994228,
                                        65.47987977994228,
                                        47.7927732778917,
                                        48.95108553391053,
                                        68.26482727272727,
                                        64.40463181818181,
                                        66.93457847707849,
                                        71.52364790764791,
                                        68.81950427350426,
                                        57.42226884920635,
                                        49.599666519086966,
                                        61.063927565124935,
                                        54.32569180819179,
                                        70.25195015085923,
                                        59.82094239094239,
                                        65.78385350255805,
                                        74.04442171717172,
                                        54.556076118326125,
                                        63.640816041366044,
                                        63.4677178030303,
                                        ]
        assert len(self.plddt_cdrh3_origin_list) == 53

        print("Env created...")

    def calc_ddg_reward(self):
        v_worst = 5
        v_best = -10
        if self.current_seq == self.cdrh3_seq:
            score_ddp = 0
            self.new_cplx = self.origin_cplx_temp
        else:
            new_cplx = test_mean_ddg.seq_2_cplx(self.current_seq, self.origin_cplx_temp, self.id)
            pdb_path = "../data/interim/mean_ddg_rl/temp_" + str(self.origin_index) + "_" + self.id + "_" + str(self.random_index) + ".pdb"   # os.path.join(res_dir, new_cplx.get_id() + f'_{n}.pdb')
            new_cplx.to_pdb(pdb_path)
            score_ddp = test_mean_ddg.pdb_2_ddg(pdb_path,
                                                origin_cplx_path='/home/gemhou/Study/hj_mean_demo/summaries/ckpt/mean_CDR3_111/version_1/checkpoint/ita_results/ita_results/original/' + self.id + '.pdb')
            if self.render_flag:
                print("score_ddp: ", score_ddp)

            self.new_cplx = new_cplx
        reward_ddg = (score_ddp - v_worst) / (v_best - v_worst)
        return reward_ddg, score_ddp

    def calc_aar_reward(self):
        ours_seq = self.current_seq
        data_seq = self.cdrh3_seq
        hit = 0
        for a, b in zip(ours_seq, data_seq):
            if a == b:
                hit += 1
        aar = hit * 1.0 / len(ours_seq)
        reward_aar = aar
        return reward_aar, aar

    def calc_plddt_reward(self):
        heavy_seq = self.new_cplx.peptides[self.new_cplx.heavy_chain].seq
        plddt_heavy, _, _, plddt_acid_list = utils_esmfold_plddt.esmfold_sequence_2_plddt(self.esm_model, heavy_seq)
        cdrh3_pos = self.new_cplx.cdr_pos['CDR-H3']
        plddt_cdrh3 = plddt_acid_list[cdrh3_pos[0]:cdrh3_pos[1]]
        plddt_cdrh3 = np.mean(plddt_cdrh3)
        plddt_cdrh3_origin = self.plddt_cdrh3_origin_list[self.origin_index]
        delta_plddt_cdrh3 = plddt_cdrh3 - plddt_cdrh3_origin
        v_worst = -20
        v_best = 20
        reward_plddt = (delta_plddt_cdrh3 - v_worst) / (v_best - v_worst)
        return reward_plddt, plddt_heavy, plddt_cdrh3, delta_plddt_cdrh3

    def calc_reward(self):

        reward_ddg, ddg = self.calc_ddg_reward()
        reward_aar, aar = self.calc_aar_reward()

        if self.plddt_flag:
            reward_plddt, plddt_heavy, plddt_cdrh3, delta_plddt_cdrh3 = self.calc_plddt_reward()
            reward = reward_ddg * (1/3) + reward_aar * (1/3) + reward_plddt * (1/3)  # (reward_ddg + reward_plddt) / 2           reward_ddg
        else:
            plddt_heavy, plddt_cdrh3, delta_plddt_cdrh3 = None, None, None
            reward = reward_ddg * 0.5 + reward_aar * 0.5

        return reward, ddg, plddt_heavy, plddt_cdrh3, delta_plddt_cdrh3, aar

    def step(self, action):
        if action < len(self.ac_table):
            self.action_str = self.ac_table[action]
            if len(self.current_seq) > 1 and self.action_str == self.current_seq[-1] == self.current_seq[-2]:
                action = random.randint(0, len(self.ac_table) - 1)
                self.action_str = self.ac_table[action]
        elif action == len(self.ac_table):
            self.action_str = self.mean_result[len(self.current_seq)]
        elif action == len(self.ac_table) + 1:
            self.action_str = self.cdrh3_seq[len(self.current_seq)]
        else:
            raise

        # debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.action_str = self.cdrh3_seq[len(self.current_seq)]
        # self.action_str = self.mean_result[len(self.current_seq)]  # debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.current_seq = self.current_seq + self.action_str
        if self.render_flag:
            print("action: ", action)
            print("self.current_seq: ", self.current_seq)

        obs = self.obs_once(self.current_seq)
        if len(self.current_seq) >= self.cdrh3_len:
            reward, ddg, plddt_heavy, plddt_cdrh3, delta_plddt_cdrh3, aar = self.calc_reward()
            done = True
            extra_info = {}
            extra_info["ddg"] = ddg
            extra_info["plddt_heavy"] = plddt_heavy
            extra_info["plddt_cdrh3"] = plddt_cdrh3
            extra_info["delta_plddt_cdrh3"] = delta_plddt_cdrh3
            extra_info["aar"] = aar
        else:
            done = False
            reward = 0
            extra_info = None
        return obs, reward, done, extra_info
