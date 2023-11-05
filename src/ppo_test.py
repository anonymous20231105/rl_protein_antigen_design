import os
import random
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from src.spinup.utils.mpi_tools import mpi_fork
import src.utils_ppo_core as core
import src.utils_ppo_core_cnn
from src.utils_basic import wait_gpu_cool
from src.utils_protein_env import ProteinEnv, ProteinEnvDDG, ProteinEnvDouble
from src import utils_esmfold_plddt
import analyse_dataset

DEVICE = "cpu"
GPU_TEMPERATURE_LIMIT = 58


def test_once(ac, env, random_flag=False):
    obs = env.reset(dataset_mode="Test")
    while True:
        if random_flag:
            action = env.action_space.sample()
        else:
            action, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, info = env.step(action)
        if done:
            plddt_entire = info["plddt_entire"]
            # plddt_specify = info["plddt_specify"]
            plddt_specify = 0  # todo: plddt_specify
            seq_length = info["true_length"]
            break
    return plddt_entire, plddt_specify, seq_length


def prepare_para():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--hid", type=int, default=HIDDEN)
    # parser.add_argument("--l", type=int, default=LAYER)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--exp_name", type=str, default="ppo")
    args = parser.parse_args()
    mpi_fork(args.cpu)  # run parallel code with mpi
    from src.spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    return args


def main_once():
    PT_FILE_NAME = "../data/interim/tensorboard_his/20230510_105055/ppo_para20230510_105055.pt"
    LAYER = 2  # 2 4
    HIDDEN = 64  # 64 256

    print("Loading...")
    actor_critic = src.utils_ppo_core_cnn.CNNActorCritic
    ac_kwargs = dict(hidden_sizes=[HIDDEN] * LAYER)
    env = ProteinEnvDouble(render_flag=False)
    ac = actor_critic(env.multi_branch_obs_dim, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(PT_FILE_NAME))

    print("Processing...")
    plddt_entire, plddt_specify, seq_length = test_once(ac, env)
    print("seq_length: ", seq_length)
    print("plddt_entire: ", plddt_entire)
    print("plddt_specify: ", plddt_specify)

    print("Finished...")


def main_once_step():
    PT_FILE_NAME = "../data/interim/tensorboard_important/20230512_185628/ppo_para20230512_185628.pt"
    LAYER = 2  # 2 4
    HIDDEN = 64  # 64 256
    random_flag = False  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    print("Loading...")
    actor_critic = src.utils_ppo_core_cnn.CNNActorCritic
    ac_kwargs = dict(hidden_sizes=[HIDDEN] * LAYER)
    env = ProteinEnvDouble(render_flag=False, pred_rate=0.5, min_length=150, max_length=200)  #
    ac = actor_critic(env.multi_branch_obs_dim, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(PT_FILE_NAME))

    print("Processing...")
    import shutil
    shutil.rmtree("../data/interim/esm_temp")
    os.mkdir("../data/interim/esm_temp")
    obs = env.reset(dataset_mode="Test")
    while True:
        print("len(env.current_seq): ", len(env.current_seq))
        if random_flag:
            action = env.action_space.sample()
        else:
            action, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, info = env.step(action)
        print("env.current_seq: ", env.current_seq)
        if done:
            plddt_entire = info["plddt_entire"]
            plddt_specify = info["plddt_specify"]
            plddt_design = info["plddt_design"]
            seq_length = info["true_length"]
            break
        utils_esmfold_plddt.esmfold_sequence_2_structure(env.esm_model, env.current_seq,
                                                         save_name="../data/interim/esm_temp/esm_temp_" + str(len(env.current_seq)) + ".pdb")
        wait_gpu_cool(58)
    print("seq_length: ", seq_length)
    print("plddt_entire: ", plddt_entire)
    print("plddt_specify: ", plddt_specify)
    print("plddt_design: ", plddt_design)


    print("Finished...")


def calc_plddt_and_save(sample_num):
    PT_FILE_NAME = "../data/interim/tensorboard_his/20230504_174401/ppo_para20230504_174401.pt"
    LAYER = 2  # 2 4
    HIDDEN = 64  # 64 256
    print("Loading...")
    actor_critic = src.utils_ppo_core_cnn.CNNActorCritic
    ac_kwargs = dict(hidden_sizes=[HIDDEN] * LAYER)
    env = ProteinEnv(render_flag=False)
    ac = actor_critic(env.multi_branch_obs_dim, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(PT_FILE_NAME))
    print("Processing...")
    plddt_entire_list = []
    for _ in tqdm.tqdm(range(sample_num)):
        plddt_entire, plddt_specify, seq_length = test_once(ac, env, random_flag=True)
        plddt_entire_list.append(plddt_entire)
        wait_gpu_cool(58)
    print("np.mean(plddt_entire_list): ", np.mean(plddt_entire_list))
    np.save("../data/processed/dataset_random_plddt_" + str(sample_num), plddt_entire_list)


def load_plddt_and_plot(sample_num):
    plddt_list = np.load("../data/processed/dataset_random_plddt_" + str(sample_num) + ".npy", allow_pickle=True)
    mean = np.mean(plddt_list)
    print("mean: ", mean)
    plddt_list = plddt_list.astype(int)
    _, ax = plt.subplots(figsize=(16 * 0.4, 8 * 0.4))
    density = analyse_dataset.ndarray_2_ax(ax, plddt_list, title=None, xlabel="plddt", set_x_min=20)

    ax.plot([mean, mean], [0-max(density)*0.02, max(density)*1.02], "--", c="green")

    plt.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.15)
    plt.savefig("../data/processed/fig_dataset_plddt_distribute.pdf")
    print("plt.show()...")
    plt.show()


def main_multi(sample_num=1000):
    calc_plddt_and_save(sample_num)
    load_plddt_and_plot(sample_num)

    print("Finished...")


def main_entire_specify():
    PT_FILE_NAME = "../data/interim/tensorboard_his/20230327_192433/ppo_para20230327_192433.pt"
    LAYER = 4  # 2 4
    HIDDEN = 256  # 64 256

    TEST_NUM = 1000

    args = prepare_para(LAYER, HIDDEN)

    actor_critic = core.MLPActorCritic
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)

    env = ProteinEnv(render_flag=False)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(PT_FILE_NAME))
    ac.to(DEVICE)
    file_name = "../data/processed/test_result.npy"
    # plt.xlim([0, 100])
    # plt.ylim([0, 100])
    result_list = []
    progress = tqdm.tqdm(range(TEST_NUM))
    for i in range(TEST_NUM):
        progress.update(1)
        plddt_entire, plddt_specify, seq_length = test_once(ac, env)
        # print("plddt_entire: ", plddt_entire)
        # print("plddt_specify: ", plddt_specify)
        # plt.scatter(plddt_entire, plddt_specify, c="blue")
        # plt.pause(0.0000000000001)
        result = [plddt_entire, plddt_specify, seq_length]
        result_list.append(result)
        wait_gpu_cool(GPU_TEMPERATURE_LIMIT)
        np.save(file_name, result_list)
    np.save(file_name, result_list)
    print("Finished...")
    # plt.show()


def test_one_method(TEST_NUM, ac, env, random_flag=False):
    plddt_entire_list = []
    plddt_specify_list = []
    for _ in tqdm.tqdm(range(TEST_NUM)):
        plddt_entire, plddt_specify, seq_length = test_once(ac, env, random_flag=random_flag)
        plddt_entire_list.append(plddt_entire)
        plddt_specify_list.append(plddt_specify)
        wait_gpu_cool(GPU_TEMPERATURE_LIMIT)
    print("np.mean(plddt_entire_list): ", np.mean(plddt_entire_list))
    print("np.std(plddt_entire_list): ", np.std(plddt_entire_list))
    print("np.mean(plddt_specify_list): ", np.mean(plddt_specify_list))
    print("np.std(plddt_specify_list): ", np.std(plddt_specify_list))


def test_one_rate(TEST_NUM, ac, env):
    print("env.pred_rate: ", env.pred_rate)
    print("")
    print("RL: ")
    test_one_method(TEST_NUM, ac, env, random_flag=False)
    print("")
    print("Random: ")
    test_one_method(TEST_NUM, ac, env, random_flag=True)


def main_maskRatio():
    PT_FILE_NAME = "../data/interim/tensorboard_his/20230327_192433/ppo_para20230327_192433.pt"
    LAYER = 4  # 2 4
    HIDDEN = 256  # 64 256
    TEST_NUM = 100

    args = prepare_para(LAYER, HIDDEN)

    actor_critic = core.MLPActorCritic
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)

    env = ProteinEnv(render_flag=False)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(PT_FILE_NAME))
    ac.to(DEVICE)

    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        env.pred_rate = i
        test_one_rate(TEST_NUM, ac, env)

    print("Finished...")


def main_length():
    TEST_NUM = 5000
    PT_FILE_NAME = "../data/interim/tensorboard_his/20230327_192433/ppo_para20230327_192433.pt"
    LAYER = 4  # 2 4
    HIDDEN = 256  # 64 256

    args = prepare_para(LAYER, HIDDEN)
    actor_critic = core.MLPActorCritic
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    env = ProteinEnv(render_flag=False)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(PT_FILE_NAME))
    ac.to(DEVICE)

    file_name = "../data/processed/test_result_length.npy"
    result_list = []

    for _ in tqdm.tqdm(range(TEST_NUM)):
        env.pred_rate = random.random() * 0.8 + 0.1
        plddt_entire, _, seq_length = test_once(ac, env)
        # print("----------------------------------")
        # print("seq_length: ", seq_length)
        # print("plddt_entire: ", plddt_entire)
        pred_length = int(seq_length * env.pred_rate)
        # print("pred_length: ", pred_length)

        result = [seq_length, pred_length, plddt_entire]
        result_list.append(result)
        # np.save(file_name, result_list)
    np.save(file_name, result_list)
    print("Finished...")


def test_ddg_once(ac, env, obs, random_flag):
    while True:
        if random_flag:
            action = env.action_space.sample()
            # action = 0
        else:
            action, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, info = env.step(action)
        if info is not None:
            ddg = info["ddg"]
            plddt_cdrh3 = info["plddt_cdrh3"]
            delta_plddt_cdrh3 = info["delta_plddt_cdrh3"]
            aar = info["aar"]
        if done:
            break
    return reward, env.new_cplx.peptides[env.new_cplx.heavy_chain].seq, ddg, plddt_cdrh3, delta_plddt_cdrh3, aar


def main_ddg_once():
    # "../data/interim/tensorboard_his/20230407_105728/ppo_para20230407_105728.pt"
    # "../data/interim/tensorboard/20230410_203749/ppo_para20230410_203749.pt"
    # "../data/interim/tensorboard/20230410_204444/ppo_para20230410_204444.pt"
    PT_FILE_NAME = "../data/interim/tensorboard/20230517_132357/ppo_para20230517_132357.pt"

    # actor_critic = core.MLPActorCritic
    actor_critic = src.utils_ppo_core_cnn.CNNActorCritic
    ac_kwargs = dict(hidden_sizes=[64, 64])
    env = ProteinEnvDDG(render_flag=True)
    ac = actor_critic(env.multi_branch_obs_dim, env.action_space, **ac_kwargs)
    load_content = torch.load(PT_FILE_NAME)
    load_content["pi.net_mlp_sum.4.weight"] = torch.rand(20, 64)
    load_content["pi.net_mlp_sum.4.bias"] = torch.rand(20)
    ac.load_state_dict(load_content)
    ac.to(DEVICE)

    random_flag = False

    obs = env.reset(set_index=0)
    reward, _, _, _, _, _ = test_ddg_once(ac, env, obs, random_flag)
    print("reward: ", reward)

    print("Finished...")


def main_ddg_multi():
    # random  mean:1.01 std:1.86
    # mean_seq_1  mean:-0.89 std:2.33
    # mean_seq_100  mean:-3.44 std:1.96             last 10: mean:-3.51 std:2.03
    # "../data/interim/tensorboard_his/20230407_105728/ppo_para20230407_105728.pt"  mean:-3.11  std:2.20
    # "../data/interim/tensorboard_his/20230410_203749/ppo_para20230410_203749.pt"  mean:-1.91  std:1.88
    # "../data/interim/tensorboard_his/20230410_204444/ppo_para20230410_204444.pt"  mean:-3.54  std:2.12  [64, 64]  10
    # "../data/interim/tensorboard/20230411_162649/ppo_para20230411_162649.pt"  mean:-2.65  std:1.99
    # "../data/interim/tensorboard_his_20230412/20230411_221927/ppo_para20230411_221927.pt"  mean:-4.28  std:2.30 CNN
    # "../data/interim/tensorboard/20230417_164804/ppo_para20230417_164804.pt"  mean:-4.08  std:2.15
    # "../data/interim/tensorboard_his/20230420_165102/ppo_para20230420_165102.pt"  mean:-3.98  std:2.15
    # "../data/interim/tensorboard/20230426_155458/ppo_para20230426_155458.pt"  mean:-4.58  std:2.39
    # "../data/interim/tensorboard_his/20230424_171003/ppo_para20230424_171003.pt"  -4.06 2.003
    # "../data/interim/tensorboard_his/20230427_151148/ppo_para20230427_151148.pt"
    # "../data/interim/tensorboard/20230515_142455/ppo_para20230515_142455.pt"  -3.66 2.00

    # mean_1  mean:4.2 std: 3.59

    random_flag = False
    if random_flag == False:
        PT_FILE_NAME = "../data/interim/tensorboard/20230517_153447/ppo_para20230517_153447.pt"
    else:
        PT_FILE_NAME = None

    HIDDEN_LIST = [64, 64]  # [64, 64]

    actor_critic = src.utils_ppo_core_cnn.CNNActorCritic
    ac_kwargs = dict(hidden_sizes=HIDDEN_LIST)
    env = ProteinEnvDDG(render_flag=False, plddt_flag=True)
    multi_branch_obs_dim = env.multi_branch_obs_dim
    ac = actor_critic(multi_branch_obs_dim, env.action_space, **ac_kwargs)
    if PT_FILE_NAME is not None:
        ac.load_state_dict(torch.load(PT_FILE_NAME))
    ac.to(DEVICE)

    reward_list = []
    ddg_list = []
    new_heavy_seq_list = []
    plddt_cdrh3_list = []
    delta_plddt_cdrh3_list = []
    result_list = []
    aar_list = []
    for i in tqdm.tqdm(range(53)):
        obs = env.reset(set_index=i)
        reward, new_heavy_seq, ddg, plddt_cdrh3, delta_plddt_cdrh3, aar = test_ddg_once(ac, env, obs, random_flag)
        print("")
        print("reward: ", reward)
        print("ddg: ", ddg)
        print("plddt_cdrh3: ", plddt_cdrh3)
        print("env.current_seq: ", env.current_seq)
        ddrh3_len = env.cdrh3_len
        print("ddrh3_len: ", ddrh3_len)
        print("aar: ", aar)
        aar_list.append(aar)
        reward_list.append(reward)
        ddg_list.append(ddg)
        plddt_cdrh3_list.append(plddt_cdrh3)
        new_heavy_seq_list.append(new_heavy_seq)
        delta_plddt_cdrh3_list.append(delta_plddt_cdrh3)
        result = [ddrh3_len, plddt_cdrh3, ddg]
        result_list.append(result)
        wait_gpu_cool(58)
    print("np.mean(reward_list): ", np.mean(reward_list))
    print("np.std(reward_list): ", np.std(reward_list))
    print("np.mean(ddg_list): ", np.mean(ddg_list))
    print("np.std(ddg_list): ", np.std(ddg_list))
    print("np.mean(plddt_cdrh3_list): ", np.mean(plddt_cdrh3_list))
    print("np.std(plddt_cdrh3_list): ", np.std(plddt_cdrh3_list))
    print("np.mean(delta_plddt_cdrh3_list): ", np.mean(delta_plddt_cdrh3_list))
    print("np.std(delta_plddt_cdrh3_list): ", np.std(delta_plddt_cdrh3_list))
    print("np.mean(aar_list): ", np.mean(aar_list))
    print("np.std(aar_list): ", np.std(aar_list))

    np.save("../data/raw/new_heavy_list_rl.npy", new_heavy_seq_list)
    np.save("../data/processed/ppo_test_53_ddg_result_random.npy", result_list)

    print("Finished...")


def main_ddg_multi_100():
    # random  mean:1.88 std:1.60
    # mean  mean: std:
    # "../data/interim/tensorboard_his/20230407_105728/ppo_para20230407_105728.pt"  mean:  std:
    # "../data/interim/tensorboard/20230410_203749/ppo_para20230410_203749.pt"  mean:  std:
    # "../data/interim/tensorboard/20230410_204444/ppo_para20230410_204444.pt"  mean:  std:
    PT_FILE_NAME = "../data/interim/tensorboard_his/754/ppo_para20230410_204444.pt"
    # LAYER = 2  # 2 4
    # HIDDEN = 64  # 64 256
    HIDDEN_LIST = [64, 64]

    args = prepare_para()
    actor_critic = core.MLPActorCritic
    ac_kwargs = dict(hidden_sizes=HIDDEN_LIST)
    env = ProteinEnvDDG(render_flag=False)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(PT_FILE_NAME))
    ac.to(DEVICE)

    random_flag = True

    reward_list = []
    progress = tqdm.tqdm(range(53 * 100))
    for protein_i in range(53):
        try_best_reward = -10000000000
        for try_j in range(100):
            progress.update(1)
            obs = env.reset(set_index=protein_i)
            reward, _, _, _, _, _ = test_ddg_once(ac, env, obs, random_flag)
            if reward > try_best_reward:
                try_best_reward = reward
        print("try_best_reward: ", try_best_reward)
        reward_list.append(try_best_reward)
    print("np.mean(reward_list): ", np.mean(reward_list))
    print("np.std(reward_list): ", np.std(reward_list))

    print("Finished...")


def main():
    # main_once()
    # main_once_step()
    # main_multi()
    # main_entire_specify()
    # main_maskRatio()
    # main_length()
    # main_ddg_once()
    main_ddg_multi()
    # main_ddg_multi_100()


if __name__ == "__main__":
    main()
