import numpy as np
import torch
import tqdm
from torch.optim import Adam
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import src.utils_ppo_core as core
import src.utils_ppo_core_cnn
from src.utils_basic import get_time_str, compress_list, wait_gpu_cool_writer
from src.utils_protein_env import ProteinEnv, ProteinEnvDDG, ProteinEnvDouble
from src.spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from src.spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


# FILE_NAME = "../src/spinup/algos/pytorch/ppo/para_temp.pt"

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPOBufferOffPolicy:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.real_size = 0

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        self.real_size += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, get_num):
        # assert self.ptr == self.max_size  # buffer has to be full before you can get
        # self.ptr, self.path_start_idx = 0, 0

        adv_buf_real = self.adv_buf[:self.real_size]
        obs_buf_real = self.obs_buf[:self.real_size]
        act_buf_real = self.act_buf[:self.real_size]
        ret_buf_real = self.ret_buf[:self.real_size]
        logp_buf_real = self.logp_buf[:self.real_size]

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(adv_buf_real)
        adv_buf_real = (adv_buf_real - adv_mean) / adv_std

        half_num = int(get_num/2)
        idxs = np.random.randint(0, self.real_size, size=half_num)
        obs_buf_return = np.concatenate((obs_buf_real[idxs], obs_buf_real[-half_num:]))
        act_buf_return = np.concatenate((act_buf_real[idxs], act_buf_real[-half_num:]))
        ret_buf_return = np.concatenate((ret_buf_real[idxs], ret_buf_real[-half_num:]))
        adv_buf_return = np.concatenate((adv_buf_real[idxs], adv_buf_real[-half_num:]))
        logp_buf_return = np.concatenate((logp_buf_real[idxs], logp_buf_real[-half_num:]))

        data = dict(
            obs=obs_buf_return,
            act=act_buf_return,
            ret=ret_buf_return,
            adv=adv_buf_return,
            logp=logp_buf_return,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


# Set up function for computing PPO policy loss
def compute_loss_pi(data, ac, clip_ratio):
    obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

    # Policy loss
    pi, logp = ac.pi(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info


# Set up function for computing value loss
def compute_loss_v(data, ac):
    obs, ret = data["obs"], data["ret"]
    return ((ac.v(obs) - ret) ** 2).mean()


def update(buf, ac, clip_ratio, train_pi_iters, pi_optimizer, target_kl, writer, train_v_iters, vf_optimizer,
           total_step, local_steps_per_epoch, data_reuse, v):
    data = buf.get()  # get_num=local_steps_per_epoch * data_reuse

    # print("data[adv].shape: ", data["adv"].shape)

    pi_l_old, pi_info_old = compute_loss_pi(data, ac, clip_ratio)
    pi_l_old = pi_l_old.item()
    v_l_old = compute_loss_v(data, ac).item()

    # Train policy with multiple steps of gradient descent
    for i in range(train_pi_iters):
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data, ac, clip_ratio)
        kl = mpi_avg(pi_info["kl"])
        if kl > 1.5 * target_kl:
            # logger.log('Early stopping at step %d due to reaching max kl.' % i)
            break
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # loss_pi.to(device)
        loss_pi.backward()
        # print("ac.pi: ", ac.pi)
        mpi_avg_grads(ac.pi)  # average grads across MPI processes
        pi_optimizer.step()

    # logger.store(StopIter=i)
    writer.add_scalar("9_others/StopIter", i, total_step)

    # Value function learning
    for i in range(train_v_iters):
        vf_optimizer.zero_grad()
        loss_v = compute_loss_v(data, ac)
        # loss_v.to("cuda:0")
        loss_v.backward()
        mpi_avg_grads(ac.v)  # average grads across MPI processes
        vf_optimizer.step()

    # Log changes from update
    kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]

    # writer.add_scalar("5_monitor/2_loss_v", loss_v, total_step)
    loss_v_sqrt = loss_v**0.5
    writer.add_scalar("5_monitor/2_loss_v_sqrt", loss_v_sqrt, total_step)
    if loss_v_sqrt / v[0] > 0:
        loss_v_sqrt_rela = loss_v_sqrt / v[0]
        # loss_v_sqrt_rela = np.clip(loss_v_sqrt_rela, a_min=0, a_max=1)
        writer.add_scalar("5_monitor/3_loss_v_sqrt_rela", loss_v_sqrt_rela, total_step)
    # writer.add_scalar("5_monitor/3_loss_pi", loss_pi, total_step)
    writer.add_scalar("5_monitor/4_loss_pi_abs", abs(loss_pi), total_step)
    writer.add_scalar("5_monitor/6_ent", ent, total_step)
    writer.add_scalar("5_monitor/7_kl", kl, total_step)
    writer.add_scalar("5_monitor/8_cf", cf, total_step)


def train_init_process(ac_kwargs, ac, gamma, lam, logger_kwargs, pi_lr, seed, steps_per_epoch, vf_lr, train_env, writer):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    # Set up logger and save configuration
    # logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())
    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Instantiate environment
    obs_dim = train_env.observation_space.shape
    act_dim = train_env.action_space.shape

    # Sync params across processes
    sync_params(ac)
    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])

    writer.add_scalar("7_para/var_counts_pi", var_counts[0], 0)
    writer.add_scalar("7_para/var_counts_pi", var_counts[0], 1)
    writer.add_scalar("7_para/var_counts_v", var_counts[1], 0)
    writer.add_scalar("7_para/var_counts_v", var_counts[1], 1)

    # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)
    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    # buf = PPOBufferOffPolicy(obs_dim, act_dim, int(10e6), gamma, lam)
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    # Set up model saving
    # logger.setup_pytorch_saver(ac)
    # Prepare for interaction with environment
    start_time = time.time()
    return ac, buf, train_env, local_steps_per_epoch, writer, pi_optimizer, start_time, \
        vf_optimizer, act_dim


def step_process(ac, buf, train_env, ep_len, ep_ret, ep_ret_plddt_specify, ep_ret_plddt_entire, local_steps_per_epoch,
                 writer, max_ep_len, o, t, total_step, PROTEIN_EXTRA):
    temp_o = torch.as_tensor(o, dtype=torch.float32)
    a, v, logp = ac.step(temp_o)
    # a = train_env.action_space.sample()
    output = train_env.step(a)
    next_o, r, d, extra_info = output

    if extra_info is not None:
        for i_str in extra_info:
            if extra_info[i_str] is not None:
                writer.add_scalar("1_train/" + i_str, extra_info[i_str], total_step)

    ep_ret += r
    # if PROTEIN_EXTRA:
    #     plddt_specify = extra_info["plddt_specify"]
    #     plddt_entire = extra_info["plddt_entire"]
    #     ep_ret_plddt_specify += plddt_specify
    #     ep_ret_plddt_entire += plddt_entire
    ep_len += 1
    # save and log
    buf.store(o, a, r, v, logp)
    writer.add_scalar("5_monitor/1_z_VVals", v, total_step)
    # Update obs (critical!)
    o = next_o
    timeout = ep_len == max_ep_len
    terminal = d or timeout
    epoch_ended = t == local_steps_per_epoch - 1
    return ep_len, ep_ret, ep_ret_plddt_specify, ep_ret_plddt_entire, epoch_ended, o, terminal, timeout, v


def terminal_process(ac, buf, train_env, ep_len, ep_ret, ep_ret_plddt_specify, ep_ret_plddt_entire, list_train_return,
                     epoch, epoch_ended, list_train_steps, writer, o, terminal, timeout, total_step, PROTEIN_EXTRA,
                     GPU_TEMPERATURE_LIMIT):
    if epoch_ended and not terminal:
        # print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
        pass
    # if trajectory didn't reach terminal state, bootstrap value target
    if timeout or epoch_ended:
        _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
    else:
        v = 0
    buf.finish_path(v)
    if terminal:
        # only save EpRet / EpLen if trajectory finished
        # logger.store(EpRet=ep_ret, EpLen=ep_len)
        list_train_return.append(ep_ret)
        list_train_steps.append(total_step)
        writer.add_scalar("1_train/1_episode_return", ep_ret, total_step)
        # if PROTEIN_EXTRA:
        #     writer.add_scalar("1_train/protein_episode_return_plddt_specify", ep_ret_plddt_specify, total_step)
        #     writer.add_scalar("1_train/protein_episode_return_plddt_entire", ep_ret_plddt_entire, total_step)
        writer.add_scalar("1_train/3_episode_length", ep_len, total_step)
        writer.add_scalar("1_train/5_step_reward", ep_ret / ep_len, total_step)
        # print("trainReward")
    o, ep_ret, ep_len = train_env.reset(dataset_mode="Train"), 0, 0
    ep_ret_plddt_specify = 0
    ep_ret_plddt_entire = 0

    if GPU_TEMPERATURE_LIMIT is not None:
        time_gpu_cool = wait_gpu_cool_writer(total_step, writer, class_name="6_time", gpu_temperature_limit=GPU_TEMPERATURE_LIMIT)
    else:
        time_gpu_cool = 0

    return ep_len, ep_ret, ep_ret_plddt_specify, ep_ret_plddt_entire, o, time_gpu_cool


def once_valid_ret(ac, valid_env, writer=None, total_step=None, random_flag=False, PROTEIN_EXTRA=False):
    start_time = time.time()
    # raw_ret = 0
    # norm_ret = 0
    return_val = 0
    return_plddt_specify = 0
    return_plddt_entire = 0

    length = 0
    o = valid_env.reset(dataset_mode="Test")
    d = False
    while not d:
        length += 1
        temp_o = torch.as_tensor(o, dtype=torch.float32)
        if random_flag:
            a = valid_env.action_space.sample()
        else:
            a, v, logp = ac.step(temp_o)
        # a = valid_env.action_space.sample()
        next_o, r, d, extra_info = valid_env.step(a)
        o = next_o
        return_val += r
        # if PROTEIN_EXTRA:
        #     plddt_specify = extra_info["plddt_specify"]
        #     plddt_entire = extra_info["plddt_entire"]
        #     return_plddt_specify += plddt_specify
        #     return_plddt_entire += plddt_entire

        if extra_info is not None:
            for i_str in extra_info:
                if extra_info[i_str] is not None:
                    writer.add_scalar("3_valid/" + i_str, extra_info[i_str], total_step)
    valid_time = time.time() - start_time
    o, ep_ret, ep_len = valid_env.reset(dataset_mode="Train"), 0, 0
    return return_val, length, valid_time, return_plddt_specify, return_plddt_entire  # , raw_ret, norm_ret


def draw_plt(list_train_return, list_valid_steps, list_valid_return, list_train_steps):
    plt.subplot(2, 1, 1)
    plt.cla()
    plt.plot(
        compress_list(list_train_steps),
        compress_list(list_train_return),
        label="train",
    )
    # plt.xlabel("Steps")
    plt.ylabel("Returns")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.cla()
    plt.plot(
        compress_list(list_valid_steps),
        compress_list(list_valid_return),
        label="valid",
    )
    plt.xlabel("Steps")
    plt.ylabel("Returns")
    plt.legend()


def epoch_process(ac, buf, clip_ratio, train_env, list_train_return, epoch, list_train_steps, epochs, writer, pi_optimizer,
                  save_freq, start_time, local_steps_per_epoch, target_kl, train_pi_iters, train_v_iters, vf_optimizer,
                  list_valid_return, list_valid_steps, act_dim, total_step, last_plot_time, date_str, time_str,
                  PROTEIN_EXTRA, PLOT_FLAG, data_reuse, v):
    # Save model
    # if (epoch % save_freq == 0) or (epoch == epochs - 1):
    #     logger.save_state({'train_env': train_env}, None)
    # Perform PPO update!
    # try:
    start_time = time.time()
    update(buf, ac, clip_ratio, train_pi_iters, pi_optimizer, target_kl, writer, train_v_iters, vf_optimizer,
           total_step, local_steps_per_epoch, data_reuse, v)
    writer.add_scalar("6_time/time_update", time.time()-start_time, total_step)
    # except TypeError:
    #     print("TypeError!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # valid_once
    valid_ret, length, valid_time, return_plddt_specify, return_plddt_entire = \
        once_valid_ret(ac, train_env, writer, total_step, PROTEIN_EXTRA=PROTEIN_EXTRA)
    writer.add_scalar("6_time/time_valid", valid_time, total_step)

    writer.add_scalar("3_valid/1_episode_return", valid_ret, total_step)
    writer.add_scalar("3_valid/3_episode_length", length, total_step)
    writer.add_scalar("3_valid/5_step_reward", valid_ret / length, total_step)
    # if PROTEIN_EXTRA:
    #     writer.add_scalar("3_valid/protein_episode_return_plddt_specify", return_plddt_specify, total_step)
    #     writer.add_scalar("3_valid/protein_episode_return_plddt_entire", return_plddt_entire, total_step)

    list_valid_return.append(valid_ret)
    list_valid_steps.append(total_step)

    # if random.random() < FIG_POSS:
    if time.time() - last_plot_time > 1 * num_procs():
        if PLOT_FLAG:
            draw_plt(list_train_return, list_valid_steps, list_valid_return, list_train_steps)
            # plt.ylim([0, 0.2])
            plt.savefig("../data/interim/fig_temp" + ".pdf")
            plt.pause(0.00000000000001)
        reward_file_name = "../data/interim/tensorboard/" + date_str + "_" + time_str + "/training_curve" +\
                           date_str + "_" + time_str + ".npy"
        np.save(reward_file_name, [list_train_return, list_train_steps, list_valid_return, list_valid_steps])
        torch.save(
            ac.state_dict(), "../data/interim/tensorboard/" + date_str + "_" + time_str + "/ppo_para" +
                             date_str + "_" + time_str + ".pt"
            # ../src/spinup/algos/pytorch/ppo/para_temp.pt
        )


def train_ppo_loop(ac, buf, clip_ratio, epochs, init_local_steps_per_epoch, writer, max_ep_len, pi_optimizer,
                   save_freq, start_time, target_kl, train_env, train_pi_iters, train_v_iters, vf_optimizer,
                   act_dim, date_str, time_str, obs_dim, gamma, lam, GPU_TEMPERATURE_LIMIT,
                   PROTEIN_EXTRA=False, PLOT_FLAG=False):
    list_train_return = []
    list_train_steps = []
    list_valid_return = []
    list_valid_steps = []
    total_step = 0
    total_optimize = 0
    last_plot_time = 0

    last_epoch_time = None

    o, ep_ret, ep_len = train_env.reset(dataset_mode="Train"), 0, 0
    ep_ret_plddt_specify = 0
    ep_ret_plddt_entire = 0

    data_reuse = 10

    for epoch in tqdm.tqdm(range(epochs)):
        local_steps_per_epoch = init_local_steps_per_epoch + int(epoch * 0.001)  # 0.001 1
        writer.add_scalar("6_time/local_steps_per_epoch", local_steps_per_epoch, total_step)
        buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
        time_epoch_gpu_cool = 0
        v = 0
        for t in range(local_steps_per_epoch):
            total_step += 1
            ep_len, ep_ret, ep_ret_plddt_specify, ep_ret_plddt_entire, epoch_ended, o, terminal, timeout, v = \
                step_process(ac, buf, train_env, ep_len, ep_ret, ep_ret_plddt_specify, ep_ret_plddt_entire,
                             local_steps_per_epoch, writer, max_ep_len, o, t, total_step, PROTEIN_EXTRA=PROTEIN_EXTRA)

            if terminal or epoch_ended:
                ep_len, ep_ret, ep_ret_plddt_specify, ep_ret_plddt_entire, o, time_gpu_cool = \
                    terminal_process(ac, buf, train_env, ep_len, ep_ret, ep_ret_plddt_specify, ep_ret_plddt_entire,
                                     list_train_return, epoch, epoch_ended, list_train_steps, writer, o, terminal,
                                     timeout, total_step, PROTEIN_EXTRA, GPU_TEMPERATURE_LIMIT)
                time_epoch_gpu_cool += time_gpu_cool

            now_time = time.time()
            writer.add_scalar("6_time/frequency_sample", total_step / (time.time()-start_time) * num_procs(), total_step)

        assert epoch_ended
        epoch_process(ac, buf, clip_ratio, train_env, list_train_return, epoch, list_train_steps, epochs, writer,
                      pi_optimizer, save_freq, start_time, local_steps_per_epoch, target_kl, train_pi_iters, train_v_iters,
                      vf_optimizer, list_valid_return, list_valid_steps, act_dim, total_step, last_plot_time, date_str,
                      time_str, PROTEIN_EXTRA, PLOT_FLAG, data_reuse, v)

        total_optimize += 1
        now_time = time.time()
        if last_epoch_time is not None:
            epoch_time = now_time - last_epoch_time
            writer.add_scalar("6_time/time_epoch", epoch_time, total_step)
            writer.add_scalar("6_time/frequency_optimize", total_optimize / (now_time-start_time), total_step)
            frequency_optimize_steps = total_optimize / (now_time-start_time) * local_steps_per_epoch * num_procs()
            writer.add_scalar("6_time/frequency_optimize_steps", frequency_optimize_steps, total_step)
            # writer.add_scalar("6_time/data_reuse", data_reuse, total_step)
        last_epoch_time = now_time

        writer.add_scalar("6_time/time_epoch_gpu_cool", time_epoch_gpu_cool, total_step)

    if PLOT_FLAG:
        draw_plt(list_train_return, list_valid_steps, list_valid_return, list_train_steps)
    date_str, time_str = get_time_str()
    plt.savefig("../data/processed/fig_" + date_str + "_" + time_str
                + "_act" + str(train_env.action_space.n)
                # + "_hid" + str(HIDDEN) + "_lay" + str(LAYER)
                + ".pdf")
    torch.save(ac.state_dict(), "../data/processed/para_" + date_str + "_" + time_str + "_act"
               + str(train_env.action_space.n)
               # + "_hid" + str(HIDDEN) + "_lay" + str(LAYER)
               + ".pt")


def train_ppo(train_env, writer, date_str, time_str, GPU_TEMPERATURE_LIMIT, load_content, ac,
              ac_kwargs=dict(), seed=0,
              init_steps_per_epoch=4000, epochs=50, gamma=0.99,
              clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
              target_kl=0.01, logger_kwargs=dict(), save_freq=10, PROTEIN_EXTRA=False, PLOT_FLAG=False):
    assert init_steps_per_epoch >= max_ep_len

    ac, buf, train_env, init_local_steps_per_epoch, writer, pi_optimizer, start_time, vf_optimizer,\
        act_dim = train_init_process(ac_kwargs, ac, gamma, lam, logger_kwargs, pi_lr, seed, init_steps_per_epoch,
                                     vf_lr, train_env, writer)

    if load_content is not None:
        ac.load_state_dict(load_content)

    # Main loop: collect experience in train_env and update/log each epoch
    train_ppo_loop(ac, buf, clip_ratio, epochs, init_local_steps_per_epoch, writer, max_ep_len, pi_optimizer,
                   save_freq, start_time, target_kl, train_env, train_pi_iters, train_v_iters,
                   vf_optimizer, act_dim, date_str, time_str, train_env.observation_space.shape, gamma, lam,
                   GPU_TEMPERATURE_LIMIT,
                   PROTEIN_EXTRA=PROTEIN_EXTRA, PLOT_FLAG=PLOT_FLAG)


def prepare_para(STEPS_PER_EPOCH, EPOCHS, CPU_NUM):
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='CartPole-v1')  # HalfCheetah-v2
    # parser.add_argument("--hid", type=int, default=HIDDEN)
    # parser.add_argument("--l", type=int, default=LAYER)
    parser.add_argument("--gamma", type=float, default=0.99)  # 0.99 1
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=CPU_NUM)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPOCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--exp_name", type=str, default="ppo")
    args = parser.parse_args()
    mpi_fork(args.cpu)  # run parallel code with mpi
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    return args, logger_kwargs


def main_train_env():
    MAX_EP_LEN = 80
    INIT_STEPS_PER_EPOCH = 80
    EPOCHS = int(500000 / INIT_STEPS_PER_EPOCH)
    PROTEIN_EXTRA = True
    CPU_NUM = 1
    HIDDEN_LIST = [64, 64]
    GPU_TEMPERATURE_LIMIT = 58
    PRED_RATE = 0.7
    LOAD_FILE_NAME = None  # "../data/interim/tensorboard/20230511_203251/ppo_para20230511_203251.pt" None

    if LOAD_FILE_NAME is not None:
        load_content = torch.load(LOAD_FILE_NAME)
        # load_content["pi.net_mlp_sum.4.weight"] = torch.rand(20, 64)
        # load_content["pi.net_mlp_sum.4.bias"] = torch.rand(20)
    else:
        load_content = None

    assert MAX_EP_LEN >= 75
    assert INIT_STEPS_PER_EPOCH >= MAX_EP_LEN

    args, logger_kwargs = prepare_para(INIT_STEPS_PER_EPOCH, EPOCHS, CPU_NUM)

    train_env = ProteinEnvDouble(pred_rate=PRED_RATE)  # ProteinEnv ProteinEnvDouble

    date_str, time_str = get_time_str()
    writer = SummaryWriter("../data/interim/tensorboard/" + date_str + "_" + time_str)
    writer.add_scalar("7_para/train_env.observation_space.shape[0]", train_env.observation_space.shape[0], 0)
    writer.add_scalar("7_para/train_env.observation_space.shape[0]", train_env.observation_space.shape[0], 1)
    writer.add_scalar("7_para/CPU_NUM", CPU_NUM, 0)
    writer.add_scalar("7_para/CPU_NUM", CPU_NUM, 1)
    writer.add_scalar("7_para/gamma", args.gamma, 0)
    writer.add_scalar("7_para/gamma", args.gamma, 1)
    writer.add_scalar("7_para/PRED_RATE", PRED_RATE, 0)
    writer.add_scalar("7_para/PRED_RATE", PRED_RATE, 1)

    actor_critic = src.utils_ppo_core_cnn.CNNActorCritic
    ac_kwargs = dict(hidden_sizes=HIDDEN_LIST)
    multi_branch_obs_dim = train_env.multi_branch_obs_dim
    ac = actor_critic(multi_branch_obs_dim, train_env.action_space, **ac_kwargs)

    train_ppo(
        train_env,
        writer,
        date_str, time_str,
        GPU_TEMPERATURE_LIMIT,
        load_content,
        ac,
        ac_kwargs=ac_kwargs,
        gamma=args.gamma,
        seed=args.seed,
        init_steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        max_ep_len=MAX_EP_LEN,
        PROTEIN_EXTRA=PROTEIN_EXTRA
    )


def main_train_env_ddg():
    MAX_EP_LEN = 30
    PROTEIN_EXTRA = False
    DATASET_NUM = 53  # None 1 53
    PLDDT_FLAG = False
    train_env = ProteinEnvDDG(dataset_num=DATASET_NUM, plddt_flag=PLDDT_FLAG)
    if PLDDT_FLAG:
        CPU_NUM = 1
        GPU_TEMPERATURE_LIMIT = 58
    else:
        CPU_NUM = 10  # 1 10 9
        GPU_TEMPERATURE_LIMIT = None
    STEPS_PER_EPOCH = 50 * CPU_NUM
    if DATASET_NUM == 1:
        EPOCHS = int(50000 / STEPS_PER_EPOCH) * CPU_NUM
    else:
        EPOCHS = int(1000000 / STEPS_PER_EPOCH) * CPU_NUM
    if CPU_NUM > 2:
        PLOT_FLAG = False
    else:
        PLOT_FLAG = True
    # LAYER = 2
    # HIDDEN = 64
    HIDDEN_LIST = [64, 64]  # 400, 200, 100

    # "../data/interim/tensorboard/20230420_102212/ppo_para20230420_102212.pt"  # 20230419_144628
    # "../data/interim/tensorboard/20230423_110331/ppo_para20230423_110331.pt"
    # "../data/interim/tensorboard/20230517_092553/ppo_para20230517_092553.pt"
    # None
    LOAD_FILE_NAME = None

    if LOAD_FILE_NAME is not None:
        load_content = torch.load(LOAD_FILE_NAME)
        load_content["pi.net_mlp_sum.4.weight"] = torch.rand(20, 64)
        load_content["pi.net_mlp_sum.4.bias"] = torch.rand(20)
        load_content["v.net_mlp_sum.4.weight"] = torch.rand(1, 64)
        load_content["v.net_mlp_sum.4.bias"] = torch.rand(1)
    else:
        load_content = None

    assert MAX_EP_LEN >= 10
    assert STEPS_PER_EPOCH / CPU_NUM >= MAX_EP_LEN

    args, logger_kwargs = prepare_para(STEPS_PER_EPOCH, EPOCHS, CPU_NUM)

    date_str, time_str = get_time_str()
    writer = SummaryWriter("../data/interim/tensorboard/" + date_str + "_" + time_str)
    writer.add_scalar("7_para/DATASET_NUM", DATASET_NUM, 0)
    writer.add_scalar("7_para/DATASET_NUM", DATASET_NUM, 1)
    writer.add_scalar("7_para/STEPS_PER_EPOCH", STEPS_PER_EPOCH, 0)
    writer.add_scalar("7_para/STEPS_PER_EPOCH", STEPS_PER_EPOCH, 1)
    writer.add_scalar("7_para/train_env.observation_space.shape[0]", train_env.observation_space.shape[0], 0)
    writer.add_scalar("7_para/train_env.observation_space.shape[0]", train_env.observation_space.shape[0], 1)
    writer.add_scalar("7_para/CPU_NUM", CPU_NUM, 0)
    writer.add_scalar("7_para/CPU_NUM", CPU_NUM, 1)
    writer.add_scalar("7_para/gamma", args.gamma, 0)
    writer.add_scalar("7_para/gamma", args.gamma, 1)

    actor_critic = src.utils_ppo_core_cnn.CNNActorCritic
    ac_kwargs = dict(hidden_sizes=HIDDEN_LIST)
    multi_branch_obs_dim = train_env.multi_branch_obs_dim
    ac = actor_critic(multi_branch_obs_dim, train_env.action_space, **ac_kwargs)

    train_ppo(
        train_env,
        writer,
        date_str, time_str,
        GPU_TEMPERATURE_LIMIT,
        load_content,
        ac,
        ac_kwargs=ac_kwargs,
        gamma=args.gamma,
        seed=args.seed,
        init_steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        max_ep_len=MAX_EP_LEN,
        PROTEIN_EXTRA=PROTEIN_EXTRA,
        PLOT_FLAG=PLOT_FLAG
    )


def main():
    # main_train_env()
    main_train_env_ddg()


if __name__ == "__main__":
    main()
