import numpy as np
import torch
from gym.spaces import Box, Discrete
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from src.utils_ppo_core import Actor


# CNN_SIZE = 96  # 16 * 8  # int(32 * (30-4)/2)


def network_cnn_mlp(sizes, activation, multi_branch_obs_dim, output_activation=nn.Identity):
    # cnn
    layers_cnn = []
    mbo = multi_branch_obs_dim  # multi_branch_obs_dim [[10, 20]], 53, 1]
    # [?, 1, mbo[0][0], mbo[0][1]]
    cnn = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, mbo[0][1]), padding=0)
    layers_cnn += [cnn, activation()]
    # [?, 16, mbo[0][0]-2, 1]
    cnn = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), padding=0)
    layers_cnn += [cnn, activation()]
    # [?, 32, mbo[0][0]-4, 1]

    # mlp_sum
    layers_mlp_sum = []
    if len(mbo) > 1:
        sizes[0] = 32 * int((mbo[0][0] - 4) / 2) + mbo[1] + mbo[2]  # 90 + 9 + 1
    else:
        sizes[0] = 32 * int((mbo[0][0] - 4) / 2)
    for j in range(len(sizes) - 1):
        fulll_connect = nn.Linear(sizes[j], sizes[j + 1])
        act = activation if j < len(sizes) - 2 else output_activation
        layers_mlp_sum += [fulll_connect, act()]

    return nn.Sequential(*layers_cnn), \
        nn.Sequential(*layers_mlp_sum)  # , nn.Sequential(*layers_cnn_2)


def network_cnn_cnn_mlp(sizes, activation, multi_branch_obs_dim, output_activation=nn.Identity):
    mbo = multi_branch_obs_dim  # multi_branch_obs_dim [[10, 20]], 53, 1]

    # cnn1
    layers_cnn1 = []
    # [?, 1, mbo[0][0], mbo[0][1]]
    cnn = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, mbo[0][1]), padding=0)
    layers_cnn1 += [cnn, activation()]
    # [?, 8, mbo[0][0]-2, 1]
    cnn = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 1), padding=0)
    layers_cnn1 += [cnn, activation()]
    # [?, 16, mbo[0][0]-4, 1]

    # cnn2
    layers_cnn2 = []
    # [?, 1, mbo[1][0], mbo[1][1]]
    cnn = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, mbo[1][1]), padding=0)
    layers_cnn2 += [cnn, activation()]
    # [?, 8, mbo[1][0]-2, 1]
    cnn = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 1), padding=0)
    layers_cnn2 += [cnn, activation()]
    # [?, 16, mbo[1][0]-4, 1]

    # mlp_sum
    layers_mlp_sum = []
    sizes[0] = 16 * int((mbo[0][0] - 4) / 2) + 16 * int((mbo[1][0] - 4) / 2)  # 90 + 9 + 1
    for j in range(len(sizes) - 1):
        fulll_connect = nn.Linear(sizes[j], sizes[j + 1])
        act = activation if j < len(sizes) - 2 else output_activation
        layers_mlp_sum += [fulll_connect, act()]

    return nn.Sequential(*layers_cnn1), nn.Sequential(*layers_cnn2), nn.Sequential(*layers_mlp_sum)


def forward_cnn_mlp(obs, net_cnn, net_mlp, multi_branch_obs_dim):
    obs_len = multi_branch_obs_dim[0][0]
    x = obs.view(-1, obs_len * multi_branch_obs_dim[0][1])
    x_cdrh3 = x

    x_cdrh3 = x_cdrh3.view(-1, 1, obs_len, multi_branch_obs_dim[0][1])
    x_cdrh3 = net_cnn(x_cdrh3)
    x_cdrh3 = F.max_pool2d(x_cdrh3, kernel_size=(2, 1), stride=2)
    x_cdrh3 = x_cdrh3.view(-1, 32 * int((multi_branch_obs_dim[0][0] - 4) / 2))

    x = x_cdrh3
    x = net_mlp(x)
    return x


def forward_cnn_direct_mlp(obs, net_cnn, net_mlp_sum, mbd):
    x = obs.view(-1, mbd[0][0] * mbd[0][1] + mbd[1] + mbd[2])
    x_cdrh3, x_id, x_pos = torch.split(x, [mbd[0][0] * mbd[0][1], mbd[1], mbd[2]], dim=1)

    x_cdrh3 = x_cdrh3.view(-1, 1, mbd[0][0], mbd[0][1])
    x_cdrh3 = net_cnn(x_cdrh3)
    x_cdrh3 = F.max_pool2d(x_cdrh3, kernel_size=(2, 1), stride=2)
    x_cdrh3 = x_cdrh3.view(-1, 32 * int((mbd[0][0] - 4) / 2))

    x_id = x_id.view(-1, mbd[1])

    x_pos = x_pos.view(-1, mbd[2])

    x = torch.cat((x_cdrh3, x_id, x_pos), dim=1)
    x = net_mlp_sum(x)
    return x


def forward_cnn_cnn_mlp(obs, net_cnn1, net_cnn2, net_mlp_sum, mbd):
    x = obs.view(-1, mbd[0][0] * mbd[0][1] + mbd[1][0] * mbd[1][1])
    x_forward, x_backward = torch.split(x, [mbd[0][0] * mbd[0][1], mbd[1][0] * mbd[1][1]], dim=1)

    x_forward = x_forward.view(-1, 1, mbd[0][0], mbd[0][1])
    x_forward = net_cnn1(x_forward)
    x_forward = F.max_pool2d(x_forward, kernel_size=(2, 1), stride=2)
    x_forward = x_forward.view(-1, 16 * int((mbd[0][0] - 4) / 2))

    x_backward = x_backward.view(-1, 1, mbd[1][0], mbd[1][1])
    x_backward = net_cnn1(x_backward)
    x_backward = F.max_pool2d(x_backward, kernel_size=(2, 1), stride=2)
    x_backward = x_backward.view(-1, 16 * int((mbd[1][0] - 4) / 2))

    # x_backward = x_backward.view(-1, mbd[1])

    x = torch.cat((x_forward, x_backward), dim=1)
    x = net_mlp_sum(x)
    return x


class CNNCategoricalActor(Actor):
    def __init__(self, multi_branch_obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        if type(multi_branch_obs_dim[1]) is list:
            self.net_cnn1, self.net_cnn2, self.net_mlp_sum = \
                network_cnn_cnn_mlp([-1] + list(hidden_sizes) + [act_dim], activation, multi_branch_obs_dim)
        else:
            self.net_cnn, self.net_mlp_sum = \
                network_cnn_mlp([-1] + list(hidden_sizes) + [act_dim], activation, multi_branch_obs_dim)
        self.multi_branch_obs_dim = multi_branch_obs_dim

    def _distribution(self, obs):
        if len(self.multi_branch_obs_dim) > 1:
            if type(self.multi_branch_obs_dim[1]) is list:
                x = forward_cnn_cnn_mlp(obs, self.net_cnn1, self.net_cnn2, self.net_mlp_sum, self.multi_branch_obs_dim)
            else:
                x = forward_cnn_direct_mlp(obs, self.net_cnn, self.net_mlp_sum, self.multi_branch_obs_dim)
        else:
            x = forward_cnn_mlp(obs, self.net_cnn, self.net_mlp_sum, self.multi_branch_obs_dim)
        logits = x
        try:
            returns = Categorical(logits=logits)
        except ValueError:
            print("ValueError!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            returns = Categorical(logits=logits)
        return returns

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class CNNGaussianActor(Actor):
    def __init__(self, multi_branch_obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net_cnn, self.mu_net_mlp_sum = network_cnn_mlp([-1] + list(hidden_sizes) + [act_dim], activation,
                                                               multi_branch_obs_dim)
        self.multi_branch_obs_dim = multi_branch_obs_dim

    def _distribution(self, obs):  # todo: _distribution
        # mu = self.mu_net(obs)
        # std = torch.exp(self.log_std)
        mu = None
        std = None
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


class CNNCritic(nn.Module):
    def __init__(self, multi_branch_obs_dim, hidden_sizes, activation):
        super().__init__()
        if type(multi_branch_obs_dim[1]) is list:
            self.net_cnn1, self.net_cnn2, self.net_mlp_sum = \
                network_cnn_cnn_mlp([-1] + list(hidden_sizes) + [1], activation, multi_branch_obs_dim)
        else:
            self.net_cnn, self.net_mlp_sum = \
                network_cnn_mlp([-1] + list(hidden_sizes) + [1], activation, multi_branch_obs_dim)
        self.multi_branch_obs_dim = multi_branch_obs_dim

    def forward(self, obs):
        if len(self.multi_branch_obs_dim) > 1:
            if type(self.multi_branch_obs_dim[1]) is list:
                x = forward_cnn_cnn_mlp(obs, self.net_cnn1, self.net_cnn2, self.net_mlp_sum, self.multi_branch_obs_dim)
            else:
                x = forward_cnn_direct_mlp(obs, self.net_cnn, self.net_mlp_sum, self.multi_branch_obs_dim)
        else:
            x = forward_cnn_mlp(obs, self.net_cnn, self.net_mlp_sum, self.multi_branch_obs_dim)
        x = torch.squeeze(x, -1)
        return x  # Critical to ensure v has right shape.


class CNNActorCritic(nn.Module):
    def __init__(
            self, multi_branch_obs_dim, action_space, hidden_sizes=(64, 64), activation=nn.Tanh
    ):
        # Create actor-critic module
        super().__init__()

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = CNNGaussianActor(
                multi_branch_obs_dim, action_space.shape[0], hidden_sizes, activation
            )
        elif isinstance(action_space, Discrete):
            self.pi = CNNCategoricalActor(
                multi_branch_obs_dim, action_space.n, hidden_sizes, activation
            )

        # build value function
        self.v = CNNCritic(multi_branch_obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy()[0], v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
