import torch
import torch.optim as optim
import argparse
import numpy as np
import time
from tensorboardX import SummaryWriter
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../..')

from pg_travel.deeprm import model
from pg_travel.deeprm.hparams import HyperParams as Hp
from pg_travel.deeprm.env_simu_sigle.environment import Env
from pg_travel.deeprm.env_simu_sigle import other_agents
from pg_travel.deeprm.env_simu_sigle import job_distribution
from pg_travel.deeprm.agent import vanila_pg


def get_traj(test_type, hp, env, pg_resume=None, render=False):
    if test_type == 'PG':
        if hp.use_cnn:
            actor = vanila_pg.ActorConv(hp.network_input_height, hp.network_input_width, hp.channel,
                                        hp.network_output_dim)
        elif hp.use_big_cnn:
            actor = vanila_pg.ActorConvBig(hp.network_input_height, hp.network_input_width, hp.channel,
                                           hp.network_output_dim)
        else:
            actor = vanila_pg.ActorLinear(hp.network_input_height * hp.network_input_width, hp.network_output_dim, hp)
        actor.load_state_dict(torch.load(pg_resume))

    env.reset()
    rews = []
    ob = env.observe()

    for _ in range(hp.episode_max_length):
        if test_type == 'PG':
            # print(hp.DEBUG, ob.shape)
            # print(hp.DEBUG, type(ob))
            if hp.use_big_cnn or hp.use_cnn:
                ob = torch.Tensor(ob).unsqueeze(0).unsqueeze(0)
            logits = actor(torch.Tensor(ob))
            a = torch.distributions.Categorical(logits=logits).sample().numpy().squeeze()
        elif test_type == 'Tetris':
            a = other_agents.get_packer_action(env.machine, env.job_slot)
        elif test_type == 'SJF':
            a = other_agents.get_sjf_action(env.machine, env.job_slot)
        elif test_type == 'Random':
            a = other_agents.get_random_action(env.job_slot)

        # print(hp.DEBUG, a)
        # print(hp.DEBUG, type(a))
        ob, rew, done, info = env.step(a, repeat=True)
        rews.append(rew)

        if done: break
        if render: env.render()

    return np.array(rews), info


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i + 1] + gamma ^ 2 * x[i + 2] + ...
    :param x:
    :param gamma:
    :return:
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    assert x.ndim >= 1
    # TODO: More efficient version:
    #     # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    # TODO: and maybe torch has similar method
    return out


def launch(hp, pg_resume=None, render=False, plot=False):

    # ----- Parameters -----
    test_types = ['Tetris', 'SJF', 'Random']

    if pg_resume is not None:
        test_types = ['PG'] + test_types

    env = Env(hp, render=render)

    all_discount_rews = {}
    jobs_slow_down = {}
    work_complete = {}
    work_remain = {}
    job_len_remain = {}
    num_job_remain = {}
    job_remain_delay = {}

    for test_type in test_types:
        all_discount_rews[test_type] = []
        jobs_slow_down[test_type] = []
        work_complete[test_type] = []
        work_remain[test_type] = []
        job_len_remain[test_type] = []
        num_job_remain[test_type] = []
        job_remain_delay[test_type] = []

    for seq_no in range(hp.num_ex):

        for test_type in test_types:
            rews, info = get_traj(test_type, hp, env, pg_resume)

            epi_rew_sum = discount(rews, hp.gamma)[0]

            all_discount_rews[test_type].append(epi_rew_sum)

            # ---- per job stat ----

            episode_len = len(info.record)
            enter_time = np.array([info.record[i].enter_time for i in range(episode_len)])
            finish_time = np.array([info.record[i].finish_time for i in range(episode_len)])
            job_len = np.array([info.record[i].len for i in range(episode_len)])
            job_total_size = np.array([np.sum(info.record[i].res_vec) for i in range(episode_len)])

            finished_idx = (finish_time >= 0)
            unfinished_idx = (finish_time < 0)

            jobs_slow_down[test_type].append(
                (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
            )
            work_complete[test_type].append(
                np.sum(job_len[finished_idx] * job_total_size[finished_idx])
            )
            work_remain[test_type].append(
                np.sum(job_len[finished_idx] * job_total_size[finished_idx])
            )
            job_len_remain[test_type].append(
                np.sum(job_len[unfinished_idx])
            )
            num_job_remain[test_type].append(
                len(job_len[unfinished_idx])
            )
            job_remain_delay[test_type].append(
                np.sum(hp.episode_max_length - enter_time[unfinished_idx])
            )

        env.seq_no = (env.seq_no + 1) % hp.num_ex

    # -- matplotlib colormap no overlap --
    if plot:
        num_colors = len(test_types)
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

        for test_type in test_types:
            jobs_slow_down_type = np.concatenate(jobs_slow_down[test_type])
            slow_down_cdf = np.sort(jobs_slow_down_type)
            slow_down_yvals = np.arange(len(slow_down_cdf)) / float(len(slow_down_cdf))
            ax.plot(slow_down_cdf, slow_down_yvals, linewidth=2, label=test_type)

            max_slow_down = np.max(jobs_slow_down_type)

            print("The test type [{}] gets mean total discounted reward: {}".
                  format(test_type, np.mean(all_discount_rews[test_type])))
            print("The test type [{}] gets mean slowdown: {}".
                  format(test_type, np.mean(jobs_slow_down_type)))
            print("The test type [{}] gets max slow down: {}".format(test_type, max_slow_down))
            print("=============================================")

        plt.legend(loc=4)
        plt.xlabel("job slowdown", fontsize=20)
        plt.ylabel("CDF", fontsize=20)

        if pg_resume is None:
            pg_resume = "None"

        # plt.show()
        plt.savefig("test_{}_slowdown_fig.pdf".format(pg_resume))
        print("Output file: " + "test_{}_slowdown_fig.pdf".format(pg_resume))


def main():
    # models/vanilla_pg_single_linear_1700[2019-06-26-18:42:47].pkl
    # vanilla-pg-cnn-2400-optim=adam-lr=0.0001-simu_len=50-nun_ex=10_4000.pkl
    # vanilla-pg-imit=200-linear-optim=adam-lr=0.0001-simu_len=50-nun_ex=10_49990.pkl
    # vanilla-pg-cnn-2400-optim=adam-lr=0.0001-simu_len=50-nun_ex=10_5480.pkl

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str,
                        default=
                        'models/vanilla-pg-imit=200-linear-optim=adam-lr=0.0001-simu_len=50-nun_ex=10_30000.pkl')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('--rms', dest='rms', action='store_true')
    parser.set_defaults(rms=False)
    parser.add_argument('--cnn', dest='cnn', action='store_true')
    parser.set_defaults(cnn=False)
    parser.add_argument('--big_cnn', dest='big_cnn', action='store_true')
    parser.set_defaults(big_cnn=False)
    parser.add_argument('--unseen', dest='unseen', action='store_true')
    parser.set_defaults(unseen=False)
    # parser.add_argument('--algorithm', type=str, default='PG',
    #                     help='select one of algorithms among Vanilla_PG, NPG, TRPO, PPO')

    args = parser.parse_args()

    hp = Hp()
    hp.gpu = args.gpu
    hp.use_cnn = args.cnn
    hp.use_big_cnn = args.big_cnn

    if args.unseen:
        hp.simu_len = 60
        hp.num_ex = 15
        hp.episode_max_length = 20000
        hp.unseen = True
    else:
        hp.simu_len = 50
        hp.num_ex = 10
        hp.episode_max_length = 200
        hp.unseen = False

    hp.compute_dependent_parameters()

    render = False

    plot = True

    # pg_resume = None
    # pg_resume = 'models/vanilla_pg_single_linear_1700[2019-06-26-18:42:47].pkl'
    # vanilla-pg-cnn-2400-optim=adam-lr=0.0001-simu_len=50-nun_ex=10_4000.pkl
    # vanilla-pg-imit=200-linear-optim=adam-lr=0.0001-simu_len=50-nun_ex=10_49990.pkl
    pg_resume = args.load_model

    launch(hp, pg_resume, render, plot)


if __name__ == '__main__':
    main()
