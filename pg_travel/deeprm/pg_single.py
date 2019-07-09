import torch
import torch.optim as optim
import argparse
import numpy as np
import time
from tensorboardX import SummaryWriter
from collections import deque

import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('../..')

from pg_travel.deeprm import model
from pg_travel.deeprm.hparams import HyperParams as Hp
from pg_travel.deeprm.env_simu_sigle.environment import Env
from pg_travel.deeprm.env_simu_sigle import job_distribution
from pg_travel.deeprm.agent import vanila_pg
from pg_travel.deeprm import slow_down_cdf


DEBUG = "~~DEBUG~~"

def cnn_single_ob_trans(ob):
    return ob.reshape((1, 1, ob.shape[0], ob.shape[1]))

# variables like actor, env, etc was defined under main, so they are thought to be global and warned here
def get_traj(actor, env, hp, render=False):
    """
    Run agent-environment loop for one whole episode(trajectory)
    :param actor:
    :param env:
    :param hp:
    :param render:
    :return: dictionary of results
    """
    episode_max_length = hp.episode_max_length

    env.reset()
    # print(DEBUG, "Now is working on Env #{}".format(env.seq_no))
    obs = []
    acts = []
    rews = []
    info = []

    ob = env.observe()

    for _ in range(episode_max_length):

        if hp.gpu:
            if hp.use_big_cnn or hp.use_cnn:  # when use cnn, ob'shape should be (batch_size, #channel, h, w)
                logits = actor(torch.Tensor(cnn_single_ob_trans(ob)).cuda())
            else:
                logits = actor(torch.Tensor(ob).cuda())
            a = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        else:
            if hp.use_big_cnn or hp.use_cnn:
                logits = actor(torch.Tensor(cnn_single_ob_trans(ob)))
            else:
                logits = actor(torch.Tensor(ob))
            a = torch.distributions.Categorical(logits=logits).sample().numpy()
        # print(hp.DEBUG, a, type(a))
        # exit(0)

        if hp.use_big_cnn or hp.use_cnn:
            a = a.squeeze()  # when use cnn, action's shape is (batch_size, #a), but we just want a single a

        obs.append(ob)
        acts.append(a)

        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)

        if done: break
        if render: env.render()

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'info': info
            }


def concatenate_all_ob(trajs, hp):

    timesteps_total = 0
    for i in range(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])

    all_ob = np.zeros((timesteps_total, hp.network_input_height, hp.network_input_width))

    timesteps = 0
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['reward'])):
            all_ob[timesteps, :, :] = trajs[i]['ob'][j]
            timesteps += 1

    return all_ob


def process_all_info(trajs):
    """
    Here info means what is returned by step method.
    :param trajs:
    :return:
    """
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        record_len = len(traj['info'].record)
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in range(record_len)]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in range(record_len)]))
        job_len.append(np.array([traj['info'].record[i].len for i in range(record_len)]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len


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


def concatenate_all_ob_across_examples(all_ob, hp):
    num_ex =len(all_ob)
    total_samp = 0
    for i in range(num_ex):
        total_samp += all_ob[i].shape[0]

    all_ob_contact = np.zeros((total_samp, hp.network_input_height, hp.network_input_width))

    total_samp = 0

    for i in range(num_ex):
        prev_samp = total_samp
        total_samp += all_ob[i].shape[0]
        all_ob_contact[prev_samp: total_samp, :, :] = all_ob[i]

    return all_ob_contact


def launch(hp, load_model):

    print("Preparing envs...")

    env = Env(hp, seed=42)
    torch.manual_seed(42)

    if hp.use_cnn:
        print("Using small cnn")
        actor = vanila_pg.ActorConv(hp.network_input_height, hp.network_input_width, hp.channel, hp.network_output_dim)
    elif hp.use_big_cnn:
        actor = vanila_pg.ActorConvBig(hp.network_input_height, hp.network_input_width, hp.channel, hp.network_output_dim)
    else:
        actor = vanila_pg.ActorLinear(hp.network_input_height * hp.network_input_width, hp.network_output_dim, hp)
    total_num = sum(p.numel() for p in actor.parameters())
    trainable_num = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print('Total: ', total_num, 'Trainable:', trainable_num)

    if load_model is not None:
        actor.load_state_dict(torch.load(load_model))

    if hp.gpu:
        actor = actor.cuda()

    if hp.rmsprop:
        actor_optim = optim.RMSprop(actor.parameters(), lr=hp.actor_lr, alpha=0.9, eps=1e-9)
    else:
        actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)

    if hp.use_cnn is True:
        net_type = "cnn"
    elif hp.use_big_cnn is True:
        net_type = "big-cnn"
    else:
        net_type = "linear"

    if hp.rmsprop is True:
        optimizer = 'rms'
    else:
        optimizer = 'adam'

    params_path = "vanilla-pg-{}-lr={}-simu_len={}-nun_ex={}-num_nw={}-nw_rate={}".\
        format(net_type, hp.actor_lr, hp.simu_len, hp.num_ex, hp.num_nw, hp.new_job_rate)

    writer = SummaryWriter('runs/' + params_path)

    timer_start = time.time()

    print("Collecting data...")

    for iteration in range(hp.num_epochs):

        timer_begin = time.time()

        # TODO: maybe with actor.no_grad() is more appropriate?
        actor.eval()

        memory = deque()

        all_ob = []
        all_action = []
        all_rew = []
        all_ret = []
        all_adv = []
        all_eprews = []
        all_eplens = []
        all_slowdown = []
        all_entropy = []

        with torch.no_grad():
            for ex in range(hp.num_ex):

                env.seq_no = ex

                # Collect trajectories until we get time_steps_per_batch total time steps
                trajs = []

                for i in range(hp.num_seq_per_batch):  # here uses "per batch", there are many batches in an epoch
                    traj = get_traj(actor, env, hp)
                    trajs.append(traj)

                all_ob.append(concatenate_all_ob(trajs, hp))

                # Compute discounted sums of rewards
                rets = [discount(traj['reward'], hp.gamma) for traj in trajs]
                all_ret.append(np.concatenate(rets))

                maxlen = max(len(ret) for ret in rets)
                padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

                # Compute time-dependent baseline
                baseline = np.mean(padded_rets, axis=0)

                # Compute advantage function
                advs = [ret - baseline[:len(ret)] for ret in rets]
                all_adv.append(np.concatenate(advs))

                all_action.append(np.concatenate([traj['action'] for traj in trajs]))
                all_rew.append(np.concatenate([traj['reward'] for traj in trajs]))

                # episode total rewards, maybe only useful for print
                all_eprews.append(np.array([discount(traj['reward'], hp.gamma)[0] for traj in trajs]))
                # episode lengths
                all_eplens.append(np.array([len(traj['reward']) for traj in trajs]))

                # All Job Stat
                enter_time, finish_time, job_len = process_all_info(trajs)
                finished_idx = (finish_time >= 0)
                all_slowdown.append(
                    (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
                )

        all_ob = concatenate_all_ob_across_examples(all_ob, hp)
        all_action = np.concatenate(all_action)
        all_adv = np.concatenate(all_adv)
        all_rew = np.concatenate(all_rew)
        all_ret = np.concatenate(all_ret)

        # NOTE: all_ob, all_action, all_ret and all_rew is of the same length, but all_adv is longer.

        all_data = {
            'all_ob': all_ob,
            'all_action': all_action,
            'all_adv': all_adv,
            'all_rew': all_rew,
            'all_ret': all_ret
        }

        all_eprews = np.concatenate(all_eprews)
        all_eplens = np.concatenate(all_eplens)

        all_slowdown = np.concatenate(all_slowdown)

        print("Starting training...")

        loss = vanila_pg.train_model(actor, all_data, actor_optim, hp)

        timer_end = time.time()

        print("-------------------------")
        print("Iteration: \t %i" % iteration)
        print("NumBatches: \t %i " % hp.num_ex)
        print("NumEpisodes(Trajs): \t %i" % len(all_eprews))
        print("NumTimesteps: \t %i" % np.sum(all_eplens))
        print("MaxRew: \t %s" % np.max(all_eprews))
        print("Loss: \t %s" % loss)
        print("MeanRew: \t %s +- %s" % (all_eprews.mean(), all_eprews.std()))
        print("MeanSlowDown: \t %s" % np.mean(all_slowdown))
        print("MeanEpisodeLen: \t %s +- %s" % (all_eplens.mean(), all_eplens.std()))
        print("Time this iteration: \t %s seconds" % (timer_end - timer_begin))
        print("Elapsed time: \t %s seconds" % (timer_end - timer_start))
        print("-------------------------")

        writer.add_scalar('MeanRew', all_eprews.mean(), iteration)
        writer.add_scalar('MeanSlowDown', np.mean(all_slowdown), iteration)
        writer.add_scalar('Loss', loss, iteration)

        if iteration > 0 and iteration % hp.output_freq == 0:
            out_path = "models/" + params_path + "_" + str(iteration) + '.pkl'
            torch.save(actor.state_dict(), out_path)
            slow_down_cdf.launch(hp, out_path, render=False, plot=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str,
                        default=
                        'models/imitation-cnn-cnn-lr_0.0001-SJF-simu_len_50-nun_ex_1000-gpu_2400.pkl')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('--rms', dest='rms', action='store_true')
    parser.set_defaults(rms=False)
    parser.add_argument('--cnn', dest='cnn', action='store_true')
    parser.set_defaults(cnn=False)
    parser.add_argument('--big_cnn', dest='big_cnn', action='store_true')
    parser.set_defaults(big_cnn=False)
    # parser.add_argument('--algorithm', type=str, default='PG',
    #                     help='select one of algorithms among Vanilla_PG, NPG, TRPO, PPO')

    args = parser.parse_args()

    # if args.algorithm == "PG":
    #     from pg_travel.deeprm.agent.vanila_pg import train_model
    # else:
    #     print('Mu Hou Yi Hei, Mei Shi Xian')
    #     exit(0)

    hp = Hp()

    hp.gpu = args.gpu
    hp.use_cnn = args.cnn
    hp.use_big_cnn = args.big_cnn
    # hp.simu_len = 50
    # hp.num_ex = 10
    # hp.actor_lr = 0.0001
    # hp.num_epochs = 30000
    # print(args.gpu)
    # print(args.load_model)
    hp.rmsprop = args.rms

    hp.compute_dependent_parameters()

    print("Using gpu: ", hp.gpu)
    if hp.gpu:
        torch.cuda.set_device(1)
        # print("{} GPUs".format(torch.cuda.device_count()))
        id = torch.cuda.current_device()
        print("Its id is {}, name is {}".format(id, torch.cuda.get_device_name(id)))
    # exit(0)
    print("Using RMSprop: ", hp.rmsprop)

    print("Using cnn: {}".format(hp.use_cnn))

    launch(hp, args.load_model)


if __name__ == "__main__":
    main()



