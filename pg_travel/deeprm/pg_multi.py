import torch
import torch.optim as optim
import argparse
import numpy as np
import time
from tensorboardX import SummaryWriter
from collections import deque

from torch.multiprocessing import Process
from torch.multiprocessing import Manager
import torch.multiprocessing as mp

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


def get_traj_worker(actor, env, hp, result):
    with torch.no_grad():
        if hp.gpu:
            actor = actor.cuda()
        trajs = []
        for i in range(hp.num_seq_per_batch):
            traj = get_traj(actor, env, hp)
            trajs.append(traj)
        all_ob = concatenate_all_ob(trajs, hp)

        # Compute discounted sums of rewards, in fact, to reflect the slowdown, discount should be 1
        rets = [discount(traj['reward'], hp.gamma) for traj in trajs]

        maxlen = max(len(ret) for ret in rets)
        padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

        # Compute time-dependent baseline
        baseline = np.mean(padded_rets, axis=0)

        # Compute advantage function
        advs = [ret - baseline[:len(ret)] for ret in rets]
        all_adv = np.concatenate(advs)

        all_action = np.concatenate([traj['action'] for traj in trajs])
        # TODO:
        # all_rew.append(np.concatenate([traj['reward'] for traj in trajs]))

        # episode total rewards, maybe only useful for print
        all_eprews = np.array([discount(traj['reward'], hp.gamma)[0] for traj in trajs])
        # episode lengths
        all_eplens = np.array([len(traj['reward']) for traj in trajs])

        # All Job Stat
        enter_time, finish_time, job_len = process_all_info(trajs)
        finished_idx = (finish_time >= 0)
        all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

        result.append({
            "all_ob": all_ob,
            "all_action": all_action,
            "all_adv": all_adv,
            "all_eprews": all_eprews,
            "all_eplens": all_eplens,
            "all_slowdown": all_slowdown
        })


def launch(hp, load_model):

    print("Preparing for workers...")

    actors = []
    envs = []

    torch.manual_seed(42)

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(hp, seed=42)

    for ex in range(hp.num_ex):
        print("--prepare for env #{}--".format(ex))
        env = Env(hp, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs)
        env.seq_no = ex
        envs.append(env)

    for p in range(hp.parallel_limit + 1):  # last worker for updating parameters
        print("--prepare for worker #{}--".format(p))
        if hp.use_cnn:
            print("Using small cnn")
            actor = vanila_pg.ActorConv(hp.network_input_height, hp.network_input_width, hp.channel,
                                        hp.network_output_dim)
        elif hp.use_big_cnn:
            actor = vanila_pg.ActorConvBig(hp.network_input_height, hp.network_input_width, hp.channel,
                                           hp.network_output_dim)
        else:
            actor = vanila_pg.ActorLinear(hp.network_input_height * hp.network_input_width, hp.network_output_dim, hp)

        actors.append(actor)

    if hp.gpu:
        actors[hp.parallel_limit] = actors[hp.parallel_limit].cuda()

    total_num = sum(p.numel() for p in actors[hp.parallel_limit].parameters())
    trainable_num = sum(p.numel() for p in actors[hp.parallel_limit].parameters() if p.requires_grad)
    print('Total: ', total_num, 'Trainable:', trainable_num)

    # last worker for updating parameters
    if hp.rmsprop:
        actor_optim = optim.RMSprop(actors[hp.parallel_limit].parameters(), lr=hp.actor_lr, alpha=0.9, eps=1e-9)
    else:
        actor_optim = optim.Adam(actors[hp.parallel_limit].parameters(), lr=hp.actor_lr)

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

    params_path = "vanilla-pg-multi-{}-lr={}-simu_len={}-nun_ex={}-num_nw={}-nw_rate={}".\
        format(net_type, hp.actor_lr, hp.simu_len, hp.num_ex, hp.num_nw, hp.new_job_rate)

    writer = SummaryWriter('runs/' + params_path)

    timer_start = time.time()

    for iteration in range(hp.num_epochs):

        timer_begin = time.time()

        ps = []
        manager = Manager()
        manager_result = manager.list([])

        ex_indices = [i for i in range(hp.num_ex)]  # each epoch has its own ex order
        np.random.shuffle(ex_indices)

        all_eprews = []
        eprews = []
        eplens = []
        all_loss = []
        all_slowdown = []

        p_counter = 0
        for ex in range(hp.num_ex):
            ex_idx = ex_indices[ex]
            p = Process(target=get_traj_worker,
                        args=(actors[p_counter], envs[ex_idx], hp, manager_result, ))
            ps.append(p)
            p_counter += 1

            if p_counter >= hp.parallel_limit or ex == hp.num_ex - 1:
                p_counter = 0
                for p in ps:
                    p.start()
                for p in ps:
                    p.join()
                result = []  # convert list from shared memory
                for r in manager_result:
                    result.append(r)

                ps = []
                manager_result = manager.list([])

                all_ob = concatenate_all_ob_across_examples([r["all_ob"] for r in result], hp)
                all_action = np.concatenate([r["all_action"] for r in result])
                all_adv = np.concatenate([r["all_adv"] for r in result])

                all_data = {
                    'all_ob': all_ob,
                    'all_action': all_action,
                    'all_adv': all_adv,
                }

                eprews.extend(np.concatenate([r["all_eprews"] for r in result]))
                all_eprews.extend([r["all_eprews"] for r in result])

                eplens.extend(np.concatenate([r["all_eplens"] for r in result]))

                all_slowdown.extend(np.concatenate([r["all_slowdown"] for r in result]))

                loss = vanila_pg.train_model(actors[hp.parallel_limit], all_data, actor_optim, hp)

                if hp.gpu:
                    loss = loss.cpu().detach().numpy()
                else:
                    loss = loss.detach().numpy()

                all_loss.append(loss)

                actor_param = actors[hp.parallel_limit].state_dict()
                for i in range(hp.parallel_limit):
                    actors[i].load_state_dict(actor_param)

        timer_end = time.time()

        print("-------------------------")
        print("Iteration: \t %i" % iteration)
        print("NumBatches: \t %i " % hp.num_ex)
        print("NumEpisodes(Trajs): \t %i" % len(eprews))
        print("NumTimesteps: \t %i" % np.sum(eplens))
        print("MaxRew: \t %s" % np.average([np.max(eprew) for eprew in all_eprews]))
        print("Loss: \t %s" % np.average(all_loss))
        print("MeanRew: \t %s +- %s" % (np.average(eprews), np.std(eprews)))
        print("MeanSlowDown: \t %s" % np.mean(all_slowdown))
        print("MeanEpisodeLen: \t %s +- %s" % (np.average(eplens), np.std(eplens)))
        print("Time this iteration: \t %s seconds" % (timer_end - timer_begin))
        print("Elapsed time: \t %s seconds" % (timer_end - timer_start))
        print("-------------------------")

        writer.add_scalar('MeanRew', np.average(eprews), iteration)
        writer.add_scalar('MeanSlowDown', np.average(all_slowdown), iteration)
        writer.add_scalar('Loss', np.average(all_loss), iteration)

        if iteration > 0 and iteration % hp.output_freq == 0:
            out_path = "models/" + params_path + "_" + str(iteration) + '.pkl'
            torch.save(actors[hp.parallel_limit].state_dict(), out_path)
            slow_down_cdf.launch(hp, out_path, render=False, plot=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str,
                        default=
                        'models/imitation-cnn-num_nw=10-lr=0.0001-SJF-simu_len=50-nun_ex=1000-nw_rate=0.4_60.pkl')
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

    mp.set_start_method('spawn')

    launch(hp, args.load_model)


if __name__ == "__main__":
    main()



