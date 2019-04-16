import torch
import torch.optim as optim
import argparse
import numpy as np
from collections import deque

from pg_travel.deeprm import model
from pg_travel.deeprm.hparams import HyperParams as Hp
from pg_travel.deeprm.env_simu_sigle.environment import Env
from pg_travel.deeprm.env_simu_sigle import job_distribution


# variables like actor, env, etc was defined under main, so they are thought to be global and warned here
def get_traj(actor, env, episode_max_length, render=False):
    """
    Run agent-environment loop for one whole episode(trajectory)
    :param actor:
    :param env:
    :param episode_max_length:
    :param render:
    :return: dictionary of results
    """
    env.reset()
    obs = []
    acts = []
    rews = []
    info = []

    ob = env.observe()

    for _ in range(episode_max_length):
        # TODO: may need adjust dtype and size for input
        _, act_prob = actor(torch.Tensor(ob))
        a = torch.multinomial(act_prob, 1)

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

    all_ob = np.zeros((timesteps_total, 1, hp.network_input_height, hp.network_input_width))

    timesteps = 0
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['reward'])):
            all_ob[timesteps, 0, :, :] = trajs[i]['ob'][j]
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

    all_ob_contact = np.zeros((total_samp, 1, hp.network_input_height, hp.network_input_width))

    total_samp = 0

    for i in range(num_ex):
        prev_samp = total_samp
        total_samp += all_ob[i].shape[0]
        all_ob_contact[prev_samp : total_samp, :, :, :] = all_ob[i]

    return all_ob_contact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--algorithm', type=str, default='PG',
                        help='select one of algorithms among Vanilla_PG, NPG, TRPO, PPO')
    # TODO: args should have one about end type, 'no_new_job' and others

    args = parser.parse_args()

    if args.algorithm == "PG":
        from pg_travel.deeprm.agent.vanila_pg import train_model
    else:
        print('Mu Hou Yi Hei, Mei Shi Xian')
        exit(0)

    hp = Hp()
    env = Env(hp, seed=368)
    torch.manual_seed(368)

    actor = model.ActorLinear(hp.network_input_height * hp.network_input_width, hp.num_nw)
    critic = model.CriticLinear(hp.network_input_height * hp.network_input_width)
    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr)

    # TODO: implement supervised learning to initialize the model

    if args.load_model is not None:
        # TODO: load pre-trained model
        pass

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(hp, seed=368)

    envs = []
    for ex in range(hp.num_ex):
        print('-prepare for env-', ex)

        env = Env(hp, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs, render=False)
        env.seq_no = ex
        envs.append(env)

    # TODO: use batch_size learners and multi-thread?
    # TODO: multi-thread and deque() vs multi-process and manager?

    for iteration in range(hp.num_epochs):

        # TODO: maybe with actor.no_grad() is more appropriate?
        actor.eval(), critic.eval()

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

        ex_indices = [i for i in range(hp.num_ex)]
        np.random.shuffle(ex_indices)

        for ex in range(hp.num_ex):

            ex_idx = ex_indices[ex]  # useless but interesting?

            # Collect trajectories until we get time_steps_per_batch total timesteps
            trajs = []

            # TODO: maybe we can adjust the use of "ex" and "batch" to make it more clear
            for i in range(hp.num_seq_per_batch):  # here uses "per batch", there is many batches in an epoch
                traj = get_traj(actor, envs[ex_idx], hp.episode_max_length)
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
            # TODO: maybe can be optimized, now there is repetitive computation.
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

        actor.train(), critic.train()

        train_model(actor, critic, all_data, actor_optim, critic_optim)


