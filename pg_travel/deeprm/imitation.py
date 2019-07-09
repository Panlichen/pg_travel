import torch
import torch.optim as optim
from torch import nn
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
from pg_travel.deeprm.env_simu_sigle import other_agents
from pg_travel.deeprm.agent import vanila_pg
from pg_travel.deeprm import slow_down_cdf


# def add_sample(X, y, idx, X_to_add, y_to_add):
#     X[idx, 0, :] = X_to_add
#     y[idx] = y_to_add


def iterate_minbatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]


def launch(hp, pg_resume=None, render=False):

    print("Preparing trace data...")

    env = Env(hp, render=render)
    if hp.use_cnn:
        print("Using small cnn")
        actor = vanila_pg.ActorConv(hp.network_input_height, hp.network_input_width, hp.channel, hp.network_output_dim)
    elif hp.use_big_cnn:
        actor = vanila_pg.ActorConvBig(hp.network_input_height, hp.network_input_width, hp.channel, hp.network_output_dim)
    else:
        actor = vanila_pg.ActorLinear(hp.network_input_height * hp.network_input_width, hp.network_output_dim, hp)

    if hp.gpu:
        actor = actor.cuda()
    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)

    if pg_resume is not None:
        pass

    evaluate_policy = None

    if hp.evaluate_policy_name == "SJF":
        evaluate_policy = other_agents.get_sjf_action
    elif hp.evaluate_policy_name == "PACKER":
        evaluate_policy = other_agents.get_packer_action
    elif hp.evaluate_policy_name == "SJF-PACKER":
        evaluate_policy = other_agents.get_packer_sjf_action
    else:
        print("Panic: no policy known to evaluate.")
        exit(1)

    # X = np.zeros((hp.simu_len * hp.num_ex * mem_alloc, 1, hp.network_input_height * hp.network_input_width))
    # y = np.zeros(hp.simu_len * hp.num_ex * mem_alloc)

    X = []
    y = []

    for train_ex in range(hp.num_ex):

        env.reset()
        # print("Now is working on Env #{}".format(env.seq_no))

        for _ in range(hp.episode_max_length):
            ob = env.observe()

            a = evaluate_policy(env.machine, env.job_slot)

            X.append(ob)
            y.append(a)

            ob, rew, done, info = env.step(a, repeat=False)

            if done:
                break

    X = np.array(X)
    # X = np.expand_dims(X, axis=1)

    y = np.array(y)
    # y = np.expand_dims(y, axis=1)

    data_size = y.size

    print("Data size is {}".format(data_size))

    num_train = int(0.8 * data_size)
    num_test = int(0.2 * data_size)

    X_train, X_test = X[:num_train], X[num_train:]
    y_train, y_test = y[:num_train], y[num_train:]

    print("Start training...")

    if hp.use_cnn is True:
        net_type = "cnn"
    elif hp.use_big_cnn is True:
        net_type = "big-cnn"
    else:
        net_type = "linear"
    params_path = "imitation-{}-num_nw={}-lr={}-{}-simu_len={}-nun_ex={}-nw_rate={}".\
        format(net_type, hp.num_nw, hp.actor_lr, hp.evaluate_policy_name, hp.simu_len, hp.num_ex, hp.new_job_rate)
    writer = SummaryWriter('runs/' + params_path)

    for epoch in range(hp.num_epochs):

        # In each epoch, do a full pass over the training data;
        train_loss = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()

        for input_batch, target_batch in iterate_minbatches(X_train, y_train, hp.batch_size, shuffle=True):
            # print(hp.DEBUG, y_train.shape)
            # print(hp.DEBUG, target_batch.shape)
            loss, logits = vanila_pg.imitation_train(input_batch, target_batch, actor, actor_optim, hp)
            # exit(0)

            # act_greedy = torch.distributions.Categorical(logits=logits).sample()
            act_greedy = np.argmax(logits, axis=1)  # Deeprm uses greedy policy in imitation learning

            train_loss += loss
            train_acc += np.sum(act_greedy == target_batch) / hp.batch_size

            # print(hp.DEBUG, probs)
            # print(hp.DEBUG, act_greedy)
            # print(hp.DEBUG, target_batch)
            # print(hp.DEBUG, sum(act_greedy == target_batch))
            # exit(0)

            train_batches += 1

        # And a full pass over all test data
        test_loss = 0
        test_acc = 0
        test_batches = 0

        for input_batch, target_batch in iterate_minbatches(X_test, y_test, hp.batch_size, shuffle=False):
            loss, logits = vanila_pg.imitation_test(input_batch, target_batch, actor, hp)
            # act_greedy = torch.distributions.Categorical(logits=logits).sample()
            act_greedy = np.argmax(logits, axis=1)  # Deeprm uses greedy policy in imitation learning
            test_loss += loss
            test_acc += np.sum(act_greedy == target_batch) / hp.batch_size
            test_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(epoch, hp.num_epochs, time.time() - start_time))
        print(" training loss:      \t{:.6f}".format(train_loss / train_batches))
        print(" training accuracy:  \t{:.4f}%".format(train_acc / train_batches * 100))
        print(" test loss:      \t{:.6f}".format(test_loss / test_batches))
        print(" test accuracy:  \t{:.4f}%".format(test_acc / test_batches * 100))

        # writer.add_scalar("Training loss", train_loss / train_batches, epoch)
        # writer.add_scalar("Training accuracy", train_acc / train_batches, epoch)
        # writer.add_scalar("Test loss", test_loss / test_batches, epoch)
        # writer.add_scalar("Test accuracy", test_acc / test_batches, epoch)

        writer.add_scalars("Loss", {"training": train_loss / train_batches, "test": test_loss / test_batches}, epoch)
        writer.add_scalars("accuracy", {"training": train_acc / train_batches, "test": test_acc / test_batches}, epoch)

        if epoch > 0 and epoch % hp.output_freq == 0:
            out_path = "models/" + params_path + "_" + str(epoch) + '.pkl'
            torch.save(actor.state_dict(), out_path)
            slow_down_cdf.launch(hp, out_path, render=False, plot=True)

    print("Imitation Learning done")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('--cnn', dest='cnn', action='store_true')
    parser.set_defaults(cnn=False)
    parser.add_argument('--big_cnn', dest='big_cnn', action='store_true')
    parser.set_defaults(big_cnn=False)
    args = parser.parse_args()

    hp = Hp()
    hp.simu_len = 50
    hp.num_ex = 1000

    hp.gpu = args.gpu
    hp.use_cnn = args.cnn
    print("Using gpu:", hp.gpu)
    if hp.gpu:
        torch.cuda.set_device(0)
        # print("{} GPUs".format(torch.cuda.device_count()))
        id = torch.cuda.current_device()
        print("Its id is {}, name is {}".format(id, torch.cuda.get_device_name(id)))
    # exit(0)

    print("Using cnn: {}".format(hp.use_cnn))

    hp.compute_dependent_parameters()

    pgresume = None
    render = False

    launch(hp)


if __name__ == "__main__":
    main()
