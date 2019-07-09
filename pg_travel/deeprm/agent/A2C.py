import numpy as np
import torch
from pg_travel.deeprm.hparams import HyperParams as Hp


hp = Hp()


def get_returns(all_ret):
    all_ret = torch.Tensor(all_ret)
    norm_rets = (all_ret - all_ret.mean()) / all_ret.std()
    return norm_rets


def train_critic(critic, all_ob, norm_rets, critic_optim):
    criterion = torch.nn.MSELoss()
    n = all_ob.shape[0]
    arr = np.arange(n)

    for epoch in range(5):
        np.random.shuffle(arr)

        for i in range(n // hp.batch_size):
            batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            # TODO: input shape is (batch_size, 1, input_height, input_width), let's see whether it is ok
            inputs = torch.Tensor(all_ob)[batch_index]
            target = norm_rets.view(-1, 1, 1)[batch_index]

            values = critic(inputs)
            # print(inputs.shape)
            # print(target.shape)
            # print(values.shape)
            loss = criterion(values, target)
            critic_optim.zero_grad()
            loss.backward()
            critic_optim.step()


def get_loss(actor, returns, all_ob, all_action):
    logits, _ = actor(torch.Tensor(all_ob))
    log_prob = torch.distributions.Categorical(logits=logits).log_prob(torch.Tensor(all_action))
    returns = returns.unsqueeze(1)

    objective = returns * log_prob
    objective = objective.mean()
    return - objective


def train_actor(actor, returns, all_ob, all_action, actor_optim):
    loss = get_loss(actor, returns, all_ob, all_action)
    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()


def train_model(actor, critic, data, actor_optim, critic_optim):
    all_ob = data['all_ob']
    all_action = data['all_action']
    all_ret = data['all_ret']
    all_adv = data['all_adv']

    # norm_rets = get_returns(all_ret)
    norm_rets = get_returns(all_adv)

    train_critic(critic, all_ob, norm_rets, critic_optim)
    train_actor(actor, norm_rets, all_ob, all_action, actor_optim)

    return norm_rets










