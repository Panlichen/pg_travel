import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEBUG = "~~DEBUG~~"


def calc_cnn_size(l_ori, k_s, s, p):
    return (l_ori - k_s + 2 * p) / s + 1


class ActorConv(nn.Module):
    def __init__(self, input_height, input_width, channel, num_output):
        super(ActorConv, self).__init__()
        self.channel = channel  # channel = 16
        self.input_height = input_height
        self.input_width = input_width

        self.k_s = 2
        self.s = 1
        self.p = 1
        self.fc_in_size = self.channel * \
                          calc_cnn_size(input_height, self.k_s, self.s, self.p) * \
                          calc_cnn_size(input_width, self.k_s, self.s, self.p)

        self.conv = nn.Conv2d(1, channel, kernel_size=self.k_s, stride=self.s, padding=self.p)
        self.fc1 = nn.Linear(self.fc_in_size, num_output)

        torch.nn.init.constant_(self.conv.bias, 0.)
        torch.nn.init.normal_(self.conv.weight, mean=0., std=0.01)
        torch.nn.init.constant_(self.fc1.bias, 0.)
        torch.nn.init.normal_(self.fc1.weight, mean=0., std=0.01)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.shape[0], self.fc_in_size)
        logits = self.fc1(x)
        return logits


class ActorConvBig(nn.Module):
    def __init__(self, input_height, input_width, channel1, channel2, num_output):
        self.channel1 = channel1  # channel1 = 8
        self.channel2 = channel2  # channel2 = 16
        self.input_height = input_height
        self.input_width = input_width
        self.num_output = num_output
        super(ActorConvBig, self).__init__()

        self.k_s1 = 4
        self.s1 = 2
        self.p1 = 2
        self.k_s2 = 2
        self.s2 = 1
        self.p2 = 1
        cnn_h1 = calc_cnn_size(input_height, self.k_s1, self.s1, self.p1)
        cnn_h2 = calc_cnn_size(cnn_h1, self.k_s2, self.s2, self.p2)
        cnn_w1 = calc_cnn_size(input_width, self.k_s1, self.s1, self.p1)
        cnn_w2 = calc_cnn_size(cnn_w1, self.k_s2, self.s2, self.p2)
        self.fc_in_size = self.channel2 * cnn_h2 * cnn_w2
        self.conv1 = nn.Conv2d(1, channel1, kernel_size=4, stride=2, padding=2)
        self.conv2 = nn.Conv2d(channel1, channel2, kernel_size=2, stride=1, padding=1)
        self.fc1 = nn.Linear(self.fc_in_size, 20)
        self.fc2 = nn.Linear(20, self.num_output)

        torch.nn.init.constant_(self.conv1.bias, 0.)
        torch.nn.init.normal_(self.conv1.weight, mean=0., std=0.01)
        torch.nn.init.constant_(self.fc1.bias, 0.)
        torch.nn.init.normal_(self.fc1.weight, mean=0., std=0.01)
        torch.nn.init.constant_(self.conv2.bias, 0.)
        torch.nn.init.normal_(self.conv2.weight, mean=0., std=0.01)
        torch.nn.init.constant_(self.fc2.bias, 0.)
        torch.nn.init.normal_(self.fc2.weight, mean=0., std=0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x.view(x.size(0), self.fc_in_size)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class ActorLinear(nn.Module):
    def __init__(self, num_input, num_output, hp):
        super(ActorLinear, self).__init__()
        self.fc1 = nn.Linear(num_input, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, num_output)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        single = False
        if x.dim() == 2:  # for single ob
            single = True
            x = x.view(-1)
        elif x.dim() == 3: # for batch ob
            x = x.view(x.shape[0], -1)
            # print(x.shape)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits


def get_returns(raw_adv):
    norm_adv = (raw_adv - raw_adv.mean()) / raw_adv.std()
    return norm_adv


def get_loss(actor, adv, ob, action):
    # print(DEBUG + "ob.shape", ob.shape)

    logits = actor(ob)
    # print(DEBUG + "log_prob_all.shape", log_prob_all.shape)
    # print(DEBUG + "action.shape", action.shape)
    # print(DEBUG + "logits.shape", logits.shape)

    log_prob = torch.distributions.Categorical(logits=logits).log_prob(action)
    # print(DEBUG + "log_prob.shape", log_prob.shape)
    #
    # print(DEBUG + "adv.shape", adv.shape)
    #
    # exit(0)

    objective = adv * log_prob
    objective = objective.mean()
    return - objective


def train_actor(actor, adv, ob, action, actor_optim):
    loss = get_loss(actor, adv, ob, action)
    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()
    return loss


def train_model(actor, data, actor_optim, hp):
    all_ob = torch.Tensor(data['all_ob'])

    if hp.use_big_cnn or hp.use_cnn:
        all_ob.unsqueeze_(1)

    all_action = torch.Tensor(data['all_action'])
    all_adv = torch.Tensor(data['all_adv'])

    assert all_ob.shape[0] == all_action.shape[0] and all_ob.shape[0] == all_adv.shape[0]
    num_data = all_ob.shape[0]

    batch_count = 0
    all_loss = 0

    for data_idx in range(0, num_data - hp.rl_batch_size, hp.rl_batch_size):
        ob_batch = all_ob[data_idx: data_idx + hp.rl_batch_size]
        action_batch = all_action[data_idx: data_idx + hp.rl_batch_size]
        adv_batch = all_adv[data_idx: data_idx + hp.rl_batch_size]

        if hp.gpu:
            ob_batch = ob_batch.cuda()
            action_batch = action_batch.cuda()
            adv_batch = adv_batch.cuda()

        # print(hp.DEBUG, ob_batch.shape)

        norm_adv_batch = get_returns(adv_batch)

        loss = train_actor(actor, adv_batch, ob_batch, action_batch, actor_optim)

        batch_count += 1
        all_loss += loss

    return all_loss / batch_count


def imitation_train(inputs, targets, actor, actor_optim, hp):
    # print(DEBUG+"inputs", inputs.shape)
    inputs = torch.Tensor(inputs)
    if hp.use_cnn:
        inputs.unsqueeze_(1)
    targets = torch.Tensor(targets).long()

    celoss = nn.CrossEntropyLoss()

    if hp.gpu:
        inputs = inputs.cuda()
        targets = targets.cuda()
        celoss = celoss.cuda()

    # print("~~DEBUG~~", inputs.shape)
    logits = actor(inputs)

    loss = celoss(logits, targets)

    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()

    if hp.gpu:
        return loss.cpu().detach().numpy(), logits.cpu().detach().numpy()
    return loss.detach().numpy(), logits.detach().numpy()


def imitation_test(inputs, targets, actor, hp):
    inputs = torch.Tensor(inputs)
    if hp.use_cnn:
        inputs.unsqueeze_(1)
    targets = torch.Tensor(targets).long()
    celoss = nn.CrossEntropyLoss()

    if hp.gpu:
        inputs = inputs.cuda()
        targets = targets.cuda()
        celoss = celoss.cuda()

    with torch.no_grad():
        logits = actor(inputs)
        # print(DEBUG, logits.shape)
        # print(DEBUG, targets.shape)
        loss = celoss(logits, targets)

    if hp.gpu:
        return loss.cpu().numpy(), logits.cpu().numpy()
    return loss.numpy(), logits.numpy()






