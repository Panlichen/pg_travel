import torch
import torch.nn as nn
import torch.nn.functional as F
from pg_travel.deeprm.hparams import HyperParams as Hp


hp = Hp()


def calc_cnn_size(l_ori, k_s, s, p):
    return (l_ori - k_s + 2 * p) / s + 1


class ActorLinear(nn.Module):
    def __init__(self, num_input, num_output):
        super(ActorLinear, self).__init__()
        self.fc1 = nn.Linear(num_input, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, num_output)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        # should "view" the input here
        # TODO: but will it work during batch operation?
        # print(x.shape)
        if x.dim() == 2:  # for single ob
            x = x.view(-1)
        elif x.dim() == 4:  # for unsqueezed batch ob
            x = x.view(x.shape[0], x.shape[1], -1)
            # print(x.shape)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        logits = self.fc3(x)
        probs = F.softmax(logits)
        return logits, probs


class CriticLinear(nn.Module):
    def __init__(self, num_input):
        super(CriticLinear, self).__init__()
        self.fc1 = nn.Linear(num_input, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        # should "view" the input here
        x = x.view(x.shape[0], x.shape[1], -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


class CriticConv(nn.Module):
    def __init__(self, input_height, input_width, channel):
        super(CriticConv, self).__init__()
        self.channel = channel
        self.input_height = input_height
        self.input_width = input_width

        self.k_s = 2
        self.s = 1
        self.p = 1
        self.fc_in_size = self.channel * \
                          calc_cnn_size(input_height, self.k_s, self.s, self.p) * \
                          calc_cnn_size(input_width, self.k_s, self.s, self.p)

        self.conv = nn.Conv2d(1, channel, kernel_size=self.k_s, stride=self.s, padding=self.p)
        self.fc1 = nn.Linear(channel * input_height * input_width, 1)

        torch.nn.init.constant_(self.conv.bias, 0.)
        torch.nn.init.normal_(self.conv.weight, mean=0., std=0.01)
        torch.nn.init.constant_(self.fc1.bias, 0.)
        torch.nn.init.normal_(self.fc1.weight, mean=0., std=0.01)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), self.fc_in_size)  # x.size(0) is the batch size
        v = self.fc1(x)
        return v


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
        probs = F.softmax(logits)
        return logits, probs


class CriticConvBig(nn.Module):
    def __init__(self, input_height, input_width, channel1, channel2):
        self.channel1 = channel1
        self.channel2 = channel2
        self.input_height = input_height
        self.input_width = input_width
        super(CriticConvBig, self).__init__()

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
        self.fc2 = nn.Linear(20, 1)

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
        v = self.fc2(x)
        return v


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
        x = x.view(x.size(0), self.fc_in_size)
        logits = self.fc1(x)
        probs = F.softmax(logits)
        return logits, probs
