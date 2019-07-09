import torch
import torch.optim as optim
import argparse
import numpy as np
import time
from tensorboardX import SummaryWriter
from collections import deque

import sys
sys.path.append('../..')

from pg_travel.deeprm import model
from pg_travel.deeprm.hparams import HyperParams as Hp
from pg_travel.deeprm.env_simu_sigle.environment import Env
from pg_travel.deeprm.env_simu_sigle import job_distribution
from pg_travel.deeprm.agent import vanila_pg

actor = torch.load('models/net_imitation_SJF_100_full.pkl')
torch.save(actor.state_dict(), "models/net_imitation_SJF_100.pkl")