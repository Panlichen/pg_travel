import math
from pg_travel.deeprm.env_simu_sigle.job_distribution import Dist


class HyperParams:
    def __init__(self):
        #
        # self.lamda = 0.98
        #
        # self.channel = 16
        # self.big_channel1 = 8
        # self.big_channel2 = 16
        # self.critic_lr = 0.0003
        #
        # self.batch_size = 64
        # self.l2_rate = 0.001
        # self.max_kl = 0.01
        # self.clip_param = 0.2

        self.num_epochs = 1000  # number of training epochs
        self.simu_len = 50  # num of jobs in a sequence
        self.num_ex = 100  # number of sequences

        self.output_freq = 10  # interval of output and save parameters.

        self.num_seq_per_batch = 10  # number of sequences to compute baseline
        self.episode_max_length = 200  # enforcing an artificial terminal

        self.num_res = 2  # number of resources in the system
        self.num_nw = 10  # maximum allowed number of work in the scheduling window

        self.time_horizon = 20  # number of time steps in the graph
        self.max_job_len = 15  # maximum duration of new jobs
        self.res_slot = 10  # maximum number of available resource slots
        self.max_job_size = self.res_slot  # maximum resource request of new work

        self.backlog_size = 60  # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 40  # maximum number of distinct colors in current work graph

        self.new_job_rate = 0.4  # lambda in new job arrival Poisson Process

        self.gamma = 1

        # distribution fro new job arrival
        self.dist = Dist(self.num_res, self.max_job_size, self.max_job_len)

        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))

        # backlog = [None] * pa.backlog_size
        self.network_input_height = self.time_horizon
        # simu_len: 总的任务数；num_nw: 同时被调度的任务数（窗口大小）
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.unseen = False  # change random seed to generate unseen example

        self.actor_lr = 0.0001
        self.hidden = 20

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "SJF"

        # for debug
        self.DEBUG = "~~DEBUG~~"

        # new for RL
        self.rl_batch_size = 200
        self.gpu = False

        # for RMSprop
        self.rmsprop = False
        self.rms_rho = 0.9
        self.rms_eps = 1e-9

        # new for CNN
        self.use_cnn = False
        self.use_big_cnn = False
        self.channel = 16
        self.big_channel1 = 8
        self.big_channel2 = 16

        # for multi-process
        self.parallel_limit = 10

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))

        # backlog = [None] * pa.backlog_size
        self.network_input_height = self.time_horizon
        # simu_len: 总的任务数；num_nw: 同时被调度的任务数（窗口大小）
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job
