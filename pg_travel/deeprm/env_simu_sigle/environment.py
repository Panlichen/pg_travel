import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class Env:
    def __init__(self, hp, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, end='no_new_job'):

        self.hp = hp
        self.render = render
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.curr_time = 0

        # new work distribution
        self.nw_dist = self.hp.dist.bi_model_dist

        # set up random seed
        if self.hp.unseen:
            np.random.seed(314159)
            torch.manual_seed(314159)
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if nw_len_seqs is None or nw_size_seqs is None:
            # generate new work
            self.nw_len_seqs, self.nw_size_seqs = \
                self.generate_sequence_work(self.hp.simu_len * self.hp.num_ex)
            # NOTE: here the simu_len parameter is self.hp.simu_len * self.hp.num_ex)

            self.workload = np.zeros(self.hp.num_res)

            for i in range(self.hp.num_res):
                self.workload[i] = \
                    np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
                    float(self.hp.res_slot) / \
                    float(len(self.nw_len_seqs))
                # NOTE: here all resources share the same float(self.hp.res_slot), but it should be
                # different, that is, should be float(self.hp.res_slot[i])
                print("Load on # " + str(i) + "resource dimension is " + str(self.workload[i]))

            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                          [self.hp.num_ex, self.hp.simu_len])
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.hp.num_ex, self.hp.simu_len, self.hp.num_res])
            # if use torch.tensor, we can use tensor.view() method

        else:
            self.nw_len_seqs = nw_len_seqs
            self.nw_size_seqs = nw_size_seqs

        self.seq_no = 0  # Now use which example sequence
        self.seq_idx = 0  # The index in that sequence
        # NOTE: the env can see all the job sequence, those two parameters just choose which to USE

        # initialize system
        self.machine = Machine(self.hp)
        self.job_slot = JobSlot(self.hp)
        self.job_backlog = JobBacklog(self.hp)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.hp)

    def generate_sequence_work(self, simu_len):

        # Later I will see whether to use torch.tensor or numpy, or both.

        nw_len_seq = np.zeros(simu_len, dtype=int)
        nw_size_seq = np.zeros((simu_len, self.hp.num_res), dtype=int)

        for i in range(simu_len):
            # We dont want new job to come every moment.
            if np.random.rand() < self.hp.new_job_rate:
                nw_len_seq[i], nw_size_seq[i, :] = self.nw_dist()

        return nw_len_seq, nw_size_seq

    def get_new_job_from_seq(self, seq_no, seq_idx):
        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
        return new_job

    def observe(self):
        image_repr = np.zeros((int(self.hp.network_input_height), int(self.hp.network_input_width)))

        ir_pt = 0

        for i in range(self.hp.num_res):
            # for each resource, the repr contains two parts: (1) the machine status, canvas(for running jobs);
            # (2) the jobs waiting to be scheduled
            image_repr[:, ir_pt:ir_pt + self.hp.res_slot] = self.machine.canvas[i, :, :]
            ir_pt += self.hp.res_slot

            for j in range(self.hp.num_nw):
                if self.job_slot.slot[j] is not None:
                    image_repr[: self.job_slot.slot[j].len, ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = 1
                ir_pt += self.hp.max_job_size

        image_repr[: self.job_backlog.curr_size // self.hp.backlog_width,
                    ir_pt: ir_pt + self.hp.backlog_width] = 1
        if self.job_backlog.curr_size % self.hp.backlog_width > 0:
            image_repr[self.job_backlog.curr_size // self.hp.backlog_width,
                        ir_pt: ir_pt + self.hp.backlog_width] = 1
        ir_pt += self.hp.backlog_width

        image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
                                        float(self.extra_info.max_tracking_time_since_last_job)
        ir_pt += 1

        assert ir_pt == image_repr.shape[1]
        return image_repr

    def plot_state(self):
        plt.figure("screen", figsize=(20, 5))

        skip_row = 0

        for i in range(self.hp.num_res):

            plt.subplot(self.hp.num_res,
                        1 + self.hp.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                        i * (self.hp.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

            plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)

            for j in range(self.hp.num_nw):

                job_slot = np.zeros((self.hp.time_horizon, self.hp.max_job_size))
                if self.job_slot.slot[j] is not None:  # fill in a block of work
                    job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res_vec[i]] = 1

                plt.subplot(self.hp.num_res,
                            1 + self.hp.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                            1 + i * (self.hp.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

                plt.imshow(job_slot, interpolation='nearest', vmax=1)

                if j == self.hp.num_nw - 1:
                    skip_row += 1

        skip_row -= 1
        backlog_width = int(math.ceil(self.hp.backlog_size / float(self.hp.time_horizon)))
        backlog = np.zeros((self.hp.time_horizon, backlog_width))

        backlog[: self.job_backlog.curr_size / backlog_width, : backlog_width] = 1
        backlog[self.job_backlog.curr_size / backlog_width, : self.job_backlog.curr_size % backlog_width] = 1

        plt.subplot(self.hp.num_res,
                    1 + self.hp.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.hp.num_nw + 1 + 1)

        plt.imshow(backlog, interpolation='nearest', vmax=1)

        plt.subplot(self.hp.num_res,
                    1 + self.hp.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.hp.num_res * (self.hp.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

        extra_info = np.ones((self.hp.time_horizon, 1)) * \
                     self.extra_info.time_since_last_new_job / \
                     float(self.extra_info.max_tracking_time_since_last_job)

        plt.imshow(extra_info, interpolation='nearest', vmax=1)

        plt.show()     # manual
        # plt.pause(0.01)  # automatic

    def get_reward(self):

        reward = 0
        for j in self.machine.running_job:
            reward += self.hp.delay_penalty / float(j.len)

        for j in self.job_slot.slot:
            if j is not None:
                reward += self.hp.hold_penalty / float(j.len)

        for j in self.job_backlog.backlog:
            if j is not None:
                reward += self.hp.dismiss_penalty / float(j.len)

        return reward

    def step(self, a, repeat=False):
        status = None
        done = False
        reward = 0
        info = None

        if a == self.hp.num_nw:  # explicit void action
            status = 'MoveOn'
        elif self.job_slot.slot[a] is None:  # implicit void action
            status = 'MoveOn'
        else:
            allocated = self.machine.allocate_job(self.job_slot.slot[a], self.curr_time)
            if not allocated:  # implicit void action
                status = 'MoveOn'
            else:
                status = 'Allocate'

        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            # add new jobs
            self.seq_idx += 1

            if self.end == "no_new_job":
                if self.seq_idx >= self.hp.simu_len:
                    done = True
            elif self.end == "all_done":
                if self.seq_idx >= self.hp.simu_len and \
                        len(self.machine.running_job) == 0 and \
                        all(s is None for s in self.job_slot.slot) and \
                        all(s is None for s in self.job_backlog.backlog):
                    done = True
                elif self.curr_time >= self.hp.episode_max_length:  # run too long, force termination
                    done = True

            if not done:

                if self.seq_idx < self.hp.simu_len:
                    new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)

                    if new_job.len > 0:  # a new job comes

                        to_backlog = True

                        for i in range(self.hp.num_nw):
                            if self.job_slot.slot[i] is None:
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                to_backlog = False
                                break

                        if to_backlog:
                            if self.job_backlog.curr_size < self.hp.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.record[new_job.id] = new_job
                            else:
                                print("BACKLOG FULL, A JOB DISCARDED")
                                # This means there is a huge job running, both job_slot and backlog is full.
                                # But maybe we should not simply discard the picked job??
                        self.extra_info.new_job_comes()
                # NO ELSE. Otherwise, end of new job sequence, i.e. no new jobs, this case means waiting for "all_done"

            reward = self.get_reward()

        elif status == "Allocate":
            self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]
            self.job_slot.slot[a] = None

            # dequeue backlog
            if self.job_backlog.curr_size > 0:
                self.job_slot.slot[a] = self.job_backlog.backlog[0]
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1

        ob = self.observe()

        info = self.job_record

        if done:
            self.seq_idx = 0

            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.hp.num_ex

            self.reset()

        if self.render:
            self.plot_state()

        return ob, reward, done, info
                
    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.hp)
        self.job_slot = JobSlot(self.hp)
        self.job_backlog = JobBacklog(self.hp)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.hp)


class Job:
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id  # it's order added to the job_record.record, job_id=len(self.job_record.record), Job.id;
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time  # the time a job is added to JobSlot or backlog
        self.start_time = -1  # the time a job is scheduled
        self.finish_time = -1  # start_time + job_len


class JobSlot:
    def __init__(self, hp):
        self.slot = [None] * hp.num_nw


class JobBacklog:
    def __init__(self, hp):
        self.backlog = [None] * hp.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Machine:
    def __init__(self, hp):
        self.num_res = hp.num_res
        self.time_horizon = hp.time_horizon
        self.res_slot = hp.res_slot  # size of each resource

        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot  # init avbl as full

        self.running_job = []

        # color for graphical representation
        self.colormap = np.arange(1 / float(hp.job_num_cap), 1, 1 / float(hp.job_num_cap))
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((hp.num_res, hp.time_horizon, hp.res_slot))

    def allocate_job(self, job, curr_time):

        allocated = False

        for t in range(0, self.time_horizon - job.len):
            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec

            if np.all(new_avbl_res[:] >= 0):
                allocated = True

                self.avbl_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len

                self.running_job.append(job)

                # update graphical representation

                used_color = np.unique(self.canvas[:])
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time

                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0]
                        self.canvas[res, i, avbl_slot[: job.res_vec[res]]] = new_color

                break

        return allocated

    def time_proceed(self, curr_time):
        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot  # add a line where resource is full

        for job in self.running_job:

            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # update graphical representation
        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0


class ExtraInfo:
    def __init__(self, hp):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = hp.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1











