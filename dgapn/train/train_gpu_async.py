import os
import gym
import random
import logging
import numpy as np
from rdkit import Chem
from collections import deque, OrderedDict
from copy import deepcopy

import torch
import torch.multiprocessing as tmp
from torch.utils.tensorboard import SummaryWriter

from dgapn.model import DGAPN, save_DGAPN

from dgapn.reward import get_reward

from dgapn.utils.general_utils import close_logger, deque_to_csv
from dgapn.utils.graph_utils import mols_to_pyg_batch
from dgapn.utils.rl_utils import Memory, Log, Scheduler

#####################################################
#                     SUBPROCESS                    #
#####################################################

class Sampler(tmp.Process):
    def __init__(self, args, env, model, lock, task_queue, result_queue, episode_count, sample_count,
                max_episodes, max_timesteps, update_timesteps):
        super(Sampler, self).__init__()
        self.args = args
        self.lock = lock
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.episode_count = episode_count
        self.sample_count = sample_count

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.update_timesteps = update_timesteps

        self.env = env
        self.model = model

        self.memory = Memory()
        self.log = Log()

        self.scheduler = Scheduler(args.innovation_reward_update_cutoff, args.iota, weight_main=False)

    def run(self):
        proc_name = self.name
        pid = self.pid
        torch.manual_seed(pid)

        self.args.run_id = self.args.run_id + proc_name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting ' % proc_name)
                self.task_queue.task_done()
                break

            update_count = next_task
            self.memory.clear()
            self.log.clear()

            print('%s: Sampling ' % proc_name)
            state, candidates, done = self.env.reset()

            while self.sample_count.value < self.update_timesteps and self.episode_count.value < self.max_episodes:
                for t in range(1, self.max_timesteps+1):
                    # Running policy:
                    state_emb, candidates_emb, action_logprob, action = self.model.select_action(
                        mols_to_pyg_batch(state, self.model.emb_3d, device=self.model.device),
                        mols_to_pyg_batch(candidates, self.model.emb_3d, device=self.model.device))
                    self.memory.states.append(state_emb[0])
                    self.memory.candidates.append(candidates_emb)
                    self.memory.states_next.append(candidates_emb[action])
                    self.memory.actions.append(action)
                    self.memory.logprobs.append(action_logprob)

                    state, candidates, done = self.env.step(action)

                    reward = 0
                    if (t==self.max_timesteps) or done:
                        main_reward = get_reward(state, reward_type=self.args.reward_type, args=self.args)
                        reward = self.scheduler.main_weight(update_count) * main_reward
                        done = True
                    if (self.args.iota > 0 and update_count < self.args.innovation_reward_update_cutoff):
                        inno_reward = self.model.get_inno_reward(mols_to_pyg_batch(state, self.model.emb_3d, device=self.model.device))
                        reward += self.scheduler.guide_weight(update_count) * inno_reward

                    # Saving rewards and terminals:
                    self.memory.rewards.append(reward)
                    self.memory.terminals.append(done)

                    if done:
                        break

                with self.lock:
                    self.sample_count.value += t
                    self.episode_count.value += 1

                self.log.ep_lengths.append(t)
                self.log.ep_rewards.append(sum(self.memory.rewards[-t:]))
                self.log.ep_main_rewards.append(main_reward)
                self.log.ep_mols.append(Chem.MolToSmiles(state))

            self.result_queue.put(Result(self.memory, self.log))
            self.task_queue.task_done()
        return

'''
class Task(object):
    def __init__(self, update_count):
        self.update_count = update_count
    def __call__(self):
        return self.update_count
    def __str__(self):
        return '%d' % self.update_count
'''

class Result(object):
    def __init__(self, memory, log):
        self.memory = memory
        self.log = log
    def __call__(self):
        return (self.memory, self.log)

#####################################################
#                   TRAINING LOOP                   #
#####################################################

def train_gpu_async(args, env, model, manager, writer=None, save_dir=None):
    # initiate subprocesses
    lock = manager.Lock()
    tasks = manager.JoinableQueue()
    results = manager.Queue()
    episode_count = manager.Value("i", 0)
    sample_count = manager.Value("i", 0)

    print('Creating %d processes' % args.nb_procs)
    workers = [Sampler(args, env, model, lock, tasks, results, episode_count, sample_count,
                args.max_episodes, args.max_timesteps, args.update_timesteps) for i in range(args.nb_procs)]
    for w in workers:
        w.start()

    update_count = 0
    save_counter = 0
    log_counter = 0

    running_length = 0
    running_reward = 0
    running_main_reward = 0

    memory = Memory()
    log = Log()
    rewbuffer_env = deque(maxlen=100)
    molbuffer_env = deque(maxlen=10000)
    # training loop
    i_episode = 0
    while i_episode < args.max_episodes:
        logging.info("\n\nCollecting rollouts")
        model.to_device(torch.device("cpu"))
        # Enqueue jobs
        for i in range(args.nb_procs):
            tasks.put(update_count)
        # Wait for all of the tasks to finish
        tasks.join()
        # Start unpacking results
        for i in range(args.nb_procs):
            result = results.get()
            m, l = result()
            memory.extend(m)
            log.extend(l)

        i_episode += episode_count.value

        # log results
        for i in reversed(range(episode_count.value)):
            running_length += log.ep_lengths[i]
            running_reward += log.ep_rewards[i]
            running_main_reward += log.ep_main_rewards[i]

            rewbuffer_env.append(log.ep_main_rewards[i])
            molbuffer_env.append(log.ep_mols[i])

            if writer is not None:
                # write to Tensorboard
                writer.add_scalar("EpMainRew", log.ep_main_rewards[i], i_episode - 1)
                writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode - 1)
        log.clear()

        # update model
        logging.info("\nUpdating model @ episode %d..." % i_episode)
        model.update(memory)
        memory.clear()
        update_count += 1

        deque_to_csv(molbuffer_env, os.path.join(save_dir, 'mol_dgapn.csv'), mode='a')
        molbuffer_env.clear()

        save_counter += episode_count.value
        log_counter += episode_count.value

        if save_dir is not None:
            # save if solved
            if np.mean(rewbuffer_env) > args.solved_reward:
                save_DGAPN(model, os.path.join(save_dir, 'solved_dgapn.pt'))

            # save every save_interval episodes
            if save_counter >= args.save_interval:
                save_DGAPN(model, os.path.join(save_dir, '{:05d}_dgapn.pt'.format(i_episode)))
                save_counter = 0

            # save running model
            save_DGAPN(model, os.path.join(save_dir, 'running_dgapn.pt'))

        if log_counter >= args.log_interval:
            logging.info('Episode {} \t Avg length: {:4.2f} \t Avg reward: {:5.3f} \t Avg main reward: {:5.3f}'.format(
                i_episode, running_length/log_counter, running_reward/log_counter, running_main_reward/log_counter))

            running_length = 0
            running_reward = 0
            running_main_reward = 0
            log_counter = 0

        episode_count.value = 0
        sample_count.value = 0

        # stop training if average main reward > solved_reward
        if np.mean(rewbuffer_env) > args.solved_reward:
            logging.info("########## Solved! ##########")
            break

    close_logger()
    writer.close()
    # Add a poison pill for each process
    for i in range(args.nb_procs):
        tasks.put(None)
    tasks.join()
