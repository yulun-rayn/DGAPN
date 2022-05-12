import os
import gym
import logging
import numpy as np
from rdkit import Chem
from collections import deque, OrderedDict

import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from dgapn.DGAPN import DGAPN, save_DGAPN

from reward.get_reward import get_reward

from utils.general_utils import initialize_logger, close_logger, deque_to_csv
from utils.graph_utils import mols_to_pyg_batch
from utils.rl_utils import Memory, Log, Scheduler

#####################################################
#                   TRAINING LOOP                   #
#####################################################

def train_serial(args, env, model):
    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + '_' + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)
    initialize_logger(save_dir)
    logging.info(model)

    sample_count = 0
    update_count = 0

    running_length = 0
    running_reward = 0
    running_main_reward = 0

    memory = Memory()
    rewbuffer_env = deque(maxlen=100)
    molbuffer_env = deque(maxlen=10000)

    scheduler = Scheduler(args.innovation_reward_update_cutoff, args.iota, weight_main=False)

    # training loop
    i_episode = 0
    while i_episode < args.max_episodes:
        logging.info("\n\ncollecting rollouts")
        while sample_count < args.update_timesteps:
            state, candidates, done = env.reset()

            for t in range(1, args.max_timesteps+1):
                # Running policy:
                state_emb, candidates_emb, action_logprob, action = model.select_action(
                    mols_to_pyg_batch(state, model.emb_3d, device=model.device),
                    mols_to_pyg_batch(candidates, model.emb_3d, device=model.device))
                memory.states.append(state_emb[0])
                memory.candidates.append(candidates_emb)
                memory.states_next.append(candidates_emb[action])
                memory.actions.append(action)
                memory.logprobs.append(action_logprob)

                state, candidates, done = env.step(action)

                reward = 0
                if (t==args.max_timesteps) or done:
                    main_reward = get_reward(state, reward_type=args.reward_type, args=args)
                    reward = scheduler.main_weight(update_count) * main_reward
                    done = True
                if (args.iota > 0 and update_count < args.innovation_reward_update_cutoff):
                    inno_reward = model.get_inno_reward(mols_to_pyg_batch(state, model.emb_3d, device=model.device))
                    reward += scheduler.guide_weight(update_count) * inno_reward
                running_reward += reward

                # Saving rewards and terminals:
                memory.rewards.append(reward)
                memory.terminals.append(done)

                if done:
                    break

            sample_count += t
            i_episode += 1

            running_length += t
            running_main_reward += main_reward

            rewbuffer_env.append(main_reward)
            molbuffer_env.append((Chem.MolToSmiles(state), main_reward))

            # write to Tensorboard
            writer.add_scalar("EpMainRew", main_reward, i_episode-1)
            writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode-1)

        # update model
        logging.info("\nupdating model @ episode %d..." % i_episode)
        model.update(memory)
        memory.clear()
        update_count += 1

        # stop training if avg_reward > solved_reward
        if np.mean(rewbuffer_env) > args.solved_reward:
            logging.info("########## Solved! ##########")
            save_DGAPN(model, os.path.join(save_dir, 'DGAPN_continuous_solved_{}.pt'.format('test')))
            break

        # save every save_interval episodes
        if i_episode % args.save_interval == 0:
            save_DGAPN(model, os.path.join(save_dir, '{:05d}_dgapn.pt'.format(i_episode)))
            deque_to_csv(molbuffer_env, os.path.join(save_dir, 'mol_dgapn.csv'))

        # save running model
        save_DGAPN(model, os.path.join(save_dir, 'running_dgapn.pt'))

        # logging
        if i_episode % args.log_interval == 0:
            logging.info('Episode {} \t Avg length: {} \t Avg reward: {:5.3f} \t Avg main reward: {:5.3f}'.format(
                i_episode, running_length/args.log_interval, running_reward/args.log_interval, running_main_reward/args.log_interval))

            running_length = 0
            running_reward = 0
            running_main_reward = 0

        sample_count = 0

    close_logger()
    writer.close()
