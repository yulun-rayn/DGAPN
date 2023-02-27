import os
import gym
import logging
import numpy as np
from rdkit import Chem
from collections import deque, OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter

from dgapn.model import DGAPN, save_DGAPN

from dgapn.reward import get_reward

from dgapn.utils.general_utils import close_logger, deque_to_csv
from dgapn.utils.graph_utils import mols_to_pyg_batch
from dgapn.utils.rl_utils import Memory, Log

#####################################################
#                   TRAINING LOOP                   #
#####################################################

def train_serial(args, env, model, writer=None, save_dir=None):
    sample_count = 0
    episode_count = 0
    save_counter = 0
    log_counter = 0

    running_length = 0
    running_reward = 0
    running_main_reward = 0

    memory = Memory()
    rewbuffer_env = deque(maxlen=100)
    molbuffer_env = deque(maxlen=10000)
    # training loop
    i_episode = 0
    while i_episode < args.max_episodes:
        logging.info("\n\nCollecting rollouts")
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
                    reward = main_reward
                    done = True
                if (args.iota > 0 and 
                    i_episode > args.innovation_reward_episode_delay and 
                    i_episode < args.innovation_reward_episode_cutoff):
                    inno_reward = model.get_inno_reward(mols_to_pyg_batch(state, model.emb_3d, device=model.device))
                    reward += args.iota * inno_reward
                running_reward += reward

                # Saving rewards and terminals:
                memory.rewards.append(reward)
                memory.terminals.append(done)

                if done:
                    break

            sample_count += t
            episode_count += 1

            running_length += t
            running_main_reward += main_reward

            rewbuffer_env.append(main_reward)
            molbuffer_env.append((Chem.MolToSmiles(state), main_reward))

            if writer is not None:
                # write to Tensorboard
                writer.add_scalar("EpMainRew", main_reward, i_episode-1)
                writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode-1)

        i_episode += episode_count
        save_counter += episode_count
        log_counter += episode_count

        # update model
        logging.info("\nUpdating model @ episode %d..." % i_episode)
        model.update(memory)
        memory.clear()

        deque_to_csv(molbuffer_env, os.path.join(save_dir, 'mol_dgapn.csv'), mode='a')
        molbuffer_env.clear()

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

        sample_count = 0
        episode_count = 0

        # stop training if average main reward > solved_reward
        if np.mean(rewbuffer_env) > args.solved_reward:
            logging.info("########## Solved! ##########")
            break

    close_logger()
    writer.close()
