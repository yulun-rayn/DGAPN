import os
import numpy as np

import time
from datetime import datetime

from rdkit import Chem

from reward.get_reward import get_reward

from environment.env import CReM_Env

def greedy_rollout(save_path,
                    reward_type,
                    K,
                    max_rollout=6,
                    args=None):
    env = CReM_Env(args.data_path,
                args.warm_start_dataset,
                nb_sample_crem=args.nb_sample_crem,
                max_timesteps=max_rollout,
                mode='mol')

    mol, mol_candidates, done = env.reset()
    mol_best = mol

    new_rew = get_reward(mol, reward_type, args=args)
    start_rew = new_rew
    best_rew = new_rew
    steps_remaining = K
    print(" Initial Reward: {:4.2f}".format(start_rew))

    for t in range(1, max_rollout+1):
        steps_remaining -= 1
        next_rewards = get_reward(mol_candidates, reward_type, args=args)

        action = np.argmax(next_rewards)

        try:
            new_rew = next_rewards[action]
        except Exception as e:
            print(e)
            break

        mol, mol_candidates, done = env.step(action, include_current_state=False)

        if new_rew > best_rew:
            mol_best = mol
            best_rew = new_rew
            steps_remaining = K

        print("  Step {:3d}: Best Reward: {:4.2f} Bad Steps: {:2d}".format(t, best_rew, K-steps_remaining))
        if (steps_remaining == 0) or done:
            break

    smile_best = Chem.MolToSmiles(mol_best, isomericSmiles=False)
    with open(save_path, 'a') as f:
        row = ''.join(['{},'] * 2)[:-1] + '\n'
        f.write(row.format(smile_best, best_rew))

    return start_rew, best_rew

def eval_greedy(artifact_path, reward_type, N=30, K=1, args=None):
    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    save_path = os.path.join(artifact_path, (args.name if args else '') + '_' + dt + '_greedy.csv')

    print("\nStarting greedy...\n")
    avg_improvement = []
    avg_best = []
    for i in range(1, N+1):
        start_rew, best_rew = greedy_rollout(save_path,
                                             reward_type,
                                             K,
                                             args=args)
        improvement = best_rew - start_rew
        print("Episode {:2d}: Initial Reward: {:4.2f} Best Reward: {:4.2f} Improvement: {:4.2f}\n".format(
                                                      i,
                                                      start_rew,
                                                      best_rew,
                                                      improvement))
        avg_improvement.append(improvement)
        avg_best.append(best_rew)
    avg_improvement = sum(avg_improvement) / len(avg_improvement)
    avg_best = sum(avg_best) / len(avg_best)
    print("Avg improvement over {} samples: {:5.2f}".format(N, avg_improvement))
    print("Avg best        over {} samples: {:5.2f}".format(N, avg_best))
