#####################################################
#                   HELPER MODULES                  #
#####################################################

class Memory:
    def __init__(self):
        self.states = []        # state representations: pyg graph
        self.candidates = []    # next state (candidate) representations: pyg graph
        self.states_next = []   # next state (chosen) representations: pyg graph
        self.actions = []       # action index: long
        self.logprobs = []      # action log probabilities: float
        self.rewards = []       # rewards: float
        self.terminals = []     # trajectory status: logical

    def extend(self, memory):
        self.states.extend(memory.states)
        self.candidates.extend(memory.candidates)
        self.states_next.extend(memory.states_next)
        self.actions.extend(memory.actions)
        self.logprobs.extend(memory.logprobs)
        self.rewards.extend(memory.rewards)
        self.terminals.extend(memory.terminals)

    def clear(self):
        del self.states[:]
        del self.candidates[:]
        del self.states_next[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminals[:]


class Log:
    def __init__(self):
        self.ep_mols = []
        self.ep_lengths = []
        self.ep_rewards = []
        self.ep_main_rewards = []

    def extend(self, log):
        self.ep_mols.extend(log.ep_mols)
        self.ep_lengths.extend(log.ep_lengths)
        self.ep_rewards.extend(log.ep_rewards)
        self.ep_main_rewards.extend(log.ep_main_rewards)

    def clear(self):
        del self.ep_mols[:]
        del self.ep_lengths[:]
        del self.ep_rewards[:]
        del self.ep_main_rewards[:]


class Scheduler(object):
    def __init__(self, cutoff=100, coeff=0.01, weight_main=True):
        self.cutoff = cutoff
        self.coeff = coeff
        self.weight_main = weight_main

    def main_weight(self, episode):
        if self.weight_main and episode < self.cutoff:
            return episode/self.cutoff
        else:
            return 1.

    def guide_weight(self, episode):
        if episode < self.cutoff:
            return 2*self.coeff*(self.cutoff - episode)/self.cutoff
        else:
            return 0
