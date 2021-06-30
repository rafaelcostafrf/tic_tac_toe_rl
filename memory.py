import numpy as np

class memory:
    def __init__(self):
        self.clear_memory()

    def clear_memory(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def append_memory_rt(self, reward, terminal):
        self.rewards.append(reward)
        self.is_terminals.append(terminal)


    def append_memory_as(self, actions, states, logprobs, state_value):
        self.actions.append(actions[0])
        self.states.append(states[0])
        self.logprobs.append(logprobs[0])
        self.values.append(state_value[0, 0])

    def close_memory(self):
        self.actions = np.asarray(self.actions)
        self.states = np.asarray(self.states)
        self.logprobs = np.asarray(self.logprobs)
        self.rewards = np.asarray(self.rewards)
        self.is_terminals = np.asarray(self.is_terminals)
        self.values = np.asarray(self.values)