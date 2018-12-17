import numpy as np
import sys
import operator
from task import Task
from collections import defaultdict 

class Quadcopter_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        # Score tracker and learning parameters
        self.score = 0        

        # Episode variables
        self.reset_episode()

        self.Q = defaultdict(lambda: np.zeros(self.task.action_size)) # initialise the Q value


    def reset_episode(self):
        self.score = 0
        state = self.task.reset()
        return np.around(state, decimals=0)

    
    def step (self, alpha, gamma, action, reward, state, next_state):
        self.Q[tuple(state)] = self.task.update_Q_sarsamax(alpha, gamma, self.Q, \
                                             tuple(state), action, reward, tuple(next_state))
        state = next_state      # S <- S'
        self.score += reward    # add reward to agent's score
        return state, self.score

    
    # Choose action by following the policy
    def act(self, state, eps):
         # epsilon-greedy action selection
        action = self.task.epsilon_greedy(self.Q, tuple(state), self.task.action_size, eps)

        return action