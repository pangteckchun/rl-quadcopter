import numpy as np
import random
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pose=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pose: target/goal (x,y,z & eular angles) pose for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6 # six dimensional pose
        self.action_low = 10 # 0 was original value
        self.action_high = 900
        self.action_size = 4

        # Default goal if not set
        self.target_pose = target_pose if target_pose is not None else np.array([0., 0., 10., 0., 0., 0.]) 

        print ('Initial vel = ', self.sim.init_velocities)
        print ('Initial angle vel = ', self.sim.init_angle_velocities)
        print ('Initial pose = ', self.sim.init_pose)
        print ('Target pose = ', self.target_pose)


    """reward is 10 for matching target pose, -ve as you go farther"""
    def get_reward(self):
        pose_diff = np.around((self.sim.pose - self.target_pose), decimals=0)
      
        x_pos = np.around(self.sim.pose[0], decimals=0)
        y_pos = np.around(self.sim.pose[1], decimals=0)
        z_pos = np.around(self.sim.pose[2], decimals=0)

        target_x_pos = np.around(self.target_pose[0], decimals=0)
        target_y_pos = np.around(self.target_pose[1], decimals=0)
        target_z_pos = np.around(self.target_pose[2], decimals=0)

        reward = 0
        reached_target = False

        # penalize zero vertical velocity
        if self.sim.v[2] <= 0:
            reward -= 10
        
        # reward or penalize pose during simulation
        xyz_diff = np.around(np.square(np.tanh(pose_diff [:3])), decimals=1)
        for each_pos_delta in xyz_diff: 
            if each_pos_delta > 0:
                reward -= 10
            elif each_pos_delta == 0:
                reward += 10
                reached_target = True
        
        # reward when target pose is reached for each of its X, Y & Z positions
        if x_pos == target_x_pos:
            reward += 10
            reached_target = True
        if y_pos == target_y_pos:
            reward += 10
            reached_target = True
        if z_pos == target_z_pos:
            reward += 10
            reached_target = True
        
        return reward, reached_target


    """Uses action to obtain next state, reward, done"""
    def step(self, rotor_speeds): # - rotor_speeds is the expected input
        reward = 0

        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            next_state = self.sim.pose

            reward, reached_target = self.get_reward()
            if reached_target:
                done = True
                break

        return np.around(next_state, decimals=0), reward, done


    """Reset the sim to start a new episode."""
    def reset(self):
        self.sim.reset()
        state = self.sim.pose
        return np.around(state, decimals=0)


    def epsilon_greedy(self, Q, state, nA, eps):
        """Selects epsilon-greedy action for supplied state.
        
        Params
        ======
            Q (dictionary): action-value function
            state (int): current state
            nA (int): number actions in the environment
            eps (float): epsilon
        """

        # initial rotor_speeds could be zero, hence select a random action to begin with
        if Q[state].sum() == 0:
            #return np.random.randint((self.action_low+15), self.action_high, size=nA)
            return np.random.randint((self.action_low), self.action_high, size=nA)
        
        if random.random() > eps: # select greedy action with probability epsilon
            return Q[state]
        else:                     # otherwise, select an action randomly
            #return np.random.randint((self.action_low+15), self.action_high, size=nA)
            return np.random.randint((self.action_low), self.action_high, size=nA)


    """ Referencing work done by Udacity-Alexis Cook
    https://github.com/udacity/deep-reinforcement-learning/blob/master/temporal-difference/Temporal_Difference_Solution.ipynb
    """
    def update_Q_expsarsa(self, alpha, gamma, nA, eps, Q, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""
        current = Q[state]    # estimate in Q-table (for current state, action pair)
        policy_s = np.ones(nA) * eps / nA  # current policy (for next state S')
        policy_s[np.argmax(Q[next_state])] = 1 - eps + (eps / nA) # greedy action
        Qsa_next = np.dot(Q[next_state], policy_s)         # get value of state at next time step
        target = reward + (gamma * Qsa_next)               # construct target
        new_value = current + (alpha * (target - current)) # get updated value 
        return new_value
