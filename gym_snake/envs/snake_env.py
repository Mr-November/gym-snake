import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pybullet as p
from time import sleep

class SnakeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        # Parameters that given by me.
        self.n = 15 # Number of oscillator.
        self.step = 0.01 # Time step of each loop.
        
        # These are learnable, but not now.
        self.order = 1 # Order of Fourier expension.
        self.nu = 1 # Nu = 2 / T.

        # These are generated through action.
        self.coefficient_mat = np.zeros((self.n, self.order + 1), dtype = np.float64) # Fourier coefficients.
        self.weight_mat = np.zeros((self.n, self.n), dtype = np.float64) # Weight of CPG phase difference.
        self.phase_mat = np.zeros((self.n, self.n), dtype = np.float64) # Phase difference of each oscillator.
        
        # This is the result of CPG ode, which is automatically calculated.
        self.x = np.zeros((self.n, self.order + 1 + 1), dtype = np.float64)
        
        # Action space: (n + n + n^2 + n) array.
        # First n columns define the first column of Fourier coefficients(n-by-2).
        # Following n columns define the second column of Fourier coefficients.
        # Following n * n columns define the weight matrix(n-by-n).
        # Following n columns define the phase differences matrix(n-by-n).
        # I think state equals action is ok.
        action_low = np.hstack(
            np.array([-np.pi / 4.0] * self.n * (self.order + 1)),
            np.array([0.0] * self.n * self.n),
            np.array([-np.pi] * self.n)
        )
        action_high = np.hstack(
            np.array([np.pi / 4.0] * self.n * (self.order + 1)),
            np.array([1.0] * self.n * self.n),
            np.array([np.pi] * self.n)
        )
        self.action_space = spaces.Box(
            low = action_low,
            high = action_high,
            shape = ((self.order + 1 + self.n + 1) * self.n, ),
            dtype = np.float64
        )
        self.observation_space = spaces.Box(
            low = action_low,
            high = action_high,
            shape = ((self.order + 1 + self.n + 1) * self.n, ),
            dtype = np.float64
        )

        self.seed()

        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.ProcessAction(action)
        self.x = RungeKutta4(self.CPGode, self.step, self.x)
        self.state = action
        reward = self.GetReward()

        return self.state, reward, False, {}

    def reset(self):
        action0 = np.zeros(((self.order + 1 + self.n + 1) * self.n), dtype = np.float64)
        self.ProcessAction(action0)
        self.x = np.zeros((self.n, self.order + 1 + 1), dtype = np.float64)
        self.state = action0

        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def GetReward(self):
        
        return 0

    def ProcessAction(self, action):
        # Get coefficients from action.
        for i in range(self.order + 1):
            self.coefficient_mat[:, i] = action[i * self.n: (i + 1) * self.n]
        
        # Get weights from action.
        self.weight_mat = action[(self.order + 1) * self.n: (self.order + 1 + self.n) * self.n]

        # Get phase differences from action.
        p = action[(self.order + 1 + self.n) * self.n: (self.order + 1 + self.n + 1) * self.n]
        for i in range(self.n):
            for j in range(self.n):
                self.phase_mat[i, j] = p[i] - p[j]

    # Calculate all the positions of oscillators.
    def GetPos(self):
        y = np.zeros((self.n), dtype = np.float64)

        for i in range(self.n):
            for j in range(self.order + 1):
                if j == 0:
                    y[i] += self.x[i, j]
                else:
                    y[i] += self.x[i, j] * np.sin(j * self.x[i, self.order + 1])

        return y

    # The structure of self.x:
    # /fourier coefficients               /phase
    # 0,    1,    2,  ...  ,    order,    theta
    # This is a n-by-(order + 1 + 1) matrix
    def CPGode(self):

        y = np.zeros(self.x.shape, dtype = np.float64)

        for i in range(self.order + 1):
            y[:, i] = 1.0 * (self.coefficient_mat[:, i] - self.x[:, i])

        for i in range(self.n):
            delta = 0.0
            for j in range(self.n):
                delta += self.weight_mat[i, j] * np.arctan(self.x[i, self.order + 1] - self.x[j, self.order + 1] - self.phase_mat[i, j])
            y[i, self.order + 1] = delta + np.pi * self.nu

        return y

def RungeKutta4(ode, step, x):

    k1 = step * ode(x)
    k2 = step * ode(x + k1 / 2)
    k3 = step * ode(x + k2 / 2)
    k4 = step * ode(x + k3)
    y = x + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return y