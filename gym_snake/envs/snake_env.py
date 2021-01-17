import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pybullet as p
from time import sleep
import matplotlib.pyplot as plt

def RungeKutta4(ode, step, x):
    k1 = step * ode(x)
    k2 = step * ode(x + k1 / 2)
    k3 = step * ode(x + k2 / 2)
    k4 = step * ode(x + k3)
    y = x + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return y

class SnakeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        # Each episode consists of self._max_episode_steps steps.
        self._max_episode_steps = 20
        self._episode_steps = 0
        self.reset_steps = 500 # Reset the snake at the origin.

        # Parameters that given by me.
        self.n = 15 # Number of oscillator.
        self.stride = 0.01 # Time step of each loop. Unit: second.
        
        # These are learnable, but not now.
        self.order = 1 # Order of Fourier expension.
        self.nu = 1 # Nu = 2 / T. Here T is the period of servo motion.

        # These are generated through action.
        self.coefficient_mat = np.zeros((self.n, self.order + 1), dtype = np.float64) # Fourier coefficients.
        #self.weight_mat = np.diag(np.ones((self.n - 1, ), dtype = np.float64), -1) + np.diag(np.ones((self.n - 1, ), dtype = np.float64), 1) # Weight of CPG phase difference.
        self.weight_mat = np.diag(np.ones((self.n - 2, ), dtype = np.float64), -2) + np.diag(np.ones((self.n - 2, ), dtype = np.float64), -2) # Weight of CPG phase difference.
        self.phase_mat = np.zeros((self.n, self.n), dtype = np.float64) # Phase difference of each oscillator.
        
        # This is the result of CPG ode, which is automatically calculated.
        self.x = np.zeros((self.n, self.order + 1 + 1), dtype = np.float64)

        # State is the position of each joint, as well as the x-y coordinate.
        self.state = np.zeros((self.n + 2), dtype = np.float64)

        # 2021.01.04 update.
        # Plan A:
        # Action is the second Fourier coefficient of each joint and two phase differences.
        # action_low = np.hstack(
        #     (np.array([0.0] * self.n),
        #     np.array([0.0] * 2))
        # )
        # action_high = np.hstack(
        #     (np.array([np.pi / 3.0] * self.n),
        #     np.array([2.0 * np.pi] * 2))
        # )
        # self.action_space = spaces.Box(
        #     low = action_low,
        #     high = action_high,
        #     shape = (self.n + 2, ),
        #     dtype = np.float64
        # )

        # Plan B:
        # Action is two coefficients and two phase differences.
        action_low = np.hstack(
            (np.array([0.0] * 2),
            np.array([0.0] * 2))
        )
        action_high = np.hstack(
            (np.array([np.pi / 4.0] * 2),
            np.array([2.0 * np.pi] * 2))
        )
        self.action_space = spaces.Box(
            low = action_low,
            high = action_high,
            shape = (2 + 2, ),
            dtype = np.float64
        )

        # The observation is just the state.
        state_low = np.array([-np.pi / 4.0] * self.n + [-2.0] + [-2.0])
        state_high = np.array([np.pi / 4.0] * self.n + [2.0] + [2.0])
        self.observation_space = spaces.Box(
            low = state_low,
            high = state_high,
            shape = (self.n + 2, ),
            dtype = np.float64
        )

        self.seed()

        # Initialize pybullet environment.
        self.cid = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("./plane.urdf")
        self.start_pos = [0, 0, 0.2]
        self.start_ori = p.getQuaternionFromEuler([0, np.pi / 2.0, 0])
        self.id = p.loadURDF("./snake_description.urdf", self.start_pos, self.start_ori)

        # Set frictions
        lateral_friction = 0.8
        for i in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, i, lateralFriction = lateral_friction)
        p.changeDynamics(self.id, -1, lateralFriction = lateral_friction)

        for i in range(self.reset_steps):
            p.stepSimulation()

        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self._episode_steps += 1
        old_pos, _ = p.getBasePositionAndOrientation(self.id)
        self.ProcessAction(action)
        # for i in range(1): # Each step consists of one self.stride of time.
        for i in range(int(2 / (self.nu * self.stride))): # Each step consists of T.
            self.x = RungeKutta4(self.CPGode, self.stride, self.x)
            p.setJointMotorControlArray(self.id, range(self.n), p.POSITION_CONTROL, targetPositions = self.GetPos())
            p.stepSimulation()
            sleep(self.stride)
        new_pos, _ = p.getBasePositionAndOrientation(self.id)

        # Get state.
        for i in range(self.n):
            self.state[i], _, _, _ = p.getJointState(self.id, i)
        self.state[self.n] = new_pos[0] # x coordinate.
        self.state[self.n + 1] = new_pos[1] # y coordinate.

        # Calculate the step reward.
        reward = (new_pos[0] - old_pos[0]) - abs(new_pos[1] - old_pos[1])
        if self._episode_steps >= self._max_episode_steps or self.state[self.n + 1] > 1.0 or self.state[self.n + 1] < -1.0 or self.state[self.n] < -1.0:
            reward -= 1.0
            done = True
            self._episode_steps = 0
        elif self.state[self.n] > 1.0:
            reward += 1.0
            done = True
            self._episode_steps = 0
        else:
            done = False

        return self.state, reward, done, {}

    def reset(self):
        # Plan A:
        # action0 = np.zeros((self.n + 2), dtype = np.float64)

        # Plan B:
        action0 = np.zeros((2 + 2), dtype = np.float64)

        self.ProcessAction(action0)
        self.x = np.zeros((self.n, self.order + 1 + 1), dtype = np.float64)

        p.setJointMotorControlArray(self.id, range(self.n), p.POSITION_CONTROL, targetPositions = self.GetPos())
        for i in range(self.reset_steps):
            p.stepSimulation()

        p.resetBasePositionAndOrientation(self.id, self.start_pos, self.start_ori)
        for i in range(self.reset_steps):
            p.stepSimulation()
        sleep(0.5)

        for i in range(self.n):
            self.state[i], _, _, _ = p.getJointState(self.id, i)

        return self.state

    def render(self, mode='human'):
    
        return

    def close(self):
        
        return

    def ProcessAction(self, action):
        # Plan A:
        # Get coefficients from action.
        # self.coefficient_mat[:, self.order] = action[0: self.n]
        # 
        # Get phase differences from action.
        # p = action[self.n: self.n + 2]
        # for i in [2, 4, 6, 8, 10, 12, 14]:
        #     self.phase_mat[i-2][i] = p[0]
        #     self.phase_mat[i][i-2] = -p[0]
        # for i in [3, 5, 7, 9, 11, 13]:
        #     self.phase_mat[i-2][i] = p[1]
        #     self.phase_mat[i][i-2] = -p[1]

        # Plan B:
        # Get coefficients from action.
        c = action[0: 2]
        for i in [0, 2, 4, 6, 8, 10, 12, 14]:
            self.coefficient_mat[i][self.order] = c[0]
        for i in [1, 3, 5, 7, 9, 11, 13]:
            self.coefficient_mat[i][self.order] = c[1]
        
        # Get phase differences from action.
        p = action[2: 2 + 2]
        for i in [2, 4, 6, 8, 10, 12, 14]:
            self.phase_mat[i-2][i] = p[0]
            self.phase_mat[i][i-2] = -p[0]
        for i in [3, 5, 7, 9, 11, 13]:
            self.phase_mat[i-2][i] = p[1]
            self.phase_mat[i][i-2] = -p[1]
        

    # Calculate all the positions of oscillators.
    def GetPos(self):
        y = np.zeros((self.n), dtype = np.float64)

        for i in range(self.n):
            for j in range(self.order + 1):
                if j == 0:
                    y[i] += self.x[i][j]
                else:
                    y[i] += self.x[i][j] * np.sin(j * self.x[i][self.order + 1])

        return y

    # The structure of self.x:
    # /fourier coefficients               /phase
    # 0,    1,    2,  ...  ,    order,    theta
    # This is a n-by-(order + 1 + 1) matrix
    def CPGode(self, x):
        y = np.zeros(x.shape, dtype = np.float64)

        for i in range(self.order + 1):
            y[:, i] = 2.0 * (self.coefficient_mat[:, i] - x[:, i])

        for i in range(self.n):
            delta = 0.0
            for j in range(self.n):
                delta += self.weight_mat[i][j] * np.arctan(x[j][self.order + 1] - x[i][self.order + 1] - self.phase_mat[i][j])
            y[i][self.order + 1] = delta + np.pi * self.nu

        return y

if __name__ == "__main__":
    env = gym.make("gym_snake:snake-v0")

    # PLan A:
    # action = np.hstack(
    #     (np.array([np.pi / 3.0] * env.n), # Amplitude is all pi / 3.
    #     np.array([np.pi / 4.0, np.pi / 2.0])) # Phase difference of 0, 2, 4, ... is pi/4, 1, 3, 5, ... is also pi/4.
    # )

    # Plan B:
    action = np.hstack(
            (np.array([np.pi / 3.0, 0.0]),
            np.array([np.pi / 4.0] * 2))
    )

    env.ProcessAction(action)
    y0 = [0.0]
    y1 = [0.0]
    y2 = [0.0]
    y3 = [0.0]
    y4 = [0.0]
    y5 = [0.0]
    time = [0.0]
    while time[-1] < 20:
        env.x = RungeKutta4(env.CPGode, env.stride, env.x)
        y0.append(env.GetPos()[0])
        y1.append(env.GetPos()[1])
        y2.append(env.GetPos()[2])
        y3.append(env.GetPos()[3])
        y4.append(env.GetPos()[4])
        y5.append(env.GetPos()[5])
        time.append(time[-1] + env.stride)
    plt.figure()
    plt.plot(time, y0, "r-", time, y2, "g-", time, y4, "b-")
    plt.figure()
    plt.plot(time, y1, "r-", time, y3, "g-", time, y5, "b-")
    plt.show()
