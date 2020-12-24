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
        self._max_episode_steps = 2500
        self._episode_steps = 0
        self.reset_steps = 500 # Reset the snake at the origin.

        # Parameters that given by me.
        self.n = 15 # Number of oscillator.
        self.stride = 0.01 # Time step of each loop.
        
        # These are learnable, but not now.
        self.order = 1 # Order of Fourier expension.
        self.nu = 1 # Nu = 2 / T.

        # These are generated through action.
        self.coefficient_mat = np.zeros((self.n, self.order + 1), dtype = np.float64) # Fourier coefficients.
        #self.weight_mat = np.diag(np.ones((self.n - 1, ), dtype = np.float64), -1) + np.diag(np.ones((self.n - 1, ), dtype = np.float64), 1) # Weight of CPG phase difference.
        self.weight_mat = np.diag(np.ones((self.n - 2, ), dtype = np.float64), -2) + np.diag(np.ones((self.n - 2, ), dtype = np.float64), -2) # Weight of CPG phase difference.
        self.phase_mat = np.zeros((self.n, self.n), dtype = np.float64) # Phase difference of each oscillator.
        
        # This is the result of CPG ode, which is automatically calculated.
        self.x = np.zeros((self.n, self.order + 1 + 1), dtype = np.float64)

        # State is the position of each joint.
        self.state = np.zeros((self.n), dtype = np.float64)
        
        # Action space: (n + n + n * n + n) array.
        # First n columns define the first column of Fourier coefficients(n-by-2).
        # Following n columns define the second column of Fourier coefficients.
        # Following n * n columns define the weight matrix(n-by-n).
        # Following n columns define the phase differences matrix(n-by-n).
        # action_low = np.hstack(
        #     (np.array([-np.pi / 4.0] * self.n * (self.order + 1)), # Something wrong here.
        #     np.array([0.0] * self.n * self.n),
        #     np.array([-np.pi] * self.n))
        # )
        # action_high = np.hstack(
        #     (np.array([np.pi / 4.0] * self.n * (self.order + 1)),
        #     np.array([1.0] * self.n * self.n),
        #     np.array([np.pi] * self.n))
        # )

        # 2020.12.10 I change the action into:
        # First n columns define the first column of Fourier coefficients(n-by-2).
        # Following n columns define the second column of Fourier coefficients.
        # Following 1 column define the phase differences matrix(n-by-n).
        # When self.order == 1, the action looks like this:
        # action_low = np.hstack(
        #     (np.array([-np.pi / 4.0] * self.n),
        #     np.array([0] * self.n),
        #     np.array([0]))
        # )
        # action_high = np.hstack(
        #     (np.array([np.pi / 4.0] * self.n),
        #     np.array([np.pi / 4.0] * self.n),
        #     np.array([2.0 * np.pi]))
        # )
        # self.action_space = spaces.Box(
        #     low = action_low,
        #     high = action_high,
        #     shape = ((self.order + 1) * self.n + 1, ),
        #     dtype = np.float64
        # )

        # 2020.12.20 I change the action again:
        # First n columns define the first column of Fourier coefficients(n-by-2).
        # Following n columns define the second column of Fourier coefficients.
        # Following 2 column define the phase difference of 0, 2, 4, ..., 14 and 1, 3, 5, ..., 13.
        # When self.order == 1, the action looks like this:
        action_low = np.hstack(
            (np.array([-np.pi / 4.0] * self.n),
            np.array([0.0] * self.n),
            np.array([0.0] * 2))
        )
        action_high = np.hstack(
            (np.array([np.pi / 4.0] * self.n),
            np.array([np.pi / 4.0] * self.n),
            np.array([2.0 * np.pi] * 2))
        )
        self.action_space = spaces.Box(
            low = action_low,
            high = action_high,
            shape = ((self.order + 1) * self.n + 2, ),
            dtype = np.float64
        )

        # The observation is just the state.
        state_low = np.array([-np.pi / 2.0, -np.pi] * 7 + [-np.pi / 2.0])
        state_high = np.array([np.pi / 2.0, np.pi] * 7 + [np.pi / 2.0])
        self.observation_space = spaces.Box(
            low = state_low,
            high = state_high,
            shape = (self.n, ),
            dtype = np.float64
        )

        self.seed()

        # Initialize pybullet environment.
        self.cid = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("/home/ellipse/anaconda3/lib/python3.8/site-packages/pybullet_data/plane.urdf")
        self.start_pos = [0, 0, 0.2]
        self.start_ori = p.getQuaternionFromEuler([0, np.pi / 2.0, 0])
        self.id = p.loadURDF("/home/ellipse/Desktop/pysnake_ws/src/snake_description/urdf/snake_description.urdf", self.start_pos, self.start_ori)

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
        old_pos, _ = p.getBasePositionAndOrientation(self.id)

        self.ProcessAction(action)
        for i in range(1): # Each step consists of one self.stride
        # for i in range(int(2 / (self.nu * self.stride))): # Each step consists of T.
            self.x = RungeKutta4(self.CPGode, self.stride, self.x)
            p.setJointMotorControlArray(self.id, range(self.n), p.POSITION_CONTROL, targetPositions = self.GetPos())
            p.stepSimulation()
            sleep(self.stride)
        new_pos, _ = p.getBasePositionAndOrientation(self.id)
        # print(new_pos)

        for i in range(self.n):
            self.state[i], _, _, _ = p.getJointState(self.id, i)
        # reward = self.GetReward(old_pos, new_pos)
        self._episode_steps += 1
        # print(f"Max epsisode steps: {self._max_episode_steps}. Current episode steps: {self._episode_steps}")
        distance = np.sqrt((new_pos[0]) ** 2 + (new_pos[1] - 1.0) ** 2)
        # print(distance)
        if self._episode_steps >= self._max_episode_steps:
            reward = -self._episode_steps
            done = True
            self._episode_steps = 0
        elif distance <= 0.4:
            reward = self._max_episode_steps - self._episode_steps
            done = True
            self._episode_steps = 0
        else:
            reward = 0
            done = False

        return self.state, reward, done, {}

    def reset(self):
        # action0 = np.zeros(((self.order + 1 + self.n + 1) * self.n), dtype = np.float64)
        action0 = np.zeros(( 2 * self.n + 2), dtype = np.float64)
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

    def GetReward(self, old_pos, new_pos):
        reward = (new_pos[1] - old_pos[1]) - abs(new_pos[0] - old_pos[0])

        return reward

    def ProcessAction(self, action):
        # Get coefficients from action.
        for i in range(self.order + 1):
            self.coefficient_mat[:, i] = action[i * self.n: (i + 1) * self.n]
        
        # Get weights from action.
        # self.weight_mat = action[(self.order + 1) * self.n: (self.order + 1 + self.n) * self.n].reshape(self.n, self.n)

        # Get phase differences from action.
        # p = action[(self.order + 1 + self.n) * self.n: (self.order + 1 + self.n + 1) * self.n]
        # p = np.zeros((self.n), dtype = np.float64)
        p = action[(self.order + 1) * self.n: (self.order + 1) * self.n + 2]
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
    action = np.hstack(
        (np.array([0] * env.n), # Deviation is 0.
        np.array([np.pi / 3.0] * env.n), # Amplitude is all pi / 3.
        np.array([np.pi / 4.0] * 2)) # Phase difference of 0, 2, 4, ... is pi/4, 1, 3, 5, ... is also pi/4.
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

# if __name__ == "__main__":
#     env = gym.make("gym_snake:snake-v0")
#     plt.figure()
#     pos_array = []
#     pos_array2 = []
#     time_array = []
#     for i in range(10000):
#         action = np.hstack(
#             (np.array([0] * env.n),
#             np.array([np.pi / 4.0, 0.0] * 7 + [np.pi / 4.0]),
#             np.array([np.pi / 4.0]))
#         )
#         env.step(action)

#         time = i * env.stride
#         time_array.append(time)
#         pos, _, _, _ = p.getJointState(env.id, 0)
#         pos2, _, _, _ = p.getJointState(env.id, 2)

#         pos_array.append(pos)
#         pos_array2.append(pos2)
#         plt.clf
#         plt.plot(time_array, pos_array, "b-", time_array, pos_array2, "r-")
#         plt.pause(0.001)

#         if i > 100:
#             time_array.pop(0)
#             pos_array.pop(0)
#             pos_array2.pop(0)
