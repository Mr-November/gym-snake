import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pybullet as p
from time import sleep

import oscillator as osc

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Time step of each loop
        self.step = 0.01
        # Number of oscillator.
        self.n = 15
        # Order of fourier expansion.
        self.order = 2
        # Equals 2 / T.
        self.nu = np.ones((self.n, 1), dtype = np.float64)
        
        # Fourier coefficients.
        self.coefficient_mat = np.zeros((self.n, self.order + 1), dtype = np.float64)
        
        # Weight of CPG phase difference
        self.weight_mat = np.zeros((self.n, self.n), dtype = np.float64)

        # Phase difference of each oscillator
        self.phase_mat = osc.GetPhaseMatrix(np.zeros((self.n, 1), dtype = np.float64))
        
        # Action: [coefficient_mat, weight_mat, phase_mat]
        self.action_min = np.hstack((osc.c_min, osc.w_min, osc.p_min))
        self.action_max = np.hstack((osc.c_max, osc.w_max, osc.p_max))
        self.action_space = spaces.Box(
            low = self.action_min,
            high = self.action_max,
            shape = (self.n, 2 * self.n + 1),
            dtype = np.float64
        )

        self.state = np.zeros((self.n, self.order + 2), dtype = np.float64)
        
        # State: []
        self.observation_space = None

        physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        plane_id = p.loadURDF("/home/ellipse/anaconda3/lib/python3.8/site-packages/pybullet_data/plane.urdf")
        self.start_pos = [0, 0, 0]
        self.start_ori = p.getQuaternionFromEuler([0, np.pi / 2.0, 0])
        self.id = p.loadURDF("/home/ellipse/Desktop/pysnake_ws/src/snake_description/urdf/snake_description.urdf", self.start_pos, self.start_ori)
        lf = 0.01
        sf = 0.0
        rf = 0.0
        p.changeDynamics(self.id, -1, lateralFriction = lf, spinningFriction = sf, rollingFriction = rf)
        for i in range(self.n):
            p.changeDynamics(self.id, i, lateralFriction = lf, spinningFriction = sf, rollingFriction = rf)
        
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.coefficient_mat = action[:, 0:self.n]
        self.weight_mat = action[:, self.n:2*self.n]
        self.phase_mat = osc.GetPhaseMatrix(action[:, 2*self.n:(2*self.n+1)])
        
        old_pos, old_vel, joint_ref, joint_mot = p.getJointStates(self.id, range(self.n))
        self.state = osc.RungeKutta4(self.CPGode, self.step, self.state)
        p.setJointMotorControlArray(self.id, range(self.n), p.POSITION_CONTROL, targetPositions = osc.GetPos(self.n, self.order, self.state)[:, 0])
        new_pos, new_vel, joint_ref, joint_mot = p.getJointStates(self.id, range(self.n))
        reward = self.GetReward(old_pos, new_pos, old_vel, new_vel)

        sleep(self.step)
        p.stepSimulation()

        return self.state, reward, False, {}

    def reset(self):
        self.state = np.zeros((self.n, self.order + 2), dtype = np.float64)

        return self.state

    def render(self, mode='human'):
        return

    def close(self):
        return

    def GetReward(self, op, np, ov, nv):
        r = (np - op) + (nv - ov)

        return r

    # The structure of x:
    # /fourier coefficients               /phase
    # 0,    1,    2,  ...  ,    order,    theta
    def CPGode(self, x):

        y = np.zeros(x.shape)

        for i in range(self.order + 1):
            y[:, i] = 1.0 * (self.coefficient_mat[:, i] - x[:, i])

        for i in range(self.n):
            delta = 0.0
            for j in range(self.n):
                delta += self.weight_mat[i, j] * np.arctan(x[i, self.order + 1] - x[j, self.order + 1] - self.phase_mat[i, j])
            y[i, self.order + 1] = delta + np.pi * self.nu[i]

        return y