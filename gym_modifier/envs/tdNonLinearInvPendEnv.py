import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import matplotlib.pyplot as plt
import time, random
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


class NonLinInvPendulumEnv(gym.Env):

    """
    State-space:
    x = [theta; alpha; dtheta; dalpha] - Radians
    y = [theta, alpha] - Radians
    u = Vm - Voltage
    
    """

    def __init__(self):

        """
        TODO: Initiate the environment here
        - Define the state space, action space and restrictions in both spaces.
        The restrictions on the state space determine the terminal state

        - Also define all static variables here. E.g. gravity etc.
        """

        self.stepTimeStart = time.time()


        """
        
        Enable/Disable for plotting/rendering
        
        """
        self.do_render = False
        self.do_plotting = False

        # Visualization initialization
        if self.do_render:
            self.tdrend1 = TdRenderer()
            self.tdrend2 = TdRenderer()
            self.tdrend2.move_window()


        # Plotting initialization
        if self.do_plotting:
            self.app = QtGui.QApplication([])

            # Set graphical window, its title and size
            self.win = pg.GraphicsWindow(title="Sample process")
            self.win.resize(2600, 1200)
            self.win.setWindowTitle('pyqtgraph example')

            # Enable antialiasing for prettier plots
            pg.setConfigOptions(antialias=True)

            self.p1 = self.win.addPlot(row=1, col=1, title="Theta 1")
            self.p2 = self.win.addPlot(row=1, col=2, title="Theta 2")
            self.p3 = self.win.addPlot(row=1, col=3, title="Alpha 1")
            self.p4 = self.win.addPlot(row=1, col=4, title="Alpha 2")
            self.p5 = self.win.addPlot(row=2, col=1, title="DTheta 1")
            self.p6 = self.win.addPlot(row=2, col=2, title="DTheta 2")
            self.p7 = self.win.addPlot(row=2, col=3, title="DAlpha 1")
            self.p8 = self.win.addPlot(row=2, col=4, title="DAlpha 2")
            self.theta1plot = self.p1.plot()
            self.theta2plot = self.p2.plot()
            self.alpha1plot = self.p3.plot()
            self.alpha2plot = self.p4.plot()
            self.dtheta1plot = self.p5.plot()
            self.dtheta2plot = self.p6.plot()
            self.dalpha1plot = self.p7.plot()
            self.dalpha2plot = self.p8.plot()



        self.state_seq = np.zeros((1, 8))
        self.dt = 0.02

        # Define restrictions:
        self.action_clamp_high = np.array([10.0, 10.0])
        self.action_clamp_low = -self.action_clamp_high
        self.state_clamp_high = np.array(
            [20.0 * np.pi / 180.0, 20.0 * np.pi / 180.0, 100.0 * np.pi / 180.0, 100.0 * np.pi / 180.0,
             20.0 * np.pi / 180.0, 20.0 * np.pi / 180.0, 100.0 * np.pi / 180.0, 100.0 * np.pi / 180.0])
        self.state_clamp_low = - self.state_clamp_high

        self.action_space = spaces.Box(low=self.action_clamp_low, high=self.action_clamp_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.state_clamp_low * 1.1, high=self.state_clamp_high * 1.1,
                                            dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def dd_theta_alpha(self, alpha, dtheta, dalpha, Vm):
        Mp = 0.127
        Lr = 0.127
        Lp = 0.3111
        Jr = 0.0083
        Jp = 0.0012
        Dr = 0.069
        Co = 0.1285
        g = 0.981
        ddth = -1/2*((4.0*Mp*np.power(Lp, 2)*alpha*dtheta*dalpha - 8.0*Co*Vm+8.0*Dr*dtheta)*Jp+np.power(Mp, 2)\
                     *np.power(Lp, 4)*alpha*dtheta*dalpha)/((4.0*Jr+4.0*Mp*np.power(Lr, 2))*Jp + Mp*np.power(Lp, 2)\
                    *Jr)-(-1.0/2.0*((np.power(Mp, 2)*np.power(Lp, 3)*Lr*np.power(dtheta, 2) + 20.0*np.power(Mp, 2)\
                     *np.power(Lp, 2)*Lr*g)*alpha - 2.0*Mp*np.power(Lp, 2)*Dr*dtheta + 2.0*Mp*np.power(Lp, 2)*Co*Vm)/\
                          ((4.0*Jr+4.0*Mp*np.power(Lr, 2))*Jp + Mp*np.power(Lp, 2)*Jr))

        ddal = ((np.power(Mp, 2)*np.power(Lp, 2)*np.power(Lr, 2) + Mp*np.power(Lp, 2)*Jr)*np.power(dtheta, 2)\
                + 20.0*Jr*Mp*Lp*g+20.0*np.power(Mp, 2)*np.power(Lr, 2)*Lp*g)*alpha/((4.0*Jr+4.0*Mp*np.power(Lr, 2))*Jp\
                + Mp*np.power(Lp, 2)*Jr) + (2.0*Mp*Lr*Lp*Co*Vm - 2.0*Mp*Lr*Lp*Dr*dtheta - np.power(Mp, 2)\
                *np.power(Lp, 3)*Lr*alpha*dtheta*dalpha)/((4.0*Jr+4.0*Mp*np.power(Lr, 2))*Jp + Mp*np.power(Lp, 2)*Jr)
        return np.array([ddth, ddal])

    def step(self, u):

        """
        Performs one state-transition based on the input
        action and returns the reward of the new state.

        :param u: An action [ndarray]
        :return:
        - new_state:
        - reward: The reward of the new state [float]
        - terminal: Whether or not the new state is a terminal state [bool]
        - {}: ...
        """
        self.stepTimeStart = time.time()

        # Example save temp instance of last state

        last_state = self.state

        # Example of how to clip actions according to defined restrictions
        u = np.clip(u, self.action_clamp_low, self.action_clamp_high)[0:2]

        """
        TODO: Perform state transition to the next state here:
        E.g.:
        new_state = transition_matrix_A * last_state + transition_matrix_B * u  (Ax + Bu)
        """
        ddst = np.concatenate((self.dd_theta_alpha(last_state[1], last_state[2], last_state[3], u[0]),
                    self.dd_theta_alpha(last_state[5], last_state[6], last_state[7], u[1])))
        dstate = np.concatenate((last_state[2:4], ddst[0:2], last_state[6:8], ddst[2:4]))
        last_state = last_state + self.dt*dstate/2.0

        ddst = np.concatenate((self.dd_theta_alpha(last_state[1], last_state[2], last_state[3], u[0]),
                    self.dd_theta_alpha(last_state[5], last_state[6], last_state[7], u[1])))
        dstate = np.concatenate((last_state[2:4], ddst[0:2], last_state[6:8], ddst[2:4]))
        new_state = last_state + self.dt*dstate/2.0


        # Check if state has overstepped any of the defined constraints and reached a terminal state
        done = False
        for s, cl, ch in zip(new_state, self.state_clamp_low, self.state_clamp_high):
            if s < cl or s > ch:
                done = True

        # Example how to update the state when done
        self.state = np.array(new_state)


        # Uncite this if computer is burning up
        #time.sleep(0.005)

        if done:
            reward = 0
        else:
            reward = self.reward_function(u)

        # Returns a state observation, a reward, whether the current state was terminal
        return self._get_obs(), reward, done, {}

    def reward_function(self, action_val):
        q = np.array([40, 60, 20, 10, 40, 60, 20, 10])
        r = np.array([0, 0])
        unscaled_x = 1.0 - np.divide(np.power(self.state, 2), np.power(self.state_clamp_high, 2))
        unscaled_u = 1.0 - np.divide(np.power(action_val, 2), np.power(self.action_clamp_high, 2))
        return (np.dot(unscaled_x, q) + np.dot(unscaled_u, r)) * self.dt * 1 / (np.sum(q) + np.sum(r))

    def reset(self):
        """
        Resets the environment. This is where to set up how the state space is initialized.

        :return: The set initial state of the environment
        """

        """
        TODO: Setup an initial state generator. Either random or deterministic
        """

        # Example of how to reset the environment
        high = np.array(
            [0.0 * np.pi / 180.0, 0.5 * np.pi / 180.0, 0.0, 0.0, 0.0 * np.pi / 180.0, 0.5 * np.pi / 180.0, 0.0, 0.0])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.state_seq = [self.state]
        return self._get_obs()

    def _get_obs(self):
        """
        This method is not necessary for the functionality of the environment,
        but rather demonstrates a widely used concept

        :return: The observation of the current state
        """
        # Example of how to generate an observation. E.g. by returning a portion of the state space
        return self.state

    def render(self, mode='human'):
        if self.do_render:
            self.tdrend1.update(self.state[0:3])
            self.tdrend2.update(self.state[4:7])
        if self.do_plotting:
            self.plotting()
        #delay = np.max([0, self.dt-(time.time() - self.stepTimeStart)])
        #time.sleep(delay)
        return None

    def plotting(self):
        self.state_seq = np.append(self.state_seq, [self.state * 180 / np.pi], axis=0)
        #x_range = np.asarray(np.arange(0, np.size(self.state_seq, axis=0)))
        n_steps = np.size(self.state_seq, axis=0)
        x_range = np.linspace(start=0, stop=n_steps*self.dt, num=n_steps)
        self.theta1plot.setData(x_range, self.state_seq[:, 0])
        self.theta2plot.setData(x_range, self.state_seq[:, 4])
        self.alpha1plot.setData(x_range, self.state_seq[:, 1])
        self.alpha2plot.setData(x_range, self.state_seq[:, 5])
        self.dtheta1plot.setData(x_range, self.state_seq[:, 2])
        self.dalpha1plot.setData(x_range, self.state_seq[:, 3])
        self.dtheta2plot.setData(x_range, self.state_seq[:, 6])
        self.dalpha2plot.setData(x_range, self.state_seq[:, 7])

        QtGui.QApplication.processEvents()

    def close(self):
        if self.tdrend1:
            self.tdrend1.close()
        if self.tdrend2:
            self.tdrend2.close()


class TdRenderer:
    def __init__(self):
        from gym_modifier.envs import tdRendering as rendering

        self.screen_width = 600
        screen_height = 400

        self.world_width = 60.0 * np.pi / 180.0 * 2.0

        self.scale = self.screen_width / self.world_width

        self.carty = 100  # TOP OF CART
        self.polewidth = 10.0
        self.polelen = self.scale * (2.0 * 1.0)
        self.cartwidth = 20.0
        self.cartheight = 20.0

        self.viewer = rendering.Viewer(self.screen_width, screen_height)
        l, r, t, b = -self.cartwidth / 2, self.cartwidth / 2, self.cartheight / 2, -self.cartheight / 2
        axleoffset = self.cartheight / 4.0
        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        self.carttrans = rendering.Transform()
        cart.add_attr(self.carttrans)
        self.viewer.add_geom(cart)
        l, r, t, b = -self.polewidth / 2, self.polewidth / 2, self.polelen - self.polewidth / 2, -self.polewidth / 2
        pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        pole.set_color(.8, .6, .4)
        self.poletrans = rendering.Transform(translation=(0, axleoffset))
        pole.add_attr(self.poletrans)
        pole.add_attr(self.carttrans)
        self.viewer.add_geom(pole)
        self.axle = rendering.make_circle(self.polewidth / 2)
        self.axle.add_attr(self.poletrans)
        self.axle.add_attr(self.carttrans)
        self.axle.set_color(.5, .5, .8)
        self.viewer.add_geom(self.axle)
        self.track = rendering.Line((0, self.carty), (self.screen_width, self.carty))
        self.track.set_color(0, 0, 0)
        self.viewer.add_geom(self.track)
        self._pole_geom = pole

    def update(self, state):
        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -self.polewidth / 2, self.polewidth / 2, self.polelen - self.polewidth / 2, -self.polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        stick_len = 0.3111
        x = [np.sin(state[0]) * stick_len, 0, state[1], 0]
        cartx = x[0] * self.scale + self.screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, self.carty)
        self.poletrans.set_rotation(x[2])
        self.viewer.render(return_rgb_array='human' == 'rgb_array')

    def move_window(self):
        self.viewer.move_window()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None