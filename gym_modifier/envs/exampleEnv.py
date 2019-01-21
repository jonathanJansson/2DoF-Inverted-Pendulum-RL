import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class ExampleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        """
        TODO: Initiate the environment here
        - Define the state space, action space and restrictions in both spaces.
        The restrictions on the state space determine the terminal state

        - Also define all static variables here. E.g. gravity etc.
        """

        self.np_random = 0
        self.viewer = None

        self.action_clamp_high = np.array([1., 1., 1.])
        self.action_clamp_low = -self.action_clamp_high
        self.state_clamp_high = np.array([1., 1., 1., 1., 1.])
        self.state_clamp_low = - self.state_clamp_high
        n_actions = len(self.action_clamp_high)
        self.action_space = spaces.Box(low=self.action_clamp_low, high=self.action_clamp_high, shape=(n_actions,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=self.state_clamp_low, high=self.state_clamp_high, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        """
        Seeds the environment. A given seed sets the environment such
        that it will always yield the same random sequences

        :param seed: A seed for the environment [float?]
        :return: The generated seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

        # Example save temp instance of last state
        last_state = self.state

        # Example of how to clip actions according to defined restrictions
        u = np.clip(u, self.action_clamp_low, self.action_clamp_high)[0]


        """
        TODO: Perform state transition to the next state here:
        E.g.:
        new_state = transition_matrix_A * last_state + transition_matrix_B * u  (Ax + Bu)
        """
        new_state = "whatever"




        # Example of how to clip states according to defined restrictions
        new_state = np.clip(new_state, -self.state_clamp_low, self.state_clamp_high)

        # Example how to update the state when done
        self.state = np.array(new_state)




        """
        TODO: Define a reward function based on the current state
        """
        reward = 42  # placeholder



        # Returns a state observation, a reward, whether the current state was terminal
        return self._get_obs(), reward, False, {}

    def reset(self):
        """
        Resets the environment. This is where to set up how the state space is initialized.

        :return: The set initial state of the environment
        """

        """
        TODO: Setup an initial state generator. Either random or deterministic
        """

        # Example of how to reset the environment
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        """
        This method is not necessary for the functionality of the environment,
        but rather demonstrates a widely used concept

        :return: The observation of the current state
        """
        # Example of how to generate an observation. E.g. by returning a portion of the state space
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            # ...

        # Do rendering stuffs...

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
