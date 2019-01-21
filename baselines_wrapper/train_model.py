import gym
from stable_baselines import PPO2
from stable_baselines.deepq import DQN
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np
import os
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from gym_modifier.envs.cartpole import CartPoleEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras_rl.agents.dqn import DQNAgent
from keras_rl.policy import BoltzmannQPolicy
from keras_rl.memory import SequentialMemory
from keras_rl.callbacks import TrainIntervalLogger
from keras_rl.callbacks import FileLogger
from keras_rl.callbacks import ModelIntervalCheckpoint




def train_doubledqn(env_name='CartPole-v1',
                steps=10000,
                lr=5e-4,
                exploration_rate=1.0,
                log_dir='./Logs/',
                log_name = None,
                prev_model = None):
    """
    Wrapper for training a network with DQN

    :param env_name: The name of the environment to load [String]
    :param steps: The number of time-steps to train for [Int]
    :param exploration_rate: The exploration rate for the algorithm [double or whatever]
    :param lr: The learning rate for the algorithm [double or whatever]
    :param log_dir: The base log folder [String]
    :param log_name: Puts the logs in a subdir of this name [String]
    """

    # Generates a folder hierarchy for the logging:

    if log_name is None:
        log_dir = log_dir + env_name + '/' + 'DoubleDQN/double_dqn_{0:.0E}'.format(lr) + '/'
    else:
        log_dir = log_dir + env_name + '/' + log_name + '/DoubleDQN' + '/double_dqn_{0:.0E}'.format(lr) + '/'

    init_logging(log_dir)

    # Get the environment and extract the number of actions.
    env = gym.make(env_name)
    np.random.seed(123)
    env.seed(123)
    #nb_actions = len(env.action_space.sample())
    nb_actions = env.action_space.n

    # Next, we build a very simple model regardless of the dueling architecture
    # if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
    # Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy(tau=exploration_rate)
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, enable_double_dqn=True,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=lr), metrics=['mae'])
    if prev_model is not None:
        dqn.load_weights(prev_model)

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    callbacks = [ModelIntervalCheckpoint(filepath=log_dir + "/double_dqn_{0:.0E}.h5f".format(lr), interval=1000),
                 FileLogger(filepath=log_dir + "/monitor.json", interval=1000)]
    dqn.fit(env, nb_steps=steps, visualize=False, verbose=2, callbacks=callbacks)

    # After training is done, we save the final weights.
    dqn.save_weights(log_dir + "/double_dqn_{0:.0E}.h5f".format(lr), overwrite=True)



def train_deep(env_name='CartPole-v1',
               steps=10000,
               lr=5e-4,
               exploration_fraction=0.1,
               exploration_final_eps=0.02,
               log_dir='./Logs/',
               log_name=None):
    """
    Wrapper for training a network with DQN

    :param env_name: The name of the environment to load [String]
    :param steps: The number of time-steps to train for [Int]
    :param exploration_fraction: The exploration rate for the algorithm [double or whatever]
    :param exploration_final_eps: The final exploration rate after decay [double or whatever]
    :param lr: The learning rate for the algorithm [double or whatever]
    :param log_dir: The base log folder [String]
    :param log_name: Puts the logs in a subdir of this name [String]
    """

    # Generates a folder hierarchy for the logging:
    if log_name is None:
        log_dir = log_dir + env_name + '/' + 'DeepQ/deep_{0:.0E}'.format(lr) + '/'
    else:
        log_dir = log_dir + env_name + '/' + log_name + '/' + 'DeepQ/deep_{0:.0E}'.format(lr) + '/'
    init_logging(log_dir)

    # Generates an environment for the algorithm to train against
    env = DummyVecEnv([lambda: Monitor(gym.make(env_name), log_dir, allow_early_resets=True)])

    # Sets up a modified callback funtion to be able to handle saving etc. (Not really needed)
    best_mean_reward, n_steps, hist_rew = -np.inf, 0, 0

    def callback(_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        nonlocal n_steps, best_mean_reward, hist_rew
        # Print stats every 1000 calls
        if (n_steps + 1) % 5 == 0:
            # Evaluate policy performance
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                # mean_rew_plot(y, len(x))
                hist_rew = y.copy()
                mean_reward = np.mean(y[-100:])
                if (n_steps + 1) % 100 == 0:
                    print(x[-1], 'timesteps')
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(log_dir + "/deep_{0:.0E}.pkl".format(lr))

        n_steps += 1
        return False

    # Creates the training model etc.
    dqn_nw = DQN('MlpPolicy',
                 env,
                 learning_rate=lr,
                 exploration_fraction=exploration_fraction,
                 exploration_final_eps=exploration_final_eps,
                 checkpoint_freq=2000,
                 learning_starts=1000,
                 target_network_update_freq=500)

    # Starts the training:
    dqn_nw.learn(total_timesteps=steps, callback=callback)


def make_env(env_id, rank, seed=0, log_dir=''):
    """
    Generates a function handle for creating a modified Monitor

    :param env_id: The environment name [String]
    :param rank: An additon to the seed to make the monitors have unique seeds
    :param seed:
    :param log_dir: The directory that the monitor saves its "monitor.csv" file in
    :return: The generated function handle
    """

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env, log_dir + 'cpu_' + str(rank) + '/', allow_early_resets=True)
        return env

    set_global_seeds(seed)
    return _init


def train_ppo(env_name='CartPole-v1',
              steps=10000,
              lr=5e-4,
              gamma=0.99,
              max_grad_norm=0.5,
              n_mini_batches=4,
              log_dir='./Logs/',
              n_cpus=1,
              prev_model=None,
              log_name=None):
    """
    Wrapper for training a network with ppo2. 

    :param env_name: The name of the environment to load [String]
    :param steps: The number of timesteps to train for [Int]
    :param lr: The learning rate for the algorithm [Double or whatever]
    :param gamma: Discount factor [Double or whatever]
    :param max_grad_norm: The maximum value for the gradient clipping [Double or whatever]
    :param n_mini_batches: Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches. [Int]
    :param log_dir: The base log folder [String]
    :param n_cpus: The number of environments to setup and run in parallel [Int]
    :param prev_model: Path to any previously trained model to continue training on [String]
    :param log_name: Puts the logs in a subdir of this name [String]
    """

    # Generates a folder hirachy for the logging:
    if log_name is None:
        log_dir = log_dir + env_name + '/' + 'PPO/ppo_{0:.0E}'.format(lr) + '/'
    else:
        log_dir = log_dir + env_name + '/' + log_name + '/' + 'PPO/ppo_{0:.0E}'.format(lr) + '/'
    monitor_dir = log_dir + 'monitors/'
    init_logging(monitor_dir)

    for cp in range(n_cpus):
        if not os.path.isdir(monitor_dir + 'cpu_' + str(cp) + '/'):
            os.mkdir(monitor_dir + 'cpu_' + str(cp) + '/')

    # Generates a set of environments to run on the different cpu's in parallel
    env = SubprocVecEnv([make_env(env_name, i, log_dir=monitor_dir) for i in range(n_cpus)])

    # Sets up a modified callback function to be able to handle saving etc. (Not really needed)
    best_mean_reward = -np.inf
    n_steps = 0

    def callback(_locals, _globals):
        nonlocal n_steps, best_mean_reward
        if (n_steps + 1) % 10 == 0:
            y = np.array([dic['r'] for dic in reversed(_locals['ep_info_buf'])])
            if len(y) > 0:
                mean_reward = np.mean(y)
                print('--------------------------------------')
                print('| Timestep: {0}'.format(_locals['timestep'] * 128))
                print("| Best mean reward: {:.2f} - Reward mean 100: {:.2f}".format(best_mean_reward,
                                                                                    mean_reward))
                print("| Buf length: {:.2f}".format(len(y)))
                print('--------------------------------------')
                #if _locals['ep_info_buf'][-1]['r'] > best_mean_reward:
                #    best_mean_reward = _locals['ep_info_buf'][-1]['r']
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    #print("Saving new best model with rewards {:.2F}".format(_locals['ep_info_buf'][-1]['r']))
                    #_locals['self'].save(log_dir + "/ppo_{0:.0E}".format(lr) + ".pkl")
                    print("Saving new best model with mean rewards {:.2F}".format(mean_reward))
                    _locals['self'].save(log_dir + "/ppo_{0:.0E}".format(lr) + ".pkl")
        n_steps += 1
        return False

    # Loads a previously trained model if there exists one, otherwise creates a new one
    if prev_model is not None:
        ppo2_nw = PPO2.load(prev_model, env=env)
    else:
        ppo2_nw = PPO2('MlpPolicy',
                       env,
                       learning_rate=lr,
                       gamma=gamma,
                       max_grad_norm=max_grad_norm,
                       nminibatches=n_mini_batches)

    # Starts the training:
    ppo2_nw.learn(total_timesteps=int(steps), callback=callback)


def init_logging(log_dir='./Logs/'):
    """
    Generates a folder hierarchy for the logging
    
    :param log_dir: The base directory for the logs
    """
    dirs = log_dir.split(sep='/')
    acc_dir = dirs[0] + '/'
    for d in dirs[1:]:
        acc_dir = acc_dir + '/' + d
        if not os.path.isdir(acc_dir):
            os.mkdir(acc_dir)
