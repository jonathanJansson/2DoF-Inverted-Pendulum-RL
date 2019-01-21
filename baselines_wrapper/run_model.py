import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.deepq import DQN
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras_rl.agents.dqn import DQNAgent
from keras_rl.policy import BoltzmannQPolicy
from keras_rl.memory import SequentialMemory
from keras_rl.callbacks import TrainIntervalLogger
from keras_rl.callbacks import FileLogger
from keras_rl.callbacks import ModelIntervalCheckpoint
import keras.models


def run_model(save_name, nw_type, log_dir='./Logs/', log_name=None, env_name='CartPole-v2', runs=100,
              save_results=False):
    # Sets up an environment and a model:
    env = DummyVecEnv([lambda: gym.make(env_name)])
    model = load_model(nw_type=nw_type, log_dir=log_dir, env_name=env_name, log_name=log_name, save_name=save_name)

    # Runs environment with the loaded model "runs" times
    max_reward = 0
    max_steps = 0
    rew_vec = []

    header = 'theta1,alpha1,dtheta1,dalpha1,theta2,alpha2,dtheta2,dalpha2'

    for i in range(runs):
        # Resets the environment
        obs, done = env.reset(), False
        episode_rew = 0
        ep_steps = 0
        obs_vec = obs.reshape(-1,1)
        # This loop runs the environment until a terminal state is reached
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            episode_rew += rewards[-1]
            ep_steps += 1
            obs_vec = np.append(obs_vec, obs.reshape(-1, 1) * 180 / np.pi, axis=1)

        # Saves the reached reward and checks if its a record etc.
        rew_vec.append(episode_rew)
        print("Ep reward: ", '{0:.2f}'.format(episode_rew), '\tRecord: ', '{0:.2f}'.format(max_reward),
              '\tEp steps: ', ep_steps, '\tSteps record: ', max_steps)
        np.savetxt('rew_vec.csv', rew_vec, delimiter=',')
        if episode_rew > max_reward:
            max_reward = episode_rew
            if save_results:
                np.savetxt('obs_vec.csv', obs_vec.T, delimiter=',', header=header, fmt='%1.3f', comments='')
        if ep_steps > max_steps:
            max_steps = ep_steps


def run_double_1DoF(save_name, nw_type, log_dir='./Logs/', log_name=None, runs=100,
              save_results=False):
    # Sets up an environment and a model:
    env_name_od = 'NonLinear1dInvPend-v1'
    env_name_td = 'NonLinear2dInvPend-v1'
    env = DummyVecEnv([lambda: gym.make(env_name_td)])
    model = load_model(nw_type=nw_type, log_dir=log_dir, env_name=env_name_od, log_name=log_name, save_name=save_name)

    # Runs environment with the loaded model "runs" times
    max_reward = 0
    max_steps = 0
    rew_vec = []

    header = 'theta1,alpha1,dtheta1,dalpha1,theta2,alpha2,dtheta2,dalpha2'

    for i in range(runs):
        # Resets the environment
        obs, done = env.reset(), False
        episode_rew = 0
        ep_steps = 0
        obs_vec = obs.reshape(-1, 1)

        # This loop runs the environment until a terminal state is reached
        while not done:
            obs1 = obs[0, 0:4]
            obs2 = obs[0, 4:8]
            action1 = model.predict(obs1)[0]
            action2 = model.predict(obs2)[0]
            action = np.array([action1, action2])
            obs, rewards, done, info = env.step(action)
            env.render()
            episode_rew += rewards[-1]
            ep_steps += 1
            obs_vec = np.append(obs_vec, obs.reshape(-1, 1) * 180 / np.pi, axis=1)

        # Saves the reached reward and checks if its a record etc.
        rew_vec.append(episode_rew)
        print("Ep reward: ", '{0:.2f}'.format(episode_rew), '\tRecord: ', '{0:.2f}'.format(max_reward),
              '\tEp steps: ', ep_steps, '\tSteps record: ', max_steps)
        np.savetxt('rew_vec.csv', rew_vec, delimiter=',')
        if episode_rew > max_reward:
            max_reward = episode_rew
            if save_results:
                np.savetxt('obs_vec.csv', obs_vec.T, delimiter=',', header=header, fmt='%1.3f', comments='')
        if ep_steps > max_steps:
            max_steps = ep_steps


def run_nonLin_2Dof_ddqn(save_name, log_dir='./Logs/', log_name=None, runs=100, save_results=False):
    # Sets up an environment and a model:
    # model = load_model(nw_type=nw_type, log_dir=log_dir, env_name=env_name, log_name=log_name, save_name=save_name)

    env_name_od = 'DiscNonLinear1dInvPend-v1'
    env_name_td = 'NonLinear2dInvPend-v1'

    od_env = gym.make(env_name_od)
    td_end = gym.make(env_name_td)
    nb_actions = od_env.action_space.n
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + od_env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy(tau=1.0)
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, enable_double_dqn=True,
                   target_model_update=1e-2, policy=policy)

    dqn.compile(Adam(lr=0.1), metrics=['mae'])
    dqn = load_model(nw_type='ddqn', log_dir=log_dir, env_name=env_name_od, log_name=log_name, save_name=save_name,
                     model=dqn)
    # Runs environment with the loaded model "runs" times
    max_reward = 0
    max_steps = 0
    rew_vec = []

    header = 'theta1,alpha1,dtheta1,dalpha1,theta2,alpha2,dtheta2,dalpha2'

    obs_od = od_env.reset()

    for i in range(runs):
        # Resets the environment
        obs, done = td_end.reset(), False
        episode_rew = 0
        ep_steps = 0
        obs_vec = obs.reshape(-1, 1)

        # This loop runs the environment until a terminal state is reached
        while not done and ep_steps < 10000:
            obs1 = obs[0:4]
            obs2 = obs[4:8]
            action1 = od_env.env.d_2_c_action(dqn.forward(obs1))[0]
            action2 = od_env.env.d_2_c_action(dqn.forward(obs2))[0]
            action = np.array([action1, action2])
            obs, reward, done, info = td_end.step(action)
            td_end.render()
            episode_rew += reward
            ep_steps += 1
            obs_vec = np.append(obs_vec, obs.reshape(-1, 1) * 180 / np.pi, axis=1)

        # Saves the reached reward and checks if its a record etc.
        rew_vec.append(episode_rew)
        print("Ep reward: ", '{0:.2f}'.format(episode_rew), '\tRecord: ', '{0:.2f}'.format(max_reward),
              '\tEp steps: ', ep_steps, '\tSteps record: ', max_steps)
        np.savetxt('rew_vec.csv', rew_vec, delimiter=',')
        if episode_rew > max_reward:
            max_reward = episode_rew
            if save_results:
                np.savetxt('obs_vec.csv', obs_vec.T, delimiter=',', header=header, fmt='%1.3f')
        if ep_steps > max_steps:
            max_steps = ep_steps


def load_model(nw_type, log_dir, env_name, log_name, save_name, model=None):
    log_dir = log_dir + env_name + '/'
    if log_name is not None:
        log_dir = log_dir + log_name + '/'
    if nw_type.lower() == 'deepq' or nw_type.lower() == 'deep' or nw_type.lower() == 'dqn':
        return DQN.load(log_dir + 'DeepQ/' + save_name + '/' + save_name + '.pkl')
    elif nw_type.lower() == 'ppo' or nw_type.lower() == 'ppo2' or nw_type.lower() == 'ppo1':
        return PPO2.load(log_dir + 'PPO/' + save_name + '/' + save_name + '.pkl')
    elif nw_type.lower() == 'ddqn':
        model.load_weights(log_dir + 'DoubleDQN/' + save_name + '/' + save_name + '.h5f')
        return model
