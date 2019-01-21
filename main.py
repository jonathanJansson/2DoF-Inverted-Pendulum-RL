import baselines_wrapper.train_model as tm
from baselines_wrapper.run_model import run_model
from baselines_wrapper.run_model import run_nonLin_2Dof_ddqn
from baselines_wrapper.run_model import run_double_1DoF

if __name__ == '__main__':
    # env_name = 'OdInvPend2d-v1'
    env_name = 'NonLinear2dInvPend-v1'
    # env_name = 'NonLinear1dInvPend-v1'
    # env_name = 'DiscNonLinear1dInvPend-v1'

    tm.train_ppo(env_name=env_name, n_cpus=12, lr=5e-5, steps=200000)


def ppo_lr_decay(env_name, lrs, log_name):
    tm.train_ppo(env_name=env_name, n_cpus=12, lr=lrs[0], steps=1000000, log_name=log_name)

    for i in range(len(lrs) - 1):
        load_name = './Logs/NonLinear2dInvPend-v1/' + log_name + '/PPO/ppo_{:.0E}/ppo_{:.0E}.pkl'.format(lrs[i], lrs[i])
        tm.train_ppo(env_name=env_name, n_cpus=12, lr=lrs[i + 1], steps=500000, log_name=log_name, prev_model=load_name)


"""
------------------------------------------------

Examples on how to use different parts of the project:

------------------------------------------------
"""


def example_train_model(nw_type='ppo'):
    if nw_type == 'ppo':
        tm.train_ppo(env_name='NonLinear2dInvPend-v1', n_cpus=12, lr=1e-4, steps=2000000, log_name='testing')
    elif nw_type == 'ddqn':
        tm.train_doubledqn(env_name='DiscNonLinear1dInvPend-v1', steps=80000000, lr=1e-4, log_name='testing')
    elif nw_type == 'dqn':
        tm.train_deep(env_name=env_name, lr=7e-5, steps=100000)


def example_load_prev_trained_model(nw_type='ppo'):
    if nw_type == 'ppo':
        tm.train_ppo(env_name='NonLinear2dInvPend-v1', n_cpus=12, lr=1e-4, steps=2000000, log_name='testing',
                     prev_model='./Logs/NonLinear2dInvPend-v1/testing/PPO/ppo_1E-05/ppo_1E-05.pkl')
    elif nw_type == 'ddqn':
        tm.train_doubledqn(env_name='DiscNonLinear1dInvPend-v1', steps=80000000, lr=1e-4, log_name='testing',
                           prev_model='./Logs/DiscNonLinear1dInvPend-v1/testing/DoubleDQN/double_dqn_5E-04'
                                      '/double_dqn_5E-04.h5f')


def example_run_trained_model(nw_type='ppo'):
    if nw_type == 'ppo':
        run_model(env_name='NonLinear2dInvPend-v1', save_name='ppo_1E-04', nw_type='ppo', runs=1000, save_results=True,
                  log_name='testing')
    elif nw_type == 'ddqn':
        run_nonLin_2Dof_ddqn(save_name='double_dqn_5E-04', save_results=True, log_name='testing')
    elif nw_type == 'dqn':
        run_model(save_name='deep_5E-04.pkl', nw_type='deep')
