from gym.envs.registration import register

register(
    id='CartPole-v2',
    entry_point='gym_modifier.envs.cartpole:CartPoleEnv',
)
register(
    id='InvPend2d-v2',
    entry_point='gym_modifier.envs.tdInvPendEnv:InvPendulumEnv',
)
register(
    id='OdInvPend2d-v1',
    entry_point='gym_modifier.envs.odInvPendEnv:OdInvPendulumEnv',
)
register(
    id='NonLinear2dInvPend-v1',
    entry_point='gym_modifier.envs.tdNonLinearInvPendEnv:NonLinInvPendulumEnv',
    max_episode_steps=750,
)
register(
    id='NonLinear1dInvPend-v1',
    entry_point='gym_modifier.envs.NonLinodInvPendEnv:NonLinOdInvPendulumEnv',
    max_episode_steps=750,
)
register(
    id='DiscNonLinear1dInvPend-v1',
    entry_point='gym_modifier.envs.discreteNLODInvPend:DiscNonLinOdInvPendulumEnv',
    max_episode_steps=750,
)