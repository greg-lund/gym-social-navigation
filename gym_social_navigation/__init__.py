from gym.envs.registration import register

register(
    id='social-nav-train-env-v0',
    entry_point='gym_social_navigation.envs:SocialNavEnv',
    kwargs={'test':False},
)
register(
    id='social-nav-test-env-v0',
    entry_point='gym_social_navigation.envs:SocialNavEnv',
    kwargs={'test':True},
)
