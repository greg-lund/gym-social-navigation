from gym.envs.registration import register

register(
    id='eth-simple-v0',
    entry_point='gym_social_navigation.envs:SocialNavigationEnv',
)
