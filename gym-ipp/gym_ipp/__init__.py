from gym.envs.registration import register

register(
    id="ipp-v0",
    entry_point="gym_ipp.envs:IppEnv",
)
