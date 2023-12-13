from gym.envs.registration import register

register(
    id = "rotamer_env/rotamer_env-v0",
    entry_point="rotamer_env.envs:rotamerEnv",
)