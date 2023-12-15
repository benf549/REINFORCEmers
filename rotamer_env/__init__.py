from gym.envs.registration import register
from rotamer_env.rotamer_env import rotamer_env

register(
    id="rotamer_env/rotamer_env-v0",
    entry_point="rotamer_env.rotamer_env:rotamer_env",
)