import argparse
import yaml
from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_world_comm_v3, simple_speaker_listener_v4

type_mapping = {
    'num_episodes': int,
    'len_episodes': int,
    'buffer_size': int,
    'hidden_dim': int,
    'batch_size': int,
    'update_interval': int,
    'minimal_size': int,
    'actor_lr': float,
    'critic_lr': float,
    'gamma': float,
    'tau': float
}


def get_args(config_path):
    """Get the arguments from the config file."""
    parser = argparse.ArgumentParser()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        for key, value in config.items():
            if key in type_mapping:
                arg_type = type_mapping[key]
                parser.add_argument(f'--{key}', type=arg_type, default=value, help=f'{key} argument')
            else:
                parser.add_argument(f'--{key}', default=value, help=f'{key} argument')

    args = parser.parse_args()
    return args


# register more if needed
REGISTRY_ENV = {
    "simple_adversary_v3": simple_adversary_v3,
    "simple_spread_v3": simple_spread_v3,
    "simple_world_comm_v3": simple_world_comm_v3,
    "simple_speaker_listener_v4": simple_speaker_listener_v4,
}


def make_env(env_id):
    env = REGISTRY_ENV[env_id].parallel_env()
    env.reset()
    return env
