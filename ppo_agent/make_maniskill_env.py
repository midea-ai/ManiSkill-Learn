from mani_skill_learn.env.env_utils import build_env

def make_maniskill_env(env_args):
    def make_env(rank):

        def _thunk():

            seed = rank
            env = build_env(env_args)

            return env

        return _thunk

    return make_env