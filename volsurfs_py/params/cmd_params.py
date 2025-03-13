import os
import volsurfs
from volsurfs_py.params.params import Params


class CmdParams(Params):

    root = os.path.dirname(os.path.abspath(volsurfs.__file__))
    paths_config = os.path.join(root, "config", "paths_config.cfg")

    def __init__(self, args_dict):
        # load config file
        super().__init__(None)

        for key, value in args_dict.items():
            # create class attribute from key, value
            setattr(self, key, value)
