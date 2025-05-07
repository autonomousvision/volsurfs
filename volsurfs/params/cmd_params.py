import os
import volsurfs
from volsurfs.params.params import Params

class CmdParams(Params):

    # use launching pwd as root
    root = os.getcwd()
    paths_config = os.path.join(root, "config", "paths_config.cfg")

    def __init__(self, args_dict):
        # load config file
        super().__init__(None)

        for key, value in args_dict.items():
            # create class attribute from key, value
            setattr(self, key, value)
