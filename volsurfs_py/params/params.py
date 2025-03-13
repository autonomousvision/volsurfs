import os
import hjson
import inspect

# from abc import ABC, abstractmethod


def props(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if name != "cfg" and not name.startswith("__") and not inspect.ismethod(value):
            pr[name] = value
    return pr


class Params:
    def __init__(self, cfg_path=None):
        # assert cfg_path exists
        if cfg_path is not None:
            assert os.path.exists(
                cfg_path
            ), f"configuration file {cfg_path} does not exist"
            print("loading config file from: ", cfg_path)
            with open(cfg_path, "r") as j:
                self.cfg = hjson.loads(j.read())
        else:
            self.cfg = {}

    def dict(self):
        return props(self)

    def __str__(self):
        # concat all class attributes into a string
        attributes = []
        for key, value in self.dict().items():
            attributes.append(f"{key} : {value}")
        # sort
        attributes.sort()
        return "\n".join(attributes)

    # make parameters subscriptable
    def __getitem__(self, key):
        if key not in self.dict().keys():
            return None
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
