from .cfg_object import CfgObject


def load(cfg):
    cfg_obj = CfgObject(cfg)
    return cfg_obj.load()
