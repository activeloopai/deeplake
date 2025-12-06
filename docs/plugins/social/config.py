from mkdocs.config.base import Config
from mkdocs.config.config_options import Deprecated, Type

class SocConfig(Config):
    enabled = Type(bool, default = True)