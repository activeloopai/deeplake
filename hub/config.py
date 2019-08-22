import os

curr_path = os.path.dirname(os.path.abspath(__file__))

TOKEN_FILE_PATH = os.path.expanduser("~/.hub/token")
CACHE_FILE_PATH = os.path.expanduser("~/.hub/tmp")
CLOUDVOLUME_PATH = os.path.expanduser("~/.cloudvolume/secrets/aws-secret.json")
AWSCRED_PATH = os.path.expanduser("~/.aws/credentials")
