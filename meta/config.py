import os

curr_path = os.path.dirname(os.path.abspath(__file__))

TOKEN_FILE_PATH = os.path.expanduser("~/.meta/token")
CLOUDVOLUME_PATH = os.path.expanduser("~/.cloudvolume/secrets/aws-secret.json")