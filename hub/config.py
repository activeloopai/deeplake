import os

curr_path = os.path.dirname(os.path.abspath(__file__))

TOKEN_FILE_PATH = os.path.expanduser("~/.snarkai/token")
STORE_CONFIG_PATH = os.path.expanduser("~/.snarkai/store")
CACHE_FILE_PATH = os.path.expanduser("~/.snarkai/tmp")
AWSCRED_PATH = os.path.expanduser("~/.aws/credentials")

HUB_REST_ENDPOINT = "https://app.snark.ai"

DEFAULT_TIMEOUT = 170

GET_TOKEN_REST_SUFFIX = "/api/user/token"
GET_CREDENTIALS_SUFFIX = "/api/credentials"
GET_REGISTER_SUFFIX = "/api/user/register"
