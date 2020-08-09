import os

curr_path = os.path.dirname(os.path.abspath(__file__))

TOKEN_FILE_PATH = os.path.expanduser("~/.activeloop/token")
STORE_CONFIG_PATH = os.path.expanduser("~/.activeloop/store")
CACHE_FILE_PATH = os.path.expanduser("~/.activeloop/tmp")
AWSCRED_PATH = os.path.expanduser("~/.aws/credentials")

HUB_REST_ENDPOINT = "https://app.activeloop.ai"
HUB_LOCAL_REST_ENDPOINT = "http://localhost:5000"

DEFAULT_TIMEOUT = 170

GET_TOKEN_REST_SUFFIX = "/api/user/token"
GET_CREDENTIALS_SUFFIX = "/api/credentials"
GET_REGISTER_SUFFIX = "/api/user/register"
GET_DATASET_SUFFIX = "/api/dataset/get"
GET_DATASET_PATH_SUFFIX = "/api/dataset/get/path"

DISTRIBUTED = True