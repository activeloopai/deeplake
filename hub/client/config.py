import os

TOKEN_FILE_PATH = os.path.expanduser("~/.activeloop/token")
REPORTING_CONFIG_FILE_PATH = os.path.expanduser("~/.activeloop/reporting_config.json")

HUB_REST_ENDPOINT = "https://app.activeloop.ai"
HUB_REST_ENDPOINT_DEV = "https://app.dev.activeloop.ai"
HUB_REST_ENDPOINT_LOCAL = "http://localhost:5000"
USE_LOCAL_HOST = False
USE_DEV_ENVIRONMENT = False

GET_TOKEN_SUFFIX = "/api/user/token"
REGISTER_USER_SUFFIX = "/api/user/register"
GET_DATASET_CREDENTIALS_SUFFIX = "/api/org/{}/ds/{}/creds"
CREATE_DATASET_SUFFIX = "/api/dataset/create"
DATASET_SUFFIX = "/api/dataset"
UPDATE_SUFFIX = "/api/org/{}/dataset/{}"

DEFAULT_REQUEST_TIMEOUT = 170

HUB_AUTH_TOKEN = "HUB_AUTH_TOKEN"
