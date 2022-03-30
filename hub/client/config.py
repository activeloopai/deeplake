import os

TOKEN_FILE_PATH = os.path.expanduser("~/.activeloop/token")
HUB_PYPI_VERSION_PATH = os.path.expanduser("~/.activeloop/pypi_version.json")
REPORTING_CONFIG_FILE_PATH = os.path.expanduser("~/.activeloop/reporting_config.json")
ALL_AGREEMENTS_PATH = os.path.expanduser("~/.activeloop/agreements")

HUB_REST_ENDPOINT = "https://app.activeloop.ai"
HUB_REST_ENDPOINT_DEV = "https://app.dev.activeloop.ai"
HUB_REST_ENDPOINT_LOCAL = "http://localhost:7777"
USE_LOCAL_HOST = False
USE_DEV_ENVIRONMENT = False

GET_TOKEN_SUFFIX = "/api/user/token"
REGISTER_USER_SUFFIX = "/api/user/register"
GET_DATASET_CREDENTIALS_SUFFIX = "/api/org/{}/ds/{}/creds"
GET_PRESIGNED_URL_SUFFIX = "/api/org/{}/ds/{}/chunks/url/presigned"
CREATE_DATASET_SUFFIX = "/api/dataset/create"
SEND_EVENT_SUFFIX = "/api/event"
DATASET_SUFFIX = "/api/dataset"
UPDATE_SUFFIX = "/api/org/{}/dataset/{}"
LIST_DATASETS = "/api/datasets/{}"
GET_USER_PROFILE = "/api/user/profile"

DEFAULT_REQUEST_TIMEOUT = 170

HUB_AUTH_TOKEN = "HUB_AUTH_TOKEN"
