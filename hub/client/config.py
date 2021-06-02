import os

TOKEN_FILE_PATH = os.path.expanduser("~/.activeloop/token")

HUB_REST_ENDPOINT = "https://app.activeloop.ai"

GET_TOKEN_SUFFIX = "/api/user/token"
REGISTER_USER_SUFFIX = "/api/user/register"
GET_DATASET_CREDENTIALS_SUFFIX = "/api/org/%s/ds/%s/creds"

DEFAULT_REQUEST_TIMEOUT = 170

HUB_AUTH_TOKEN = "HUB_AUTH_TOKEN"
