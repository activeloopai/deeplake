import os

TOKEN_FILE_PATH = os.path.expanduser("~/.activeloop/token")
HUB_PYPI_VERSION_PATH = os.path.expanduser("~/.activeloop/pypi_version.json")
REPORTING_CONFIG_FILE_PATH = os.path.expanduser("~/.activeloop/reporting_config.json")

HUB_REST_ENDPOINT = "https://app.activeloop.ai"
HUB_REST_ENDPOINT_STAGING = "https://app-staging.activeloop.dev"
HUB_REST_ENDPOINT_DEV = "https://app-dev.activeloop.dev"
HUB_REST_ENDPOINT_TESTING = "https://testing.activeloop.dev"
HUB_REST_ENDPOINT_LOCAL = "http://localhost:7777"
USE_LOCAL_HOST = False
USE_DEV_ENVIRONMENT = False
USE_TESTING_ENVIRONMENT = False
USE_STAGING_ENVIRONMENT = False

GET_DATASET_CREDENTIALS_SUFFIX = "/api/org/{}/ds/{}/creds"
GET_PRESIGNED_URL_SUFFIX = "/api/org/{}/ds/{}/chunks/url/presigned"
GET_BLOB_PRESIGNED_URL_SUFFIX = "/api/org/{}/managed-credentials/blob/url/presigned"
CREATE_DATASET_SUFFIX = "/api/dataset/create"
SEND_EVENT_SUFFIX = "/api/event"
DATASET_SUFFIX = "/api/dataset"
UPDATE_SUFFIX = "/api/org/{}/dataset/{}"
GET_MANAGED_CREDS_SUFFIX = "/api/org/{}/storage/name"
ACCEPT_AGREEMENTS_SUFFIX = "/api/organization/{}/dataset/{}/agree"
REJECT_AGREEMENTS_SUFFIX = "/api/organization/{}/dataset/{}/disagree"
GET_USER_PROFILE = "/api/user/profile"
CONNECT_DATASET_SUFFIX = "/api/dataset/connect"
REMOTE_QUERY_SUFFIX = "/api/query/dataset/{}/{}"

DEFAULT_REQUEST_TIMEOUT = 170

DEEPLAKE_AUTH_TOKEN = "ACTIVELOOP_TOKEN"
DEEPLAKE_AUTH_PROVIDER = "ACTIVELOOP_AUTH_PROVIDER"
ORG_PERMISSION_SUFFIX = "/api/organizations/{}/features/dataset_query"
