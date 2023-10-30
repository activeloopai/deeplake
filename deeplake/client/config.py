import os

TOKEN_FILE_PATH = os.path.expanduser("~/.activeloop/token")
HUB_PYPI_VERSION_PATH = os.path.expanduser("~/.activeloop/pypi_version.json")
REPORTING_CONFIG_FILE_PATH = os.path.expanduser("~/.activeloop/reporting_config.json")

HUB_REST_ENDPOINT = "https://app.activeloop.ai"
HUB_REST_ENDPOINT_STAGING = "https://app-staging.activeloop.dev"
HUB_REST_ENDPOINT_DEV = "https://app-dev.activeloop.dev"
HUB_REST_ENDPOINT_LOCAL = "http://localhost:7777"
USE_LOCAL_HOST = False
USE_DEV_ENVIRONMENT = False
USE_STAGING_ENVIRONMENT = False

GET_TOKEN_SUFFIX = "/api/user/token"
REGISTER_USER_SUFFIX = "/api/user/register"
GET_DATASET_CREDENTIALS_SUFFIX = "/api/org/{}/ds/{}/creds"
GET_PRESIGNED_URL_SUFFIX = "/api/org/{}/ds/{}/chunks/url/presigned"
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
ORG_PERMISSION_SUFFIX = "/api/organizations/{}/features/dataset_query"

# ManagedService Endpoints
INIT_VECTORSTORE_SUFFIX = "/api/dlserver/vectorstore/init"
GET_VECTORSTORE_SUMMARY_SUFFIX = "/api/dlserver/vectorstore/{}/{}/summary"
DELETE_VECTORSTORE_SUFFIX = "/api/dlserver/vectorstore"

VECTORSTORE_SEARCH_SUFFIX = "/api/dlserver/vectorstore/search"
VECTORSTORE_ADD_SUFFIX = "/api/dlserver/vectorstore/add"
VECTORSTORE_REMOVE_ROWS_SUFFIX = "/api/dlserver/vectorstore/remove"
