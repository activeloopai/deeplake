#!/bin/bash

# shellcheck source=/dev/null
source /emsdk/emsdk_env.sh
source ./.build_env

if [ "${ENVIRONMENT}" == 'dev' ]; then
    timeout -k "${TIMEOUT}m" "${WARNING}m" bash -c "BUILD_ST=${MULTI_BUILD} yarn build:dev"
else
    timeout -k "${TIMEOUT}m" "${WARNING}m" bash -c "BUILD_ST=${MULTI_BUILD} yarn build"
fi
