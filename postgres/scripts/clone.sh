#!/bin/bash

set -e

mkdir -p "${REPOSITORY}"

AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}" AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}" \
aws s3 ls s3://"${R2_BUCKET_NAME}"/ --recursive --endpoint-url "${R2_ENDPOINT_URL}" | awk '{print $4}' | while read -r object_key; do
    dir_path=$(dirname "${object_key}")

    if [ "${dir_path}" != "." ]; then
        mkdir -p "${REPOSITORY}/${dir_path}"
    fi

    touch "${REPOSITORY}/${object_key}"
done
