#!/usr/bin/env bash

VCPKG_CACHE_PATH="${VCPKG_CACHE_PATH:-/vcpkg-cache}"
CACHE_BUCKET="${CACHE_BUCKET:-activeloop-platform-tests}"

if [ "${CREDENTIALS_PATH}" ] && [ -f "${CREDENTIALS_PATH}" ]; then
	set -a
	# shellcheck disable=SC1090
	source "${CREDENTIALS_PATH}"
	set +a
fi

case "$(arch)" in
"x86_64" | "amd64")
	ARCH_NAME="x86_64"
	;;
"aarch64" | "arm64")
	ARCH_NAME="aarch64"
	;;
*)
	echo "Unsupported architecture: $(arch)"
	exit 1
	;;
esac

CACHE_PREFIX="vcpkg-cache/indra/${ARCH_NAME}"

if [ "${AWS_ENDPOINT_URL}" ]; then
	__cmd__="s5cmd --endpoint-url ${AWS_ENDPOINT_URL}"
else
	__cmd__="s5cmd"
fi

case "$1" in
"download")
	mkdir -p "${VCPKG_CACHE_PATH}"
	$__cmd__ cp "s3://${CACHE_BUCKET}/${CACHE_PREFIX}/${2}/*" "${VCPKG_CACHE_PATH}/" || {
		echo -e "\n\n\n[WARN] Failed to download cache: $2\n\n\n"
		exit 0
	}
	;;
"upload")
	$__cmd__ cp "${VCPKG_CACHE_PATH}/" "s3://${CACHE_BUCKET}/${CACHE_PREFIX}/${2}/" || {
		echo -e "\n\n\n[WARN] Failed to upload cache: $2\n\n\n"
		exit 0
	}
	;;
*)
	echo "Unsupported command: $1"
	exit 1
	;;
esac
