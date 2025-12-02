#!/bin/bash

set -eo pipefail

# Define global variables
DEB_PATH=$(mktemp -dt pg-deeplake-XXXXXX)
LOG_FILE="update-deb.log"

# Define text styles
YELLOW="\e[93m"
GREEN="\e[92m"
RED="\e[91m"
DEFAULT="\e[0m"

# Define usage function
usage() {
    echo -e "Usage: bash $0 <version-release> <repository> <architecture> <gpg-keyid> <supported-versions>\n\
For example: bash $0 1.2.5-2 /tmp/repo arm64 1F8B584DBEA11E9D"
}

# Check if the arguments are passed
if [ "$#" -ne 5 ]; then
    usage && exit 1
fi

# Error handling function
if_failed() {
    if [ $? -ne 0 ]; then
        touch $LOG_FILE
        echo -e "${RED}$1${DEFAULT}" | tee -a $LOG_FILE
        exit 1
    fi
}

VERSION=$1
REPOSITORY=$2
ARCH=$3
GPG_KEY=$4
CLEAN_VERSION=${VERSION/-*/}
readarray -t -d',' SUPPORTED_VERSIONS < <(printf "%s" "$5")

# Determine artifact directory
if [ "${ARCH}" == "amd64" ]; then
    ARTIFACT_DIR="postgres-x86_64"
elif [ "${ARCH}" == "arm64" ]; then
    ARTIFACT_DIR="postgres-aarch64"
fi

# Create repo directory
mkdir -p "${REPOSITORY}/deb/pg-deeplake/pool/main/"

for version in "${SUPPORTED_VERSIONS[@]}"; do

    # Copy build artifacts
    echo -e "${YELLOW}Copying build artifacts...${DEFAULT}"
    mkdir -p "${DEB_PATH}/pg-deeplake-${version}/DEBIAN/"
    mkdir -p "${DEB_PATH}/pg-deeplake-${version}/usr/lib/postgresql/${version}/lib/"
    mkdir -p "${DEB_PATH}/pg-deeplake-${version}/usr/share/postgresql/${version}/extension/"
    install -m 644 "${ARTIFACT_DIR}"/pg_deeplake_"${version}".so "${DEB_PATH}/pg-deeplake-${version}/usr/lib/postgresql/${version}/lib/pg_deeplake"
    install -m 644 "${ARTIFACT_DIR}"/pg_deeplake*.sql            "${DEB_PATH}/pg-deeplake-${version}/usr/share/postgresql/${version}/extension/"
    install -m 644 "${ARTIFACT_DIR}"/pg_deeplake.control         "${DEB_PATH}/pg-deeplake-${version}/usr/share/postgresql/${version}/extension/"
    echo -e "${GREEN}Done.${DEFAULT}\n"

    # Define control file
    echo -e "${YELLOW}Generating the control file...${DEFAULT}"
    cat << EOF > "${DEB_PATH}/pg-deeplake-${version}/DEBIAN/control"
Package: pg-deeplake-${version}
Version: ${CLEAN_VERSION}
Section: database
Priority: optional
Architecture: ${ARCH}
Maintainer: Activeloop SRE <sre@activeloop.dev>
Description: PostgreSQL ${version} DeepLake Extension.
EOF
    if_failed "Error while generating the control file."
    echo -e "${GREEN}Done.${DEFAULT}\n"

    # Build the deb package
    echo -e "${YELLOW}Building the deb package for ${ARCH} PostgreSQL ${version}...${DEFAULT}"
    rm -f pg-deeplake-"${version}"_"${VERSION}"_"${ARCH}".deb
    dpkg --build "${DEB_PATH}/pg-deeplake-${version}" "pg-deeplake-${version}_${VERSION}_${ARCH}.deb"
    mv -f "pg-deeplake-${version}_${VERSION}_${ARCH}.deb" "${REPOSITORY}/deb/pg-deeplake/pool/main/"
    echo -e "${GREEN}Done.${DEFAULT}\n"
done
if_failed "Error building the packages."

# Create the Packages file for the repository
echo -e "${YELLOW}Generating and gzipping the Packages file...${DEFAULT}"
mkdir -p "${REPOSITORY}/deb/pg-deeplake/dists/stable/main/binary-${ARCH}"
pushd "${REPOSITORY}/deb/pg-deeplake/" > /dev/null &&
dpkg-scanpackages --arch "${ARCH}" pool/ > dists/stable/main/binary-"${ARCH}"/Packages && popd > /dev/null || exit 1
gzip -fk9 "${REPOSITORY}/deb/pg-deeplake/dists/stable/main/binary-${ARCH}/Packages"
if_failed "Error while generating the Packages file."
echo -e "${GREEN}Done.${DEFAULT}\n"

# Function for generating hashes
do_hash() {
    HASH_NAME="$1"
    HASH_CMD="$2"
    echo "${HASH_NAME}:"
    while IFS= read -r -d '' f; do
        f=${f#./}
        case "$f" in
            "Release"|"Release.gpg"|"InRelease") continue ;;
        esac
        hash_value=$(${HASH_CMD} "$f" | awk '{print $1}')
        file_size=$(wc -c < "$f")
        echo " $hash_value $file_size $f"
    done < <(find ./ -type f -print0)
}

# Function for generating and signing the Release file
do_release() {
    local release_file="${REPOSITORY}/deb/pg-deeplake/dists/stable/Release"
    local inrelease_file="${REPOSITORY}/deb/pg-deeplake/dists/stable/InRelease"
    tmp_release=$(mktemp)

    echo -e "${YELLOW}Generating the Release file...${DEFAULT}"
    cat << EOF > "${tmp_release}"
Origin: Activeloop AI
Label: pg-deeplake
Suite: stable
Codename: stable
Version: ${CLEAN_VERSION}
Architectures: amd64 arm64
Components: main
Description: Activeloop Software Repository.
Date: $(date -Ru)
EOF
    pushd "${REPOSITORY}/deb/pg-deeplake/dists/stable" > /dev/null &&
    {
        do_hash "MD5Sum" "md5sum"
        do_hash "SHA1" "sha1sum"
        do_hash "SHA256" "sha256sum"
    } >> "${tmp_release}" && popd > /dev/null || exit 1
    cp "${tmp_release}" "${release_file}"

    echo -e "${YELLOW}Signing the Release file and creating the InRelease file...${DEFAULT}"
    rm -f "${release_file}.gpg" "${inrelease_file}"
    gpg --default-key "${GPG_KEY}" -abs < "${release_file}" > "${release_file}.gpg"
    gpg --default-key "${GPG_KEY}" -abs --clearsign < "${release_file}" > "${inrelease_file}"
}

do_release
if_failed "Error while generating the Release file."
echo -e "${GREEN}Done.${DEFAULT}\n"

# Export GPG key
mkdir -p "${REPOSITORY}"/keys/
gpg --export --armor "${GPG_KEY}" > "${REPOSITORY}"/keys/activeloop.asc
if_failed "Error while exporting the GPG key."
echo -e "${GREEN}Done.${DEFAULT}\n"
