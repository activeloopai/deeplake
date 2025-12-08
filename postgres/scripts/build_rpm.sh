#!/bin/bash

set -eo pipefail

# Define global variables
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
    RPM_ARCH="x86_64"
    ARTIFACT_DIR="postgres-x86_64"
elif [ "${ARCH}" == "arm64" ]; then
    RPM_ARCH="aarch64"
    ARTIFACT_DIR="postgres-aarch64"
fi

# Create repo directory
mkdir -p "${REPOSITORY}/rpm/pg-deeplake/packages"

for version in "${SUPPORTED_VERSIONS[@]}"; do

    # Create spec files
    echo -e "${YELLOW}Generating the spec file...${DEFAULT}"
    echo "Summary: PostgreSQL Extension.
Name: pg-deeplake-${version}
Version: ${CLEAN_VERSION}
Release: 1
Group: database
License: Apache License 2.0
Packager: Activeloop SRE <sre@activeloop.dev>
BuildRoot: $(pwd)

%ifarch aarch64
    %global __strip aarch64-linux-gnu-strip
%else
    %global __strip strip
%endif

%description
PostgreSQL ${version} DeepLake Extension.

%install
mkdir -p %{buildroot}/usr/pgsql-${version}/lib/
mkdir -p %{buildroot}/usr/pgsql-${version}/share/extension/
install -m 755 $(pwd)/${ARTIFACT_DIR}/pg_deeplake_${version}.so %{buildroot}/usr/pgsql-${version}/lib/pg_deeplake
install -m 644 $(pwd)/${ARTIFACT_DIR}/pg_deeplake*.sql          %{buildroot}/usr/pgsql-${version}/share/extension/
install -m 644 $(pwd)/${ARTIFACT_DIR}/pg_deeplake.control       %{buildroot}/usr/pgsql-${version}/share/extension/

%files
/usr/pgsql-${version}/lib/pg_deeplake
/usr/pgsql-${version}/share/extension/pg_deeplake*.sql
/usr/pgsql-${version}/share/extension/pg_deeplake.control
" >pg_deeplake.spec
    echo -e "${GREEN}Done.${DEFAULT}\n"

    # Build the rpm package
    echo -e "${YELLOW}Building the rpm package for ${ARCH} PostgreSQL ${version}...${DEFAULT}"
    rpmbuild --quiet --target "${RPM_ARCH}" -bb pg_deeplake.spec
    cp ~/rpmbuild/RPMS/"${RPM_ARCH}"/pg-deeplake-"${version}"-"${VERSION}"."${RPM_ARCH}".rpm "${REPOSITORY}"/rpm/pg-deeplake/packages/
    if_failed "Error building the PostgreSQL ${version} package."
    echo -e "${GREEN}Done.${DEFAULT}\n"
done
if_failed "Error building the packages."

# Sign the repository
echo -e "${YELLOW}Signing the repository...${DEFAULT}"
echo "%_signature gpg
%_gpg_name ${GPG_KEY}" >~/.rpmmacros
rpm --addsign "${REPOSITORY}"/rpm/pg-deeplake/packages/*.rpm
pushd "${REPOSITORY}"/rpm >/dev/null
createrepo .
gpg --detach-sign --armor --yes repodata/repomd.xml
if_failed "Error while signing the repository."
echo -e "${GREEN}Done.${DEFAULT}\n"
