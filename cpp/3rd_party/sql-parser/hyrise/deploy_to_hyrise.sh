#!/bin/sh

# Usage: deploy_to_hyrise.sh path/to/hyrise.git


BUILD_PATH=$(readlink -f $(dirname $0))/build

HYRISE_PATH=$1

SQL_PATH=${HYRISE_PATH}/src/lib/access/sql

if [ ! -d $SQL_PATH ]; then
	echo "Could not verify Hyrise path! ${HYRISE_PATH}"
	exit
fi


make -C src/ build

rm ${SQL_PATH}/parser/*
cp ${BUILD_PATH}/* ${SQL_PATH}/parser/
