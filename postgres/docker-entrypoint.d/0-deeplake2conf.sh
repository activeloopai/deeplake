#!/bin/bash

CONF="${PGDATA:-/var/lib/postgresql/data}/postgresql.conf"

if ! ( grep "^shared_preload_libraries" "${CONF}" &>/dev/null ); then
	echo "adding shared_preload_libraries to postgresql.conf"
	echo "shared_preload_libraries = 'pg_deeplake'" >> "${CONF}"
else
	if ! ( grep -E "shared_preload_libraries.+pg_deeplake" "${CONF}" &>/dev/null ); then
		echo "adding pg_deeplake to shared_preload_libraries in postgresql.conf"
    cur_content="$( grep -E "^shared_preload_libraries" "${CONF}" | tail -1 | awk '{print $NF}')"
    if [ "${cur_content}" == "''" ]; then
      echo "shared_preload_libraries = 'pg_deeplake'" >> "${CONF}"
    else
      sed -i "/^shared_preload_libraries/s/'$/,pg_deeplake'/g" "${CONF}"
    fi
	else
		echo "pg_deeplake already added to shared_preload_libraries"
	fi
fi

pg_ctl -D "${PGDATA}" restart
