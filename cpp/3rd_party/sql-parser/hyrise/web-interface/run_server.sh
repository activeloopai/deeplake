#!/bin/sh

# Default Port
PORT="10101"

if [ $# -gt 0 ]; then
	PORT=$1
fi

python -m SimpleHTTPServer ${PORT}