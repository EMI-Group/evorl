#!/bin/sh

cd "$(dirname "$0")" || exit 1
rm -rf apidocs && make clean && make html
