#!/bin/sh

work_dir=$(pwd)
export PYTHONPATH=${work_dir}/../..:$PYTHONPATH
export PATH=${work_dir}/../../tools:$PATH
chmod +x "${work_dir}"/../../tools/tensoralloy

tensoralloy run input.toml
tensoralloy print train/logfile
