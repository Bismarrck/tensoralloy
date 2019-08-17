#!/usr/bin/env bash

work_dir=$(pwd)
export PYTHONPATH=${work_dir}/../..:$PYTHONPATH
export PATH=${work_dir}/../../tools:$PATH
chmod +x "${work_dir}"/../../tools/tensoralloy

if ! [ -f qm7.db ]; then
    tensoralloy build database qm7.extxyz --energy-unit=eV
fi

tensoralloy run input.toml
