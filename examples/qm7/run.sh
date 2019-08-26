#!/bin/sh
if ! [[ -f qm7.db ]]; then
    tensoralloy build database qm7.extxyz --energy-unit=eV
fi

tensoralloy run input.toml
tensoralloy print train/logfile
