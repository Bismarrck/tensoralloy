#!/usr/bin/env bash

cwd=`pwd`

# Detect the number of cores available
if [[ "$(uname)" == "Darwin" ]]; then
	# MacOS
	njobs=`sysctl -n hw.ncpu`
elif [[ "$(uname)" == "Linux" ]]; then
	# GNU/Linux
    njobs=`grep -c ^processor /proc/cpuinfo`
else
	echo "Unknown plaform: $(uname)"
	exit 1
fi

# Cythonize
python compile.py build_ext --inplace --parallel=${njobs}

# First packaging: build a tarball of compiled '*.so' modules.
python compile.py sdist

# Get the info
version=`python -c 'import setup; print(setup.__version__)'`
package="tensoralloy-${version}"
tarball="${package}.tar.gz"

# Extract the tarball
cd dist

if [[ -f ${tarball} ]]; then
    if [[ -d ${package} ]]; then
        rm -rvf ${package}
    fi
    tar -xzvf ${tarball}
else
    echo "The tarball, ${tarball}, cannot be found"
    exit 2
fi

# Build the wheel
cd ${package}
python setup.py bdist_wheel
mv dist/*.whl ..

# Back to the initial directory
cd ${cwd}

# Remove the tmp files
rm -rvf "dist/${package}"
python compile.py clean_ext
