#!/usr/bin/env bash

############################################
## x86
mkdir -p build-x86
pushd build-x86
cmake -DCMAKE_BUILD_TYPE=DEBUG ..
make -j8
popd