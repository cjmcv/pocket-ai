
mkdir build-win-x86
pushd build-win-x86
cmake -G "MinGW Makefiles" ..
mingw32-make -j8
popd