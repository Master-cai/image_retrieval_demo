# /usr/bin/zsh
rm -r build
mkdir build
cd build
cmake ..
make -j16
