#!/usr/bin/bash

cd ~/
export mpi=cray
export blas=libsci
export CXX=CC
git clone --recursive https://github.com/icl-utk-edu/slate.git
cd slate/blaspp
make -j8
cd ../lapackpp
make -j8
cd ..
make -j8
mkdir _install
make install prefix=_install
cd _install
export SLATE_INSTALL=$(pwd)
