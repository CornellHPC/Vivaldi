# distributed-popcorn

Install CombBLAS
```bash
git clone git@github.com:PASSIONLab/CombBLAS.git
cd CombBLAS
git switch combblas-gpu
mkdir _build && mkdir _install
cd _build
cmake -DCMAKE_INSTALL_PREFIX=../_install ..
cmake --build . --target install
cd ../_install
export COMBBLAS_INSTALL=$(pwd)
```

Install SLATE
```bash
git clone --recursive https://github.com/icl-utk-edu/slate.git
cd slate
export CXXFLAGS=-DSLATE_HAVE_MT_BCAST
mkdir _build && mkdir _install
cd _build
cmake -DCMAKE_INSTALL_PREFIX=../_install -Dbuild_tests=false ..
cmake --build . --target install
cd ../_install
export SLATE_INSTALL=$(pwd)
```

Build source
```bash
git clone git@github.com:mattrrubino/distributed-popcorn.git
cd distributed-popcorn
git switch matt-test
mkdir build && cd build
cmake -DCOMBBLASS_INSTALL=$COMBBLAS_INSTALL -DSLATE_INSTALL=$SLATE_INSTALL ..
cmake --build .
mpiexec main
```
