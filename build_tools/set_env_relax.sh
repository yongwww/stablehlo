### Env Setup

# My Env: AWS p3, Ubuntu 18

#   install cmake
sudo apt install build-essential libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.24.4/cmake-3.24.4-linux-x86_64.tar.gz
tar -zxvf cmake-3.24.4-linux-x86_64.tar.gz
cd cmake-3.24.4
./bootstrap
make
sudo make install

#   miniconda python3.8
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
sh Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
# add path like `/home/ubuntu/miniconda3/bin` into PATH

# install clang 14. gcc might run into issue like <https://github.com/tensorflow/mlir-hlo/issues/9>
wget https://apt.llvm.org/llvm.sh
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo ./llvm.sh 14
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang++-14 100


# install Pybind related packages, it was required for python binding.
pip install numpy
pip install pybind11
sudo apt-get install libxml2-dev

### Build mlir

# add the following tags in `build_tools/build_mlir.sh` to use clang explicitly.
# -DCMAKE_BUILD_TYPE=RelWithDebInfo \
# -DCMAKE_CXX_COMPILER=clang++ \
# -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
# -DCMAKE_C_COMPILER=clang \
# -DCMAKE_C_COMPILER_LAUNCHER=ccache

# set up some Env Variables
    [[ "$(uname)" != "Darwin" ]] && LLVM_ENABLE_LLD="ON" # OFF if on MacOS
    # Set up some Env variables
    GITHUB_WORKSPACE=$HOME/stablehlo
    STABLEHLO_ROOT_DIR=$HOME/stablehlo
    LLVM_PROJECT_DIR=$GITHUB_WORKSPACE/llvm-project
    LLVM_BUILD_DIR=$GITHUB_WORKSPACE/llvm-build
    STABLEHLO_BUILD_DIR=$GITHUB_WORKSPACE/stablehlo-build
    STABLEHLO_PYTHON_BUILD_DIR=$GITHUB_WORKSPACE/stablehlo-python-build

# clone repos
git clone https://github.com/openxla/stablehlo
cd stablehlo && git clone https://github.com/llvm/llvm-project.git
(cd llvm-project && git fetch && git checkout $(cat ../build_tools/llvm_version.txt))

# build MLIR
cd $STABLEHLO_ROOT_DIR
./build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build

#   build StableHLO
mkdir -p $STABLEHLO_BUILD_DIR && cd $STABLEHLO_BUILD_DIR

cmake .. -GNinja \
  -DLLVM_ENABLE_LLD="$LLVM_ENABLE_LLD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DSTABLEHLO_ENABLE_STRICT_BUILD=ON

# test the build, no errors are expected.
cd "$STABLEHLO_BUILD_DIR"
ninja check-stablehlo

# [Build Python API](https://github.com/openxla/stablehlo/blob/5ded1b1e3e99df37fdbf876b0d11e03a758a5bcf/build_tools/README.md)
mkdir -p $STABLEHLO_PYTHON_BUILD_DIR && cd $STABLEHLO_PYTHON_BUILD_DIR
# Build Python API
cmake -GNinja \
  -B"$STABLEHLO_PYTHON_BUILD_DIR" \
  $LLVM_PROJECT_DIR/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=stablehlo \
  -DLLVM_EXTERNAL_STABLEHLO_SOURCE_DIR="$STABLEHLO_ROOT_DIR" \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DPython3_EXECUTABLE=$(which python3) \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DSTABLEHLO_ENABLE_STRICT_BUILD=ON

# test the build, expects to pass all test cases
ninja check-stablehlo-python

# use the python api
ninja StablehloUnifiedPythonModules
export PYTHONPATH=$STABLEHLO_PYTHON_BUILD_DIR/tools/stablehlo/python_packages/stablehlo
python -c "import mlir.dialects.chlo; import mlir.dialects.stablehlo"
