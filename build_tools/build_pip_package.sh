#### My instance to build on: 
# Thelio Ubuntu 20.04.3 LTS
# CMake version: 3.25.4, the minimum version of CMake required to build LLVM will become 3.20.0
# currently TVM uses 3.18.4 in CI.
# Python 3.7

# 1. Install cmake
mkdir ${HOME}/DEL
cd ${HOME}/DEL
sudo apt install build-essential libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.25.3/cmake-3.25.3.tar.gz
tar -zxvf cmake-3.25.3.tar.gz
cd cmake-3.25.3
./bootstrap
make
sudo make install

# 2. Install Miniconda Python 3.7
cd ${HOME}/DEL
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
sh Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
source ${HOME}/.bashrc
# If Python doesn't work, please manually add path like `${HOME}/miniconda3/bin`
# into env. var "PATH" of ${HOME}/.bashrc

# 3. Install clang 14. gcc might run into issue like <https://github.com/tensorflow/mlir-hlo/issues/9>
wget https://apt.llvm.org/llvm.sh
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
chmod 755 ./llvm.sh
sudo ./llvm.sh 14
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100

# 4. Install Pybind related packages, which are required for Python Binding.
pip install numpy
pip install pybind11
sudo apt-get install libxml2-dev
sudo apt install ninja-build lld ccache

### Build MLIR

# add the following tags in `build_tools/build_mlir.sh` to use clang explicitly.
# -DCMAKE_BUILD_TYPE=RelWithDebInfo \
# -DCMAKE_CXX_COMPILER=clang++ \
# -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
# -DCMAKE_C_COMPILER=clang \
# -DCMAKE_C_COMPILER_LAUNCHER=ccache

# 5. Set up some Env Variables
[[ "$(uname)" != "Darwin" ]] && LLVM_ENABLE_LLD="ON" # OFF if on MacOS
GITHUB_WORKSPACE=$HOME/stablehlo
STABLEHLO_ROOT_DIR=$HOME/stablehlo
LLVM_PROJECT_DIR=$GITHUB_WORKSPACE/llvm-project
LLVM_BUILD_DIR=$GITHUB_WORKSPACE/llvm-build
STABLEHLO_BUILD_DIR=$GITHUB_WORKSPACE/stablehlo-build
STABLEHLO_PYTHON_BUILD_DIR=$GITHUB_WORKSPACE/stablehlo-python-build

# 6. Clone repos
cd $HOME
git clone https://github.com/yongwww/stablehlo
git checkout hlo2relax
cd stablehlo && git clone https://github.com/llvm/llvm-project.git
(cd llvm-project && git fetch && git checkout $(cat ../build_tools/llvm_version.txt))

# 7. Build MLIR
cd $STABLEHLO_ROOT_DIR
./build_tools/build_mlir.sh $LLVM_PROJECT_DIR $LLVM_BUILD_DIR

# 8. Build StableHLO
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

# 9. [Build Python API](https://github.com/openxla/stablehlo/blob/5ded1b1e3e99df37fdbf876b0d11e03a758a5bcf/build_tools/README.md)
# "Building MLIR Python bindings is known to rely on CMake features that
#  require at least version 3.19.  Recommend upgrading to 3.19+ for full"
mkdir -p $STABLEHLO_PYTHON_BUILD_DIR && cd $STABLEHLO_PYTHON_BUILD_DIR
cmake -GNinja -B. ${PWD}/../llvm-project/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=stablehlo \
  -DLLVM_EXTERNAL_STABLEHLO_SOURCE_DIR=${PWD}/.. \
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
