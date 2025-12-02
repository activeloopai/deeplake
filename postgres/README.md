# pg_deeplake

## Building on Linux

### Requirements

- pkg-config
- autoconf-archive
- libglew
- libiconv
- CMake 3.29.2 or upper
- Clang 16 or upper
- vcpkg


To build the extension you have to install the dependencies mantioned above.

For building from  python you have to have following dependencies installed for your python version:

- python-venv
- python-dev # for python3.10 -> python3.10-dev

### Installing Dependencies

```
sudo apt update
sudo apt install curl zip unzip tar -y
sudo apt install clang -y
sudo apt install pkg-config -y
sudo apt install autoconf-archive -y
sudo apt install make -y
sudo apt install flex -y
sudo apt install bison -y
sudo apt install libglew-dev -y

wget https://ftp.gnu.org/gnu/libiconv/libiconv-1.17.tar.gz
tar -xzf libiconv-1.17.tar.gz
cd libiconv-1.17
./configure --prefix=/usr/local
make
sudo make install
cd ../

wget https://github.com/Kitware/CMake/releases/download/v4.0.5/cmake-4.0.5-linux-x86_64.sh
chmod +x cmake-4.0.5-linux-x86_64.sh
./cmake-4.0.5-linux-x86_64.sh
cd cmake-4.0.5-linux-x86_64/bin/
export PATH="$PATH:`pwd`"
cd ../../
```

### vcpkg Configuration

VCPkg is installed via a git repository which a VCPKG_ROOT environment variable points to.
The VCPKG_ROOT can be whatever location on your machine you want.

```
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg/
export VCPKG_ROOT=`pwd`
git checkout 6f29f12e82a8293156836ad81cc9bf5af41fe836
./bootstrap-vcpkg.sh
echo "export VCPKG_ROOT=$VCPKG_ROOT" >> ~/.bashrc # or path to your shell config file
echo "export PATH=$PATH:$VCPKG_ROOT" >> ~/.bashrc # or path to your shell config file
source ~/.bashrc
cd ../
```

### Build Instructions

TODO: Refine build commands and steps

```
python3 scripts/build_pg_ext.py [debug|dev|prod]
```

### To run the tests

```
cd postgres/tests
make test
```
