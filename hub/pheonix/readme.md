

# Get Started with development
### Setup Compiler

Need gcc to be 10 to support coroutines and c++20. Why not clang? because pytorch is compiled using gcc so for now we will use gcc-10
```
sudo apt install gcc-10 g++-10
```
then point c++ to gcc-10

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 1000 --slave /usr/bin/g++ g++ /usr/bin/g++-10
```

### Install Scheduler
```
git clone https://github.com/lewissbaker/cppcoro
cd cppcoro && ./build-clang.sh
```

### Package manager
- [] try hunter gate