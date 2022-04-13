from torch.utils.cpp_extension import load
import os

module_path = os.path.dirname(__file__)
boost_path = "/home/davit/Git/tmp/boost_1_78_0"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/davit/Git/tmp/boost_1_78_0/stage/lib

load(name="pheonix_cpp",
    sources=[os.path.join(module_path, f"{el}") for el in ["libpheonix.cpp", "scheduler.cpp"]],
    extra_include_paths=[module_path, boost_path],
    extra_cflags=["-fcoroutines", "-std=c++2a"],
    extra_ldflags=["-lcurl", "-L:/home/davit/Git/tmp/boost_1_78_0/stage/lib"], 
    build_directory=os.path.join(module_path, "build"),
    verbose=True)

import pheonix_cpp

# output = pheonix_cpp.simple_request(1)

for el in pheonix_cpp.prefetch():
    print(el)
