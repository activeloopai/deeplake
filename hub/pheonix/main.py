from torch.utils.cpp_extension import load
import os

module_path = os.path.dirname(__file__)

load(name="pheonix_cpp", 
    extra_cflags=["-fcoroutines", "-std=c++2a"],
    extra_ldflags=["-lcurl"], 
    sources=[os.path.join(module_path, "libpheonix.cpp")], verbose=True)

import pheonix_cpp

output = pheonix_cpp.simple_request(1)
# print(output)


