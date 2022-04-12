from torch.utils.cpp_extension import load
import os

module_path = os.path.dirname(__file__)

load(name="pheonix_cpp",
    sources=[os.path.join(module_path, f"{el}") for el in ["libpheonix.cpp", "scheduler.cpp"]],
    extra_include_paths=[module_path],
    extra_cflags=["-fcoroutines", "-std=c++2a"],
    extra_ldflags=["-lcurl"], 
    build_directory=os.path.join(module_path, f"build"),
    verbose=True)

import pheonix_cpp

# output = pheonix_cpp.simple_request(1)

# iterator = pheonix_cpp.Iterator()

# print(iterator.next())
# print(pheonix_cpp.Iterator.__next__())
print([el for el in pheonix_cpp.Iterator()])
# for i in generator:
#    print(i)




