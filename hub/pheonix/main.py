from torch.utils.cpp_extension import load
import os
import time
module_path = os.path.dirname(__file__)
boost_path = "/home/davit/Git/tmp/boost_1_78_0"

load(name="pheonix_cpp",
    sources=[os.path.join(module_path, f"{el}") for el in ["libpheonix.cpp"]], #, "scheduler.cpp"]],
    extra_include_paths=[module_path], # boost_path
    extra_cflags=["-fcoroutines", "-std=c++2a"],
    extra_ldflags=["-lcurl"],# , "-L:/home/davit/Git/tmp/boost_1_78_0/stage/lib"], 
    build_directory=os.path.join(module_path, "build"),
    verbose=True)

import pheonix_cpp

# output = pheonix_cpp.simple_request(1)
answers = []
for i, el in enumerate(pheonix_cpp.prefetch()):
    print(f'received it {i}')
    answers.append(el)
print("final number", len(answers))
