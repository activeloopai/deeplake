from torch.utils.cpp_extension import load
import os
import time
if False: 
    module_path = os.path.dirname(__file__)
    boost_path = "/home/davit/Git/tmp/boost_1_78_0"
    aws_sdk_path = "/home/davit/Git/prefetching-experiments/aws_sdk_build/lib"
    ldaws_sdk_path = "-L:/home/davit/Git/prefetching-experiments/aws_sdk_build/lib/lib"

    load(name="pheonix_cpp",
        sources=[os.path.join(module_path, f"{el}") for el in ["libpheonix.cpp"]], #, "scheduler.cpp"]],
        extra_include_paths=[module_path], # boost_path
        extra_cflags=["-fcoroutines", "-std=c++17"],
        extra_ldflags=["-lcurl", 
                    "-laws-cpp-sdk-core", 
                    "-laws-cpp-sdk-s3"
        ],# , "-L:/home/davit/Git/tmp/boost_1_78_0/stage/lib"], 
        build_directory=os.path.join(module_path, "build"),
        verbose=True)

import pheonix

answers = []
print("fetching starts")

t1 = time.time()
# urls = ["http://localhost:8000" for _ in range(100)]
# urls = ["file://libpheonix.cpp" for _ in range(100)]

urls = ["s3://hub-2.0-datasets/empty_dataset/dataset_meta.json"]
# urls = ["s3://empty/empty"]
for i, el in enumerate(pheonix_cpp.prefetch(urls)):
    t2 = time.time()
    print(f'received it {i}')
    #time.sleep(0.1)
    answers.append(el)
    print(t2-t1)
    t1 = time.time()
    print(el)


print("final number", len(answers))
