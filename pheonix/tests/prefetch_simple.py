import pheonix
import time

answers = []
print("fetching starts")

t1 = time.time()
# urls = ["http://localhost:8000" for _ in range(100)]
# urls = ["file://libpheonix.cpp" for _ in range(100)]

urls = ["s3://hub-2.0-datasets/empty_dataset/dataset_meta.json" for _ in range(10)]
# urls = ["s3://empty/empty"]
for i, el in enumerate(pheonix.prefetch(urls)):
    t2 = time.time()
    print(f'received it {i}')
    #time.sleep(0.1)
    answers.append(el)
    print(t2-t1)
    t1 = time.time()
    print(el)
            

print("final number", len(answers))