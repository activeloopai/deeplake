import hub

my_schema = {"abc": "uint32"}
ds = hub.Dataset("./data/test_versioning/eg_1", shape=(10,), schema=my_schema, mode="w")
ds["abc", 0] = 1
a = ds.commit("first")
ds["abc", 0] = 2
b = ds.commit("second")
ds["abc", 0] = 3
c = ds.commit("third")
assert ds["abc", 0].compute() == 3
ds.checkout(a)
assert ds["abc", 0].compute() == 1
ds.checkout(b)
assert ds["abc", 0].compute() == 2
ds.checkout(c)
assert ds["abc", 0].compute() == 3
