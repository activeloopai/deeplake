# Meta arrays

Minimum viable product

1. Setup
Install, then provide AWS credentials and bucket name
```sh
pip3 install -e .
meta configure
```

2. Create an array
```python
mnist = meta.array((50000, 28, 28, 1), name="username/mnist:v1", dtype='float32')
mnist[0, :] = np.random.random((1, 28, 28, 1)).astype('float32')
```

3. Load an array
```python
mnist = meta.load(name='username/mnist:v1')
print(mnist[0,0,0,0])
```