import deeplake
import numpy as np

root = 'az://testactiveloop/indra-tests/backward_compatibility_datasets'

version = deeplake.__version__

def create_common_ds():
    ds = deeplake.create(f'{root}/{version}/common-10k')
    ds.add_column('number', 'int32')
    ds.add_column('text', 'text')
    ds.add_column('json', deeplake.types.Dict())
    ds.add_column('image', deeplake.types.Image(sample_compression='jpeg'))
    ds.add_column('embedding', deeplake.types.Embedding(32))
    ds.add_column('indexed_text', deeplake.types.Text(deeplake.types.Inverted))
    ds.add_column('struct', deeplake.types.Struct({
        'a': deeplake.types.Int32(),
        'b': deeplake.types.Image(sample_compression='jpeg'),
        'c': deeplake.types.Embedding(32),
        'd': deeplake.types.Struct({
            'e': deeplake.types.Int32(),
            'f': deeplake.types.Image(sample_compression='jpeg'),
            'g': deeplake.types.Embedding(32),
        })
    }))
    ds.commit()
    return ds

def add_data(ds):
    random_words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "pear", "quince", "raspberry", "strawberry", "tangerine", "watermelon"]
    def random_text():
        num_words = np.random.randint(1, 4)
        return " ".join(np.random.choice(random_words, num_words))

    for i in range(10000):
        ds.append([{
            'number': i,
            'text': version,
            'json': {'key': i, 'value': version},
            'image': np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8),
            'embedding': np.random.random(32).astype(np.float32),
            'indexed_text': random_text(),
            'struct': {
                'a': i,
                'b': np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8),
                'c': np.random.random(32).astype(np.float32),
                'd': {
                    'e': i,
                    'f': np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8),
                    'g': np.random.random(32).astype(np.float32),
                }
            }
        }])
    ds.commit()

if __name__ == '__main__':
    ds = create_common_ds()
    add_data(ds)
    print(ds)
