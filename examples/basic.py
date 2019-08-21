import hub
import numpy as np

# Create
x = hub.array(
    (50000, 250, 250, 3),
    name='test/test:v2',
    dtyp='uint8',
    chunk_size=(4, 250, 250, 3)
)

# Upload
x[0] = np.ones((250, 250, 3), dtype='uint8')

# Download
print(x[0].mean())
print(x[1].mean())
print(x.shape, x[1].shape)
