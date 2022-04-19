import hub
import numpy as np

test_files = ["sugarshack.mpo"]
static_test_file = ["hopper.fli"]


def test_MPO():
    for path in test_files:
        sample = hub.read(path)
        arr = np.array(sample)
        assert arr.shape[-1] == 3
        assert arr.shape[0] == 480
        assert arr.dtype == 'uint8'
            
def test_FLI():
    for path in static_test_file:
        sample = hub.read(path)
        arr = np.array(sample)
        assert arr.shape[-1] == 128
        assert arr.shape[0] == 128
        assert arr.dtype == 'uint8'