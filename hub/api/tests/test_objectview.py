import numpy as np
import hub
from hub.schema import Tensor, Image, Text, Sequence, SchemaDict, BBox

def test_objectview():
    schema = SchemaDict({'a':Tensor((None, None), dtype=int, max_shape=(20,20)),
                     'b': Sequence(dtype=BBox(dtype=float)),
                     'c': Sequence(dtype=SchemaDict({
                         'd': Sequence((), dtype= Tensor((5,5), dtype=float))
                     }))
                    })
    ds = hub.Dataset('./nested_seq', shape=(5,), schema=schema)
    
    # dataset view to objectview
    dv = ds[3:5]
    dv['c', 0] = {'d' : 5 * np.ones((2,2,5,5))}   
    assert (dv[0, 'c', 0, 'd', 0].compute() == 5 * np.ones((5, 5))).all()
    
    # tensorview to object view
    ds['b', 0] = 0.5 * np.ones((5,4))
    tv = ds['b', 0]
    assert (tv[0].compute() == 0.5 * np.ones((4,))).all()
    
    # ds to object view
    assert (ds[3, 'c', 'd'].compute() == 5 * np.ones((2,2,5,5))).all()
    
if __name__=='__main__':
    test_objectview()
