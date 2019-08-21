import hub
import numpy as np

shape = (10, 10, 10)
x = hub.array(shape, name="test/example:1", dtype='uint8')
x[10]
