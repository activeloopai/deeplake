import hub
import numpy as np
x = hub.array((10,10,10,10), name="davit/example:1", dtype='uint8')
#[0] = np.zeros((1,10,10,10), dtype='uint8') # need to assign
x[1,0,0,0] = 1 
