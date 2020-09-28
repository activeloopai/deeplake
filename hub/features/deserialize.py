from hub.features.features import Primitive,Tensor,FeatureDict 
import numpy as np
from hub.features.serialize import serialize


def deserialize(inp):
    if isinstance(inp,dict):
        if inp["type"] =="Tensor":
            return Tensor(tuple(inp["shape"]),deserialize(inp["dtype"]))
        elif inp["type"] =="FeatureDict":  
            d={}
            for k,v in inp["items"].items():
                d[k]=deserialize(v)
            return FeatureDict(d)
    else:
        return Primitive(inp)
