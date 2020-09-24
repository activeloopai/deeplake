from hub.features.features import Primitive,Tensor,FeatureDict 

def serialize(s):
    if isinstance(s,Tensor):
        return serialize_tensor(s)
    elif isinstance(s, FeatureDict):
        return serialize_featuredict(s)
    elif isinstance(s, Primitive):
        return serialize_primitive(s)

def serialize_tensor(tensor):
    d={}
    d["type"]="Tensor"
    d["shape"]=tensor.shape
    d["dtype"]=serialize(tensor.dtype)
    return d

def serialize_featuredict(fdict):
    d={}
    d["type"]="FeatureDict";
    d["items"]={}
    for k,v in fdict.__dict__["dict_"].items():
        d["items"][k]=serialize(v)
    return d

def serialize_primitive(primitive):
    return primitive._dtype.str