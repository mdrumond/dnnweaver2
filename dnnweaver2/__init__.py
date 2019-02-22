from dnnweaver2.graph import get_default_graph
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint

def get_tensor(shape, name=None, dtype=FixedPoint(16,8), trainable=True, data=None):
    g = get_default_graph()
    return g.tensor(shape=shape, name=name, dtype=dtype, trainable=trainable, data=data)
