import jax

from typing import Optional

def pmean(x, axis_name: Optional[str] = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.pmean(x, axis_name)

def psum(x, axis_name: Optional[str] = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.psum(x, axis_name)
    
def pmin(x, axis_name: Optional[str] = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.pmin(x, axis_name)

def pmax(x, axis_name: Optional[str] = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.pmax(x, axis_name)
    
def unpmap(x, axis_name: Optional[str] = None):
    """
        Only work for pmap(in_axes=0, out_axes=0)
    """
    if axis_name is None:
        return x
    else:
        return x[0]