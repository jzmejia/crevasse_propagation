"""Calculate creep closure rates for model

Currently requires data from ELMER outputs 
in one particular format described in the 
class 


(c) JZM 2023
"""
import pandas as pd
import numpy as np
from scipy import interpn


class creep():
    
    def __init__(self,
                 creep_table,
                 zcreep,
                 sigma,
                 years,
                 Zcrev,
                 interp_method='linear',
                 ):
        self.data = creep_table
        self.zcreep = zcreep
        self.sigma = sigma
        self.years = years
        self.Zcrev = Zcrev
        self.method = interp_method
        self.col_order = ['years', 'sigma', 'zcrev']
        
        
    

def inclusive_slice(a, a_min, a_max, pad=None):
    """Array subset with values on or outside of given range
    
    Given an interval, the array is clipped to the closest values
    corresponding to the interval edges such that the resulting
    array has the shortest length while encompassing the entire
    value range given by a_min and a_max.
    
    No check is performed to ensure ``a_min < a_max``
    
    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min, a_max : array_like or None
        Minimum and maximum value. If ``None``, clipping is not 
        performed on the corresponding edge. 
        Only one of `a_min` and `a_max` may be ``None``. 
        Both are broadcast against `a`.
    pad : int or None, optional
        Include additional a values on either side of range.
        Defaults to None.
    
    Returns
    -------
    clipped_array : ndarray
        An array with elements of `a` corresponding to the 
        inclusive range of `a_min` to `a_max`
    
    Examples
    --------
    >>> a = np.array([0,30,60,90,120,150,180])
    >>> inclusive_slice(a,100,120)
    array([90,120])
    >>> inclusive_slice(a,40,61.7)
    array([30,60,90])
    
    """
    # account for optional padding of window
    i_min = -1 if pad is None else -1-pad
    i_max = 0 if pad is None else pad
    return a[
        np.argwhere(a<=a_min).item(i_min):np.argwhere(a>=a_max).item(i_max)+1]

def tupled_grid_array(df):
    a = []
    for name in df.index.names:
        a.append(df.index.get_level_values(name).drop_duplicates().to_numpy())
    for name in df.columns.names:
        a.append(df.columns.get_level_values(name).drop_duplicates().to_numpy())
    return tuple(a)
        
        
        
    