import numpy as np
from skimage.util.shape import view_as_windows

def extent(r1,r2,n):
    lr1=np.log10(r1)
    lr2=np.log10(r2)

    return 10.0**(np.linspace(lr1,lr2,n))

def expand2d(a, repFact):
    """Rebin a 2d array to a large size.
    output will be xrepfact,xrepfact of the original
    Basic memory enhancements included to speed things up.
    """
    (A,B)=a.shape
    #both of the below methods take equal length of time to run regardless of array size
    #but much much faster for larger array sizes. Making code easier with the faster version
    #old version kept here.
    #if A*repFact<1000 and B*repFact<1000:
    #    return np.repeat(np.repeat(a,repFact,axis=0),repFact,axis=1)/(repFact*repFact)
    #else:
    #    out=np.zeros((A*repFact,B*repFact),dtype=a.dtype)
    #    for i in range(A):
    #        r=np.repeat(a[i],repFact)
    #        for j in range(repFact):
    #            out[i*repFact+j,:]=r
    #    return out/(float(repFact)*float(repFact))
    out = np.zeros((A * repFact, B * repFact), dtype=a.dtype)
    for i in range(A):
        r = np.repeat(a[i], repFact)
        for j in range(repFact):
            out[i * repFact + j, :] = r
    return out / (float(repFact) * float(repFact))

def downSample2d(arr,sf):
    isf2 = 1.0/(sf*sf)
    (A,B) = arr.shape
    windows = view_as_windows(arr, (sf,sf), step = sf)
    return windows.sum(3).sum(2)*isf2



"""
try:

    from numba import jit

    def downSample2dold(a, sampFact):
        (A, B) = a.shape
        o = np.zeros((int(A/sampFact),int(B/sampFact)),dtype='float64')
        return ds2d_func(a,o,A,B,sampFact)

    @jit(nopython=True)
    def ds2d_func(a,o,A,B,sampFact):
        isf2 = 1.0/(sampFact*sampFact)
        for i in range(0,A,sampFact):
            for j in range(0,B,sampFact):
                o[int(i/sampFact),int(j/sampFact)] = np.sum(a[i:i+sampFact,j:j+sampFact])
        return o*isf2


except:
    def downSample2d(a,sampFact):
        (A,B)=a.shape
        A = int(A/sampFact)
        B = int(B/sampFact)
        return np.array([np.sum(a[i:i+sampFact,j:j+sampFact]) for i in range(0,sampFact*A,sampFact) for j in range(0,sampFact*B,sampFact)]).reshape(A,B)/(sampFact*sampFact)

"""


class line:
    def __init__(self,p1,p2):
        if p2[0]!=p1[0]:
            self.m = (p2[1]-p1[1])/(p2[0]-p1[0])
        else:
            self.m = 0.0
        self.b = p2[1]-self.m*p2[0]
        self.xlim = np.array([min(p1[0],p2[0]),max(p1[0],p2[0])])
        self.ylim = np.array([min(p1[1],p2[1]),max(p1[1],p2[1])])

    def __call__(self,x):
        return self.m*x+self.b
