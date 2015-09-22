#!/usr/bin/env python
import numpy as np
from utils import *

def Transform_4tensor(tensor,MO2AO,result_class=dict):
"""Basis transformation for 4-tensor. Tensor must be accessible
as tensor[i,j,k,l], but does not need to be sliceable.

MO2AO gives expansion coefficients for the new basis in terms
of the old basis, i.e. |MO_i> = sum_j ( MO2AO[i,j] * |AO_j> )

By default, final result is stored in a dict. However, you can send
your own class as long as it can be accessed like a dict. This allows
for some space savings due to symmetry etc. Intermediate storage
requirements are still exactly 2*N^4 floats, however. """
    
    nbasis=MO2AO.shape[0]

    #index magic for storing symmetric matrices as flat arrays
    triu_sorter=np.triu_indices(nbasis)
    triu_ind=list(zip( *triu_sorter ) )
    n_triu=len(triu_ind)
    
    print 'Transformation 1/2 ijkl->ijmn...',
    transformed=np.zeros( (n_triu,n_triu) )
    b1=np.empty( (nbasis,nbasis) )
    for i_t in IterProg(xrange(n_triu),interval=0.2):
        i,j=triu_ind[i_t]

        for k,l in ( (k,l) for k in xrange(nbasis) 
                           for l in xrange(k,nbasis)): 
            myint = tensor[i,j,k,l]
            b1[k,l] = myint
            b1[l,k] = myint

        b2 = np.dot( b1 , MO2AO.T )
        b3 = np.dot( MO2AO, b2 )

        transformed[i_t] = b3[ triu_sorter ]

    result=result_class()

    
    print 'Transformation 2/2 ijmn->yxmn...',
    c1=np.empty( (nbasis,nbasis) )
    for j_t in IterProg(xrange(n_triu),interval=0.2):
        c1=c1*0.0
        c1[triu_sorter] = transformed[:,j_t]
        c1=c1+c1.T-np.diag(c1.diagonal())

        c2 = np.dot( c1 , MO2AO.T )
        c3 = np.dot( MO2AO, c2 )

        m,n=triu_ind[j_t]
        for x,y in ( (x,y) for x in xrange(nbasis)
                           for y in xrange(x,nbasis)):
            result[ x, y, m, n ] = c3[ x , y ]
        
    
    print 'Tensor transform finished.'
    return result


def Transform_4tensorOld(tensor,MO2AO,result_class=dict):
    """
    Basis transformation for 4-tensor. Tensor must be accessible
    as tensor[i,j,k,l], but does not need to be sliceable.
    
    MO2AO gives expansion coefficients for the new basis in terms
    of the old basis, i.e. |MO_i> = sum_j ( MO2AO[i,j] * |AO_j> )

    By default, final result is stored in a dict. However, you can send
    your own class as long as it can be accessed like a dict. This allows
    for some space savings due to symmetry etc. Intermediate storage
    requirements are still exactly 2*N^4 floats, however.
    """
    
    nbasis=MO2AO.shape[0]

    #index magic for storing upper triangular matrices
    triu_i,triu_j=np.triu_indices(nbasis)
    n_triu=len(triu_i)

    
    print 'Transformation 1/2 ijkl->ijmn...',
    transformed=dict()
    b1=np.empty( (nbasis,nbasis) )
    for i,j in ( (i,j) for i in IterProg(xrange(nbasis),interval=0.2) 
                 for j in xrange(i,nbasis)):

        for k,l in ( (k,l) for k in xrange(nbasis) 
                           for l in xrange(k,nbasis)): 
            myint = tensor[i,j,k,l]
            b1[k,l] = myint
            b1[l,k] = myint

        b2 = np.dot( b1 , MO2AO.T )
        b3 = np.dot( MO2AO, b2 )
        transformed[i , j] = b3

    result=result_class()

    
    print 'Transformation 2/2 ijmn->yxmn...',
    c1=np.empty( (nbasis,nbasis) )
    for m,n in ( (m,n) for m in IterProg(xrange(nbasis),interval=0.2)
                       for n in xrange(m,nbasis)):
        for i,j in ( (i,j) for i in xrange(nbasis)
                           for j in xrange(i,nbasis)):
            c1[i,j]=transformed[i,j][m,n]
            c1[j,i]=transformed[i,j][m,n]

        c2 = np.dot( c1 , MO2AO.T )
        c3 = np.dot( MO2AO, c2 )

        for x,y in ( (x,y) for x in xrange(nbasis)
                           for y in xrange(x,nbasis)):
            result[ x, y, m, n ] = c3[ x , y ]
        
    
    print 'Tensor transform finished.'
    return result



def Transform_element_4tensor(tensor,MO2AO,p,q,r,s):
    """
    Compute element of 'tensor' after transformation from i,j,k,l.
    This is N^8 complexity if you want to transform the entire tensor,
    so don't do that. Useful for checking or just computing a single
    element, though.

    Tensor must be accessible
    as tensor[i,j,k,l], but does not need to be sliceable.
    
    MO2AO gives expansion coefficients for the new basis in terms
    of the old basis, i.e. |MO_i> = sum_j ( MO2AO[i,j] * |AO_j> )
    """
    answer=0.0
    nbasis=MO2AO.shape[0]
    for i,j,k,l in ((i,j,k,l) for i in xrange(nbasis)
                    for j in xrange(nbasis)
                    for k in xrange(nbasis)
                    for l in xrange(nbasis)):

        answer += MO2AO[p,i]*MO2AO[q,j]*MO2AO[r,k]*MO2AO[s,l] *\
                  tensor[i,j,k,l]    
    return answer
