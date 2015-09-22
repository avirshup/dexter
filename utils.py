#!/usr/bin/env python
from copy import deepcopy
import cclib

from PyQuante import Ints,Molecule
import os,sys,argparse,gzip as gz,numpy as np
    

def AOOlapNorm( mat, glf, iOrb):
    norm=0.0
    for ib in xrange(glf.nbasis):
        for jb in xrange(glf.nbasis):
            norm+=mat[iOrb,ib]*mat[iOrb,jb]*glf.mocoeffs_sao[ib,jb]

    return norm


def Depickle(fname,gzip=False):
    import cPickle as pickle
    if fname.split('.')[-1]=='gz': gzip=True
    if gzip:
      myfile=fastGZ(fname)
    else:
      myfile=open(fname,'r')
    temp=pickle.load(myfile)
    myfile.close()
    return temp

def Enpickle(object,fname,protocol=2,gzip=False):
    import cPickle as pickle
    if fname.split('.')[-1]=='gz': gzip=True
    if gzip:
      myfile=gz.open(fname,'wb')
    else:
      myfile=open(fname,'wb')
    pickle.dump(object,myfile,protocol)
    myfile.close()


def submatrix( mat, indices ):
    return mat[ [ [x] for x in indices ], indices ]

"""Read a zip file directly into memory, then iterate through it.
MUCH faster than using gzip.open, which is broken."""
def fastGZ(fname):
    import cStringIO
    io_method = cStringIO.StringIO
    import subprocess
    p=subprocess.Popen(["gunzip","-c",fname],stdout=subprocess.PIPE)
    fh=io_method(p.communicate()[0])
    assert p.returncode==0
    return fh

def extractnum(fff):
    outstr=''
    for char in fff:
        if char.isdigit(): outstr+=char
    return int(outstr)

def basename(name,strip_ext=True):
    if strip_ext==True: return name.split('/')[-1].split('.')[0]
    elif strip_ext:
        return name.split('/')[-1].rstrip(strip_ext)
    else: return name.split('/')[-1]

def IterProg( iter , interval=0.05,tot=None):
    try:
        if tot is None:
            tot=len(iter)
        if int(tot*interval)<=0:
            interval=1.0/tot
        haslen=True
    except TypeError:
        haslen=False
        if interval<1: interval=100

    for i,item in enumerate(iter):
        yield item
        if haslen:
            if i % int(tot*interval)==0:
                print str( (100*i)/tot ) + '% ',
                sys.stdout.flush()
        else:
            if i%interval==0:
                print '\rProgress:',i,
                sys.stdout.flush()
    print 'done'
    raise StopIteration



#Assumes (in the NBO basis): D,D*=1,HOMO+1, A,A*=HOMO,last
#Puts D on HOMO-1, D* on LUMO+1, A on HOMO, A* on LUMO
#mocoeffs_nbo[NBO index,AO index]
myNorm=1/np.sqrt(2.0)
def Diab( glf ):
    gl2=deepcopy(glf)
    homo=gl2.homos[0]

    # Not high enough precision - need to recompute using NBO results
    #    moen=glf.moenergies[-1]
    #    print 'Splitting!' , (moen[homo+2] + moen[homo]-
    #                          moen[homo-1] - moen[homo+1])

    fock = np.dot(glf.mocoeffs_mo2nbo,
                  ( np.dot( glf.mocoeffs_mo2nbo, glf.fock_nbo) ).T )
    moen=np.diag(fock)
    print 'Splitting / H', (moen[homo+2]-moen[homo-1]) - (
        moen[homo+1]-moen[homo] )

    #Rotate the things - easier to do with an orthonormal basis
    gl2.mocoeffs_mo2nbo[homo,:]=glf.mocoeffs_mo2nbo[homo,:]+glf.mocoeffs_mo2nbo[homo-1,:]
    gl2.mocoeffs_mo2nbo[homo-1,:]=glf.mocoeffs_mo2nbo[homo,:]-glf.mocoeffs_mo2nbo[homo-1,:]
    
    gl2.mocoeffs_mo2nbo[homo+1,:]=glf.mocoeffs_mo2nbo[homo+1,:]+glf.mocoeffs_mo2nbo[homo+2,:]
    gl2.mocoeffs_mo2nbo[homo+2,:]=glf.mocoeffs_mo2nbo[homo+1,:]-glf.mocoeffs_mo2nbo[homo+2,:]

    gl2.mocoeffs_mo2nbo[homo-1:homo+3,:] *= myNorm

    #Reindex as needed
    if abs(gl2.mocoeffs_mo2nbo[homo-1,0])<abs(gl2.mocoeffs_mo2nbo[homo,0]):
        gl2.mocoeffs_mo2nbo[[homo,homo-1],:] = gl2.mocoeffs_mo2nbo[[homo-1,homo],:]

    if abs(gl2.mocoeffs_mo2nbo[homo+2,homo+1])<abs(gl2.mocoeffs_mo2nbo[homo+1,homo+1]):
        gl2.mocoeffs_mo2nbo[[homo+2,homo+1],:] = gl2.mocoeffs_mo2nbo[[homo+1,homo+2],:]

        
    #Recalculate diabatized MOs
    gl2.mocoeffs_mo2ao=np.dot( gl2.mocoeffs_mo2nbo , gl2.mocoeffs_nbo )

    print '----- MOs in AOs: -----'
    for i in xrange( glf.nbasis ):
        print ' MO %d:'%(i+1),gl2.mocoeffs_mo2ao[i,:]

    #Delete these to make sure they are not used
    delattr(gl2,'mocoeffs')
    delattr(gl2,'moenergies')
        
    return gl2


def BetterLabels(glf,DAtoms,AAtoms):
    """Label NBOs in a straightforward, readable manner.
    Returns dict: s.t. dict[NBO index]=( (atom 1, atom 2),
                                         QuickName)
    where atoms 1 and 2 are the atom IDs on either side of the bond,
    and QuickName gives a renumbered name in which the bonding and
    antibonding versions of the same bond are given as X and X*
    """
    
    D_ids=None
    A_ids=None
    
    nboatoms=[]
    excitations=[]
    bondidx=1
    bondids={}
    nbo_map={}
    for id,id_nbo in enumerate(nbo_ids):
        name=glf.monames_nbo[id_nbo]
        fields=name.split()
        if fields[2]=='BD':
            isH=fields[8]
            nboatoms.append( ( int(fields[6]), int(fields[9]) ) )
            excitations.append('')
            firstid=nboatoms[-1][0]
        elif fields[2]=='BD*(':
            isH=fields[7]
            nboatoms.append( ( int(fields[5]), int(fields[8]) ) )
            excitations.append('*')
            firstid=nboatoms[-1][0]
        elif fields[2]=='CR':
            isH=fields[5]
            nboatoms.append( int( fields[6] ) )
            excitations.append('')
            firstid=nboatoms[-1]
        elif fields[2]=='RY*(':
            isH=fields[4]
            nboatoms.append( int( fields[5] ) )
            excitations.append('*')
            firstid=nboatoms[-1]
        else: raise ValueError( name )

        if nboatoms[-1] not in bondids:
            bondids[ nboatoms[-1] ]=bondidx
            bondidx+=1

        nbo_map[ id_nbo ] = ( nboatoms[-1] ,
                             str(bondids[nboatoms[-1]])+excitations[-1])

    return nbo_map


        
def LabelNBOs( glf, nbo_ids=None, DBond=None, ABond=None ):
    """Label NBOs in a straightforward, readable manner.
    Returns dict: s.t. dict[NBO index]=( (atom 1, atom 2),
                                         QuickName)
    where atoms 1 and 2 are the atom IDs on either side of the bond,
    and QuickName gives a renumbered name in which the bonding and
    antibonding versions of the same bond are given as X and X*
    """

    if DBond: DBond=tuple( sorted( DBond) )
    if ABond: ABond=tuple(sorted(ABond) )
    
    if nbo_ids is None: nbo_ids=range( glf.nbasis )
    
    nboatoms=[]
    excitations=[]
    bondidx=1
    bondids={}
    nbo_map={}
    for id,id_nbo in enumerate(nbo_ids):
        name=glf.monames_nbo[id_nbo]
        fields=name.split()
        if fields[2]=='BD':
            isH=fields[8]
            nboatoms.append( ( int(fields[6]), int(fields[9]) ) )
            excitations.append('')
            firstid=nboatoms[-1][0]
        elif fields[2]=='BD*(':
            isH=fields[7]
            nboatoms.append( ( int(fields[5]), int(fields[8]) ) )
            excitations.append('*')
            firstid=nboatoms[-1][0]
        elif fields[2]=='CR':
            isH=fields[5]
            nboatoms.append( int( fields[6] ) )
            excitations.append('')
            firstid=nboatoms[-1]
        elif fields[2]=='RY*(':
            isH=fields[4]
            nboatoms.append( int( fields[5] ) )
            excitations.append('*')
            firstid=nboatoms[-1]
        else: raise ValueError( name )

        if nboatoms[-1] not in bondids:
            bondids[ nboatoms[-1] ]=bondidx
            bondidx+=1

        nbo_map[ id_nbo ] = ( nboatoms[-1] ,
                             str(bondids[nboatoms[-1]])+excitations[-1])

    return nbo_map




def GetCMO2NBO(gResult):
    """Get matrix describing CMOs in NBO basis"""
    if hasattr(gResult,'mocoeffs_mo2nbo'):
        return gResult.mocoeffs_mo2nbo
    elif not hasattr(gResult,'CMO2NBO'):
        gResult.CMO2NBO=np.einsum( 'ki,lj,ij',gResult.mocoeffs[-1],
                                   gResult.mocoeffs_nbo,
                                   gResult.aooverlaps )

    return gResult.CMO2NBO
    


def ShowPrincipalNBOs( glf, dHomo ,nShow=5):
    """Describe the most important NBOs"""
    CMO2NBO=GetCMO2NBO( glf )
    moindex = glf.homos[0]+dHomo
    ranked=np.argsort( -np.abs(CMO2NBO[moindex]) )
    print '\n****** NBO components of canonical HOMO%+d'%dHomo
    for iNBO in ranked[:nShow]:
        print '%5.2f%%: NBO %s'%( 100.0*(CMO2NBO[moindex,iNBO]**2),
                          glf.monames_nbo[iNBO])

        

def WriteGauDeck( smi , basis, gaufile , name='unnamed'):
    from myutils import NewMol,myomega,oe
    mymol=NewMol(smi)
    myomega(mymol)
    if mymol.NumConfs()>1:
        print 'Only printing first conformation of %d total'%mymol.NumConfs()
    ofs=oe.oemolostream()
    ofs.SetFormat(oe.OEFormat_XYZ)
    ofs.openstring()
    oe.OEWriteMolecule(ofs,mymol.GetConfs().next())
    ofs.close()

    outfile=open(gaufile,'w')
    print >>outfile, '%%chk=./%s.chk'%gaufile.split('/')[-1].split('.')[0]
    print >>outfile, '# OPT HF/%s'%basis
    print >>outfile, '\n%s\n\n0 1'%name
    mygeom = ofs.GetString()
    for line in mygeom.split('\n')[2:]:
        print >>outfile,line

    print >>outfile,'\n--link1--'
    print >>outfile, '%%chk=./%s.chk'%gaufile.split('/')[-1].split('.')[0]
    print >>outfile, ('# HF/%s GFOldPrint GFInput pop=(NBORead) IOp(3/33=1) geom=check ' \
          'guess=(check only)')%basis
    print >>outfile, '\n%s\n\n0 1\n\n$NBO NBO AONBO FNBO RESONANCE $END\n\n'%name
    outfile.close()
    
        
    

def GetBasis(gaussResult,quiet=False):
    """Get PyQuante representation of a basis from a gaussian output file"""
    
    #Create representation of molecule within PyQuante
    PQMol = cclib.bridge.makepyquante( gaussResult.atomcoords[-1],
                                       gaussResult.atomnos )

    #Get PyQuante representation of basis set
    basis=Ints.getbasis( PQMol , gaussResult.basisname )

    #Check that PyQuante and g09 have the same basis set ordering
    nbasis=gaussResult.nbasis
    assert len(basis.bfs) == nbasis,'Gaussian and PyQuante have '\
           'different basis sets. Did you specify the same basis for each?'
    overlap_py= np.array(Ints.getS(basis))
    maxdefect=0.0
    if hasattr(gaussResult,'mocoeffs_sao'):
        sao=gaussResult.mocoeffs_sao
    else:
        sao=gaussResult.aooverlaps

    if not quiet:
        for i,vals in enumerate(zip(overlap_py.flat,
                                    sao.flat)):
            pq,g9=vals
            x,y=np.unravel_index(i,(nbasis,nbasis))
            denom=max(pq,g9)
            if denom<10**-13: continue
            if min(pq,g9)==0: denom=1.0
            defect=abs((pq-g9)/denom)
            if defect>maxdefect:
                maxdefect=defect
            if defect > 1e-3 and x<=y: print pq,g9,(x,y),100*abs(pq-g9)/denom

        print 'Maximum error between G09 and PyQuante overlap matrices:',\
              maxdefect
        if maxdefect>1e-3:
            print 'WARNING!!!! Calculated overlap matrix does not match basis set!'
            print 'Basis sets might not match!\n\n\n'


    return nbasis,basis
 





def IterProg( iter , interval=0.05,tot=None):
    try:
        if tot is None:
            tot=len(iter)
        if int(tot*interval)<=0:
            interval=1.0/tot
        haslen=True
    except TypeError:
        haslen=False
        if interval<1: interval=100
            
    for i,item in enumerate(iter):
        yield item
        if haslen:
            if i % int(tot*interval)==0:
                print str( (100*i)/tot ) + '% ',
                sys.stdout.flush()
        else:
            if i%interval==0:
                print '\rProgress:',i,
                sys.stdout.flush()
    print 'done'
    raise StopIteration

