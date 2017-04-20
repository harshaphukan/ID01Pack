#!/usr/bin/env python 

#!/usr/bin/env python
#Script to Calculate Centers-of-Mass, Strain and Stress Tensors from Beamline 1 Data

import numpy as np
from numpy import linalg as La
import itertools
import sys,os

class GHCP:
  """ Class to create hkls based on entered lattice parameters. """

  def __init__(self,a,c):
    self.a=a
    self.c=c
  
  def DictInitialize(self):

    dict={}
    dict[1]=np.array([[1,0,-1,0],[0,1,-1,0],[1,-1,0,0]])
    dict[2]=np.array([[0,0,0,2]])
    dict[3]=np.array([[1,0,-1,1],[0,1,-1,1],[1,-1,0,1],[1,0,-1,-1],\
                   [0,1,-1,-1],[1,-1,0,-1]])
    dict[4]=np.array([[1,0,-1,2],[0,1,-1,2],[1,-1,0,2],[1,0,-1,-2],\
                   [0,1,-1,-2],[1,-1,0,-2]])   
    dict[5]=np.array([[2,-1,-1,0],[-1,2,-1,0],[-1,-1,2,0]])    
    dict[6]=np.array([[1,0,-1,3],[0,1,-1,3],[1,-1,0,3],[1,0,-1,-3],[0,1,-1,-3],[1,-1,0,-3]])   
    dict[7]=np.array([[2,0,-2,0],[0,2,-2,0],[2,-2,0,0]])   
    dict[8]=np.array([[2,-1,-1,2],[2,-1,-1,-2],[-1,2,-1,2],[-1,2,-1,-2],[-1,-1,2,2],[-1,-1,2,-2]])  
    dict[9]=np.array([[2,0,-2,1],[0,2,-2,1],[2,-2,0,1],[2,0,-2,-1],[0,2,-2,-1],[2,-2,0,-1]]) 
    dict[10]=np.array([[0,0,0,4]]) 
    dict[11]=np.array([[2,0,-2,2],[0,2,-2,2],[2,-2,0,2],[2,0,-2,-2],[0,2,-2,-2],[2,-2,0.,-2]])  
    dict[12]=np.array([[1,0,-1,4],[0,1,-1,4],[1,-1,0,4],[1,0,-1,-4],[0,1,-1,-4],[1,-1,0,-4]])  
    dict[13]=np.array([[2,0,-2,3],[0,2,-2,3],[2,-2,0,3],[2,0,-2,-3],[0,2,-2,-3],[2,-2,0,-3]])  
    dict[14]=np.array([[2,1,-3,0],[2,-3,1,0],[-3,2,1,0],[1,2,-3,0],[1,-3,2,0],[-3,1,2,0]]) 
    dict[15]=np.array([[2,1,-3,1],[2,-3,1,1],[-3,2,1,1],[1,2,-3,1],[1,-3,2,1],[-3,1,2,1],\
                    [2,1,-3,-1],[2,-3,1,-1],[-3,2,1,-1],[1,2,-3,-1],[1,-3,2,-1],[-3,1,2,-1]]) 
    dict[16]=np.array([[2,-1,-1,4],[2,-1,-1,-4],[-1,2,-1,4],[-1,2,-1,-4],[-1,-1,2,4],[-1,-1,2,-4]]) 
    return dict                        
  
  def GHKL(self,dict):
    a0,c0=self.a,self.c
    b0=a0               
    alpha,beta,gamma=np.deg2rad(90),np.deg2rad(90),np.deg2rad(120)
    As=np.arccos((np.cos(beta)*np.cos(gamma)-np.cos(alpha))/np.sin(beta)/np.sin(gamma))
    A0=np.array([[a0,b0*np.cos(gamma),c0*np.cos(beta)],\
               [0,b0*np.sin(gamma),-c0*np.sin(beta)*np.cos(As)],\
               [0,0,c0*np.sin(beta)*np.sin(As)]])
    hkil=[]
  
    for i in range(len(dict)):
      hkil.append(dict[i+1])
      hkil.append((-1)*dict[i+1])
   
    h,k,l=[],[],[]
  
    for i in range(len(hkil)):
      h.append(hkil[i][:,0])
      k.append(hkil[i][:,1])
      l.append(hkil[i][:,3])
      h[i]=h[i].tolist()
      k[i]=k[i].tolist()
      l[i]=l[i].tolist()
  
    h=list(itertools.chain.from_iterable(h))
    k=list(itertools.chain.from_iterable(k))
    l=list(itertools.chain.from_iterable(l))
  
    hkl=np.zeros((len(h),3))
  
    for i in range(len(hkl)):
      hkl[:,0]=h
      hkl[:,1]=k
      hkl[:,2]=l
  
    hkl=hkl.T
  
    g0Cry=La.solve(A0.T,hkl)
  
    g_idl=np.zeros((len(dict),1))
  
    lastlen=0
  
    idx=0
  
    for i in range(len(g0Cry.T)):
      gl=La.norm(g0Cry[:,i])
      if np.abs(gl-lastlen)>0.0001:
        g_idl[idx]=gl
        idx+=1
        lastlen=gl
  
    return g0Cry, g_idl
