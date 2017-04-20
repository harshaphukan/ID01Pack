#!/usr/bin/env python
#Script to Calculate Centers-of-Mass, Strain and Stress Tensors from Cp Ti Beamline 1 Data
# This code uses the Least Squares Algorithm proposed by Marguiles et al to calculate the Centers of Mass
# and elastic Strain tensor values
#http://doi.org.proxy2.cl.msu.edu/10.1016/S1359-6454(02)00028-9
# Based on the MatLab Code suite developed by Armand Beaudoin and Leyun Wang

import numpy as np
import argparse
import matplotlib.pyplot as plt
from numpy import linalg as La
from numpy import deg2rad as d2r
from numpy import sin as sin
from numpy import cos as cos
from numpy import tan as tan


import itertools
import sys,os
from ID01Pack import GHCP, parseGrainSpotter,loadGve

# Lattice parameters for hexagonal Ti
a,c=2.95,4.683 # nm

CC=GHCP.GHCP(a,c)

Dict_hkl=CC.DictInitialize()

g0Cry,g_idel=CC.GHKL(Dict_hkl)

parser=argparse.ArgumentParser()
parser.add_argument("--log",help=" Enter the .log filename ")
parser.add_argument("--gve",help=" Enter the .gve filename ")
args=parser.parse_args()

try:
  flog=open(args.log,"r")
except IOError: 
  sys.exit("Did not find specified .log file")  

LogStr=args.log 
try:
  fgve=open(args.gve,"r")
except IDError:
  sys.exit(" Did not find specified .gve file ")  
 
GVEStr=args.gve

GvExMeasOM,PosDict,UDict,UBDict,DictRdz,EulerDict,GvecTable=parseGrainSpotter.Parser(flog)
PeakData=loadGve.GVEParse(fgve,GVEStr)

# Extract Image number from .log file

S=LogStr.split('_')
ImageNumber=int(S[1])
ScanNo=ImageNumber*np.ones((len(EulerDict),))

## twin plane and twin direction in the crystal coordinate system for the six T1 twin varients  

T1twin=np.array([[0.5860,0.3380,0.7370,-0.6380,-0.3680,0.6760],\
                 [0.,0.6760,0.7370,0.,-0.7370,0.6760],\
                 [-0.5860,0.3380,0.7370,0.6380,-0.3680,0.6760],\
                 [-0.5860,-0.3380,0.7370,0.6380,0.3680,0.6760],\
                 [0.,-0.6760,0.7370,0.,0.7370,0.6760],\
                 [0.5860,-0.3380,0.7370,-0.6380,0.3680,0.6760]])
                 
g_ideal=g_idel
grainNo=[x+1 for x in range(len(EulerDict))]
igrain=0



ThOmEt=np.zeros((100,3))#Initialize Array to hold Bragg(theta),Rotation(omega) & Azimuthal (Eta) angle values
hklS=np.zeros((100,3))#Initialize Array to hold hkl values
strn={}
stddevs={}

igrain=0
for ig in range(len(grainNo)):

  igrain+=1
  NgVec=len(GvecTable[ig+1][:,0])
  gveID=np.zeros((NgVec,))
  count=0
  
  for i in range(NgVec):
    if GvecTable[ig+1][i,-1]<2.0: # Threshold for internal angle
      count+=1
      gveID[i]=np.where(PeakData[:,8]==GvecTable[ig+1][i,2])[0][0]
      ThOmEt[count,0]=GvecTable[ig+1][i,12]
      ThOmEt[count,1]=GvecTable[ig+1][i,15]
      ThOmEt[count,2]=GvecTable[ig+1][i,18]
      hklS[count,:] = GvecTable[ig+1][i,3:6]
  NgVec=count
  gveID=gveID[~(gveID==0)]
  gve=np.concatenate((PeakData[gveID.astype(int),:3],\
                      PeakData[gveID.astype(int),5].reshape(len(gveID),1)),axis=1)
    

  ## Check to ensure that there are no existing Freidel pairs
  FriedelPairs=np.zeros((NgVec,1))
  for i in range(NgVec):
    for j in range(NgVec):
      ThklS=hklS[~(hklS==0).all(1)]
      if i!=j:
        if (hklS[i,:3]==-hklS[j,:3]).all():
          FriedelPairs[i]=1
          FriedelPairs[j]=1
  OutSide=np.zeros((NgVec,1))
  for i in range(NgVec):
    if La.norm(gve[i,:3])>0.92 and La.norm(gve[i,:3])<0.95:
      OutSide[i]=1
    else:
      OutSide[i]=0
      
  idx=np.where(OutSide!=1)[0]
  NgVec=len(idx)
  gve=gve[idx,:]
  
  B=np.zeros((NgVec,1))
  A=np.zeros((NgVec,8))
  
  for i in range(NgVec):
    gs=La.norm(gve[i,:3])-g_ideal
    g=np.abs(gs)
    midx=np.where(g==np.min(g))[0][0]
    val=np.min(g)
    B[i,0]=-gs[midx]/g_ideal[midx]
  for i in range(NgVec):
    TOE=ThOmEt[~(ThOmEt==0).all(1)]
    lmn=gve[i,:3]/La.norm(gve[i,:3]) # g-vector [Sample coordinate system!]
    dx_term = -( cos(d2r(TOE[i,1])) + (sin(d2r(TOE[i,1]))*sin(d2r(TOE[i,2]))\
                /tan(d2r(TOE[i,0]))))/999654.8 # new parameter
    dy_term = -( sin(d2r(TOE[i,1])) + (cos(d2r(TOE[i,1]))*sin(d2r(TOE[i,2]))\
               /tan(d2r(TOE[i,0]))))/999654.8 # new parameter from Ti_1516.par
    
    A[i,0],A[i,1],A[i,2],A[i,3]=lmn[0]**2,lmn[1]**2,lmn[2]**2,2*lmn[0]*lmn[1]
    A[i,4],A[i,5],A[i,6],A[i,7]=2*lmn[0]*lmn[2],2*lmn[1]*lmn[2],-dx_term,-dy_term
  
  X=La.lstsq(A,B)[0]
  fit_error=B-np.dot(A,X)
  strn[igrain]=La.lstsq(A,B)[0]
  stddevs[igrain]=np.std(fit_error)
  idx=np.where(np.abs(fit_error)<0.004)[0]
  A1=A[idx,:]
  B1=B[idx]
  strn[igrain]=La.lstsq(A1,B1)[0]
  stddevs[igrain]=np.std(B1-np.dot(A1,strn[igrain]))

Rowstrn={} # Initialize Dictionary to store 6 components of elastic strain tensor and Center of mass for each grain!
Strain={} #  Initialize Dictionary to store symmetric (9 comp) strain tensor for each grain identified![Smp Coord Sys]
Strain_c={} # Initialize Dictionary to store strain in XTal Coord system
Stress_c={}
Stress_Smp={}
for i in range(1,igrain+1):
  Rowstrn[i]=strn[i].T
  Strain[i]=np.array([[strn[i][0,0],strn[i][3,0],strn[i][4,0]],\
                      [strn[i][3,0],strn[i][1,0],strn[i][5,0]],\
                      [strn[i][4,0],strn[i][5,0],strn[i][2,0]]])
  # Rotate Orientation Matrix by 90 degress
  UDict[i]=np.dot(UDict[i],np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])) 
   # Rotate Strain Tensor to XTal Coordinate system
  Strain_c[i]=np.dot(UDict[i].T,Strain[i])
  Strain_c[i]=np.dot(Strain_c[i],UDict[i])
  # Stiffness Tensor
  C=np.array([[162.4e3,92.0e3,69.0e3,0.,0.,0.],\
             [92.0e3,162.2e3,69.0e3,0.,0.,0.],\
              [69.0e3,69.0e3,181.6e3,0.,0.,0.],\
              [0.,0.,0.,47.2e3,0.,0.],\
              [0.,0.,0.,0.,47.2e3,0.],\
              [0.,0.,0.,0.,0.,35.2e3]])
  strain_c_vec=np.array([Strain_c[i][0,0],Strain_c[i][1,1],Strain_c[i][2,2],\
                         Strain_c[i][1,2]*2.0,Strain_c[i][2,0]*2,Strain_c[i][0,1]*2.0]).reshape(6,1)
  # Use generalized Hooke's Law to Evaluate Stress Tensor in XTal Coordinate System
  Stress_c_vec=np.dot(C,strain_c_vec)
  Stress_c[i]=np.array([[Stress_c_vec[0],Stress_c_vec[5],Stress_c_vec[4]],\
                        [Stress_c_vec[5],Stress_c_vec[1],Stress_c_vec[3]],\
                        [Stress_c_vec[4],Stress_c_vec[3],Stress_c_vec[2]]]).reshape(3,3)
                        
  ## Transform Stress Tensor to Sample Coordinate System
  Stress_Smp[i]=np.dot(UDict[i],Stress_c[i])
  Stress_Smp[i]=np.dot(Stress_Smp[i],UDict[i].T)
  
## Plotting the Spatial Centers of Mass Map
X=np.zeros((len(grainNo),1))
Y=np.zeros((len(grainNo),1))
XY30=np.zeros((len(grainNo),2))
theta=d2r(20) # Rotate COM Positions by 25 degrees CCW in the XY Plane
# Define Orientation Matrix in 2D
RotMat=np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])


for i in range(len(grainNo)):
  X[i]=Rowstrn[grainNo[i]][0,6]
  Y[i]=Rowstrn[grainNo[i]][0,7]

  XY30[i,:]=np.dot(RotMat,np.array([X[i,0],Y[i,0]]).T)

fig1,ax1=plt.subplots()

ax1.plot(X,Y,'ro',markersize=9)
offset=50.0
ax1.set_xlim(np.min(X)-offset,np.max(X)+offset)
ax1.set_ylim(np.min(Y)-offset,np.max(Y)+offset)

for id in range(len(grainNo)):
  ax1.annotate(str(grainNo[id]),xy=(X[id,0]+3,Y[id,0]+3),size=8)


figure2,ax2=plt.subplots()
ax2.plot(XY30[:,0],XY30[:,1],'bo',markersize=9)
offset=50.0
ax2.set_xlim(np.min(X)-offset,np.max(X)+offset)
ax2.set_ylim(np.min(Y)-offset,np.max(Y)+offset)

for id in range(len(grainNo)):
  ax2.annotate(str(grainNo[id]),xy=(XY30[id,0]+3,XY30[id,1]+3),size=8)
plt.show()
  
    
  
  

  
    
  
  
  

  
  
  

    
    
    



  