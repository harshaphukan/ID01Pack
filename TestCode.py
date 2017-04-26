#!/usr/bin/env python

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
from ID01Pack import GHCP, parseGrainSpotter,loadGve, ID01StressStrainModule
import ID01StressStrainTest

# Lattice parameters for hexagonal Ti
a,c=2.95,4.683 # nm

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
# Lattice parameters for hexagonal Ti
a,c=2.95,4.683 # nm
PeakData,GvExMeasOM,PosDict,UDict,UBDict,DictRdz,EulerDict,GvecTable,ScanNo=ID01StressStrainTest.FileParse\
                                                                            (flog,fgve,LogStr,GVEStr)
Rowstrn,Strain,Strain_c,Stress_Smp,Stress_c=ID01StressStrainTest.StressStrain(GvecTable,PeakData,UDict,a,c)

SigC=np.zeros((len(UDict),6))
SigG=np.zeros((len(UDict),6))
Theta=np.zeros((len(UDict),1))

for i in range(len(SigC)):
  SigC[i,0],SigC[i,1],SigC[i,2]=Stress_c[i+1][0,0],Stress_c[i+1][1,1],Stress_c[i+1][2,2]
  SigC[i,3],SigC[i,4],SigC[i,5]=Stress_c[i+1][0,1],Stress_c[i+1][1,2],Stress_c[i+1][0,2]
  
  SigG[i,0],SigG[i,1],SigG[i,2]=Stress_Smp[i+1][0,0],Stress_Smp[i+1][1,1],Stress_Smp[i+1][2,2]
  SigG[i,3],SigG[i,4],SigG[i,5]=Stress_Smp[i+1][0,1],Stress_Smp[i+1][1,2],Stress_Smp[i+1][0,2]
  
  SigC[i,:]=SigC[i,:]/La.norm(SigC[i,:])
  SigG[i,:]=SigG[i,:]/La.norm(SigG[i,:])
  Theta[i]=np.arccos(np.dot(SigC[i,:],SigG[i,:].T))
  

  

  

  
