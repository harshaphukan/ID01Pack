#!/usr/bin/env python
#Script to open and parse Gvector File

import numpy as np
    
def GVEParse(fid,GVStr):

  header=0

  flagStr='#  gx  gy  gz  xc  yc  ds  eta  omega  spot3d_id  xl  yl  zl\r\n'
  K=0
  while K!=flagStr:
    K=fid.readline()
    header=header+1

  fid.close()   

  PeakData=np.genfromtxt(GVStr,skip_header=header)
  
  return PeakData

def main(fgve,GVStr):
  
  return GVEParse(fgve,GVStr)

if __name__=="__main__":
  main(fgve,GVStr)
