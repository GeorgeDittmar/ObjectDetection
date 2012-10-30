#!/usr/bin/python
import numpy as np
import math
"""
 calculate a gaussian value given a x,y point from -r to r
"""
def calc_gauss(x,y,r):
  pi = math.pi
  sigma = 1.2
  scale = 2*math.pi*((sigma*r)**2)
  gauss = (float(1)/float(2*pi*math.pow(sigma*r,2)))*math.exp(-(float(math.pow(x,2)+math.pow(y,2))/float(2*math.pow(sigma*r,2))))
  return 1-(gauss*scale)

"""
build the neighborhood suppression matrix such that it is the size of the global max bounding box.


find the "radius" of that box, 1/2 the width, and fill a NxN matrix with the values of 1-calc_gause(x,y). These will be our suppression values!
"""

def build_neighborhood(bbox):
  box_length = (bbox[2]-bbox[0])
  print box_length
  #box_length = 344 + int(344*.35)
  print box_length
  r = box_length/2
  print r
  n = 2*r+1

  # fill an nxn matrix with zeros
  neighborhood = np.zeros(shape=(n,n))

  for y in range(-(r),(r+1)):
    for x in range(-(r),(r+1)):
     y_ind = y + r
     x_ind = x + r

     gauss_val = calc_gauss(x,y,r)


     neighborhood[y_ind][x_ind] = gauss_val

  return neighborhood

"""
Helper function to find the center of a bounding box for neighborhood suppression
"""
def find_center(bbox):

  w = bbox[0]+bbox[2]
  h = bbox[1]+bbox[3]
  xc = w/2
  yc = h/2

  return xc,yc

print build_neighborhood((0,0,5,5))

