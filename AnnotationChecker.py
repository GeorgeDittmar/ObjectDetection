import os
import Image
import ImageDraw
import sys
import csv


def getGroundTruths(image,test_dir,anno_dir,obj_type):
  image = os.path.splitext(image)[0]
  ending = '.labl'
  labelfile = os.path.join(anno_dir,image+ending)
  print labelfile
  annotation = open(labelfile,'rb')
  annoReader = csv.reader(annotation,delimiter='|')
  groundTruth = list()
  
  
  for x in annoReader:
    
    carInd =  0
    
    for j in range(0,len(x)):
      if x[j] == obj_type:
        carInd = j
        break

    length = len(x)
    numBoxes = int(x[2])
    # grab all annotations for the object
    
    for i in range(3,(numBoxes*4),4):
      if i > carInd:
        break
        
      x1 = int(x[i])
      y1 = int(x[i+1])
      x2 = x1+int(x[i+2])
      y2 = y1+int(x[i+3])
      bbox = (x1,y1,x2,y2)
      
      groundTruth.append(bbox)
      
  print len(groundTruth)
  return groundTruth
  # loop through all the label files
    # read the label with csv reader
        # loop through all the xy pairs, or just index them

def getTruthDict(test_set,test_dir,anno_dir,obj_type):
  ending = '.labl'
  truthDict = {}
  
  for image in test_set:
    truthDict[image] = getGroundTruths(image,test_dir,anno_dir,obj_type)
  
  return truthDict
