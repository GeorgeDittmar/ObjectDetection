#!/usr/bin/python

from glimpse.glab import *
#~ from glimpse.models import viz2
import os
import sys
import pickle
import argparse
def GetDirContents(dir_path,n):
  #~ print n
  contents = os.listdir(dir_path)
  subset = list()
  for i in range(0,n):
    subset.append(os.path.join(dir_path,contents[i]))
  return subset
#def GetDirContents(dir_path):
#  return [ os.path.join(dir_path, f) for f in os.listdir(dir_path) ]

#~ def main(xform_dir, pos_image_dir, neg_image_dir):
def main(xform_di,prototype_dir,pos_image_dir,neg_image_dir,training_split):
  
  from glimpse.models import viz2
  params = viz2.Params()
  params.num_scales = 4
  model = SetModelClass(viz2.Model)
  SetLayer('c2')
  SetParams(params)
  #SetModelClass(viz2.Model)

  number_files = len(os.listdir(pos_image_dir))
  n = float(number_files)*training_split
  print n
  # OLD CODE
  pos_images = GetDirContents(pos_image_dir,int(n))
  neg_images = GetDirContents(neg_image_dir,int(n))
  classes = map(os.path.basename, (pos_image_dir, neg_image_dir))
  train_split = pos_images, neg_images
  test_split = ([], [])  # no test data

  # NEW CODE  thats now old again
  #images = map(GetDirContents, image_dirs)
  #classes = map(os.path.basename, image_dirs)
  #train_split = images
  #test_split = [ [] for x in classes ]  # no test data

  SetTrainTestSplit(train_split, test_split, classes)
  print "Learning Imprinted Prototypes"
  ImprintS2Prototypes(4000)
  print "training"
  TrainSvm()
  print "done training"
  
  exp_path = os.path.join(xform_dir, 'exp.dat')
  StoreExperiment(exp_path)

  
  
  

if __name__ == '__main__':
  print "No"
  if len(sys.argv) < 6:
    print "Yes"
    """  
    sys.exit("usage: %s XFORM-DIR POS-DIR NEG-DIR" % sys.argv[0])
  xform_dir, pos_image_dir, neg_image_dir = sys.argv[1:4]
  main(xform_dir, pos_image_dir, neg_image_dir)
    """
    sys.exit("usage: %s XFORM-DIR PROTO-DIR POS-DIR NEG-DIR TRAINING-SPLIT(percent)" % sys.argv[0])
  xform_dir, proto_dir, pos,neg, training_split= sys.argv[1:6]
  main(xform_dir,proto_dir,pos,neg,float(training_split))
