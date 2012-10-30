#!/usr/bin/python

from glimpse.glab import *
import os
import sys
import pickle

def GetDirContents(dir_path,n):
  contents = os.listdir(dir_path)
  subset = list()
  for i in range(0,n):
    subset.append(os.path.join(dir_path,contents[i]))
  return subset

#~ def main(xform_dir, pos_image_dir, neg_image_dir):
def main(xform_di, pos_image_dir, neg_image_dir, training_split):
  from glimpse.models import viz2
  params = viz2.Params()
  params.num_scales = 4
  SetModelClass(viz2.Model)
  SetLayer('c1')
  SetParams(params)

  number_files = len(os.listdir(pos_image_dir))
  n = float(number_files)*training_split
  pos_images = GetDirContents(pos_image_dir,int(n))
  neg_images = GetDirContents(neg_image_dir,int(n))
  print len(pos_images)
  classes = map(os.path.basename, (pos_image_dir, neg_image_dir))
  train_split = pos_images, neg_images
  test_split = ([], [])  # no test data

  SetTrainTestSplit(train_split, test_split, classes)
  print "training"
  TrainSvm()
  print "done training"
  

  StoreExperiment(os.path.join(xform_dir, 'exp.dat'))

if __name__ == '__main__':
  if len(sys.argv) < 5:
    sys.exit("usage: %s XFORM-DIR POS-DIR NEG-DIR TRAINING-SPLIT(percent)" % sys.argv[0])
  xform_dir, pos, neg, training_split = sys.argv[1:5]
  main(xform_dir, pos, neg, float(training_split))
