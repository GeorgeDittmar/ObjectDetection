#!/usr/bin/python

from glimpse.glab import *
from glimpse import util
from glimpse.util import svm
import numpy as np
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
  SetLayer('c2')
  SetParams(params)
  
  number_files = len(os.listdir(pos_image_dir))
  n = float(number_files)*training_split
  pos_images = GetDirContents(pos_image_dir,int(n))
  neg_images = GetDirContents(neg_image_dir,int(n))
  classes = map(os.path.basename, (pos_image_dir, neg_image_dir))
  train_split = pos_images, neg_images
  train_size = len(train_split[0])+len(train_split[1])
  test_split = ([], [])  # no test data
  test_size = len(test_split[0])+len(test_split[1])

  SetTrainTestSplit(train_split, test_split, classes)
  ImprintS2Prototypes(1000)
  exp = GetExperiment()
  # concat c1 features with c2 features
  model = exp.model
  print model
  images = util.UngroupLists(exp.train_images+exp.test_images)
  states = [model.MakeStateFromFilename(fn) for fn in images ]
  output_layer = model.LayerClass.C2
  
  print "building states"
  output_states = [model.BuildLayer(output_layer,state,save_all=True) for state in states]
  print "flattening features"
  print len(util.FlattenArrays(output_states[0]['c1']))+len(util.FlattenArrays(output_states[0]['c2']))
  features = [np.concatenate((util.FlattenArrays(st['c1']),util.FlattenArrays(st['c2']))) for st in output_states ]
  
  # split features up by training and testing
  train_features,test_features = util.SplitList(features,[train_size,test_size])
  # split features up by class
  
  train_features = util.SplitList(train_features,[len(pos_images),len(neg_images)])
  print len(train_features[0])
  
  exp.train_features = [ np.array(f, util.ACTIVATION_DTYPE)
        for f in train_features ]
  
  print "training"
  exp.TrainSvm()
  print "done training"
  
  exp.Store(os.path.join(xform_dir, 'exp.dat'))
if __name__ == '__main__':
  if len(sys.argv) < 5:
    sys.exit("usage: %s XFORM-DIR POS-DIR NEG-DIR TRAINING-SPLIT(percent)" % sys.argv[0])
  xform_dir, pos, neg, training_split = sys.argv[1:5]
  main(xform_dir, pos, neg, float(training_split))
