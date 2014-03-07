#!/usr/bin/python
from glimpse.glab import *
from glimpse.util import stats
import trainer
import validator
import os
import sys
import random as rand
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from sklearn.metrics import roc_curve, auc
import argparse
"""
Given the positive and negative test data, make the ROC curve.
"""
def validate(pos,neg):
  pass

def crop_exp(training_pos,trainig_neg):
  SetExperiment(layer = 'c1')

  # get all the contents of the directories
  pos = os.listdir(training_pos)
  neg = os.listdir(training_neg)
  total = len(pos)+len(neg)
  print "Total", total

  train_size = int(len(pos)*(float(2)/3))
  training_splits_pos = list()
  training_splits_neg = list()

  test_split_pos = list()
  test_split_neg = list()

  test_size = total - train_size
  print "Train, Test",train_size,test_size

  # get 2/3 of the training set data
  for n in range(0,train_size):
    example = rand.randint(0,len(pos)-1)

    postemp = pos.pop(example)
    negtemp = neg.pop(example)
    training_splits_pos.append(os.path.join(training_pos,postemp))
    training_splits_neg.append(os.path.join(training_neg,negtemp))
  for p in pos:
    test_split_pos.append(os.path.join(training_pos,p))
  for n in neg:
    test_split_neg.append(os.path.join(training_neg,n))



  print len(test_split_pos),len(test_split_neg)

  classes = map(os.path.basename, (training_pos, training_neg))
  train_split = training_splits_pos, training_splits_neg
  test_split = (test_split_pos, test_split_neg)

  # NEW CODE  thats now old again
  #images = map(GetDirContents, image_dirs)
  #classes = map(os.path.basename, image_dirs)
  #train_split = images
  #test_split = [ [] for x in classes ]  # no test data

  SetTrainTestSplit(train_split, test_split, classes)
  s = ImprintS2Prototypes(100)
  print s
  print "training"
  TrainSvm()
  accuracy,results_dict = TestSvm()
  predicted_labels = np.concatenate((results_dict['decision_values'][0],results_dict['decision_values'][1]))
  tp_label = np.zeros(len(test_split_pos))
  tp_label.fill(1)
  tn_label = np.zeros(len(test_split_neg))
  tn_label.fill(-1)
  truth_labels = np.concatenate((tp_label,tn_label))
  fps,tps,thresholds = roc_curve(truth_labels,predicted_labels)


  return (fps,tps,auc(fps,tps),accuracy)

def main(training_pos,training_neg,n):

  # main loop of execution for the validation code
  accuracies_c1 = list()
  for i in range(0,n):
    accuracies_c1.append(crop_exp(training_pos,training_neg))

  mean_auc = 0.0
  auc_list = list()
  for i in range(0, len(accuracies_c1)):
    mean_auc += accuracies_c1[i][2]
    auc_list.append(accuracies_c1[i][2])
    pl.plot(accuracies_c1[i][0],accuracies_c1[i][1],label="Run "+str(i)+"")

  pl.xlabel("False Positive Rate")
  pl.ylabel("True Positive Rate")
  pl.title("ROC for Car C1")
  pl.legend(loc="lower right")
  pl.savefig("car_plot_c1")
  print "AVG AUC", float(mean_auc)/len(accuracies_c1)
  auc_array = np.array(auc_list)
  print np.std(auc_array)


if __name__ == '__main__':
  if len(sys.argv) < 3:
    #sys.exit("usage: %s XFORM-DIR IMAGE WINDOW-STEP-SIZE THRESHOLD" % sys.argv[0])
    sys.exit("usage: %s Pos-Dir Neg-Dir N" % sys.argv[0])
  training_pos,training_neg,n = sys.argv[1:4]
  main(training_pos,training_neg,int(n))
