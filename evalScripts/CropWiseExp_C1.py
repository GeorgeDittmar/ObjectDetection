#!/usr/bin/python
from glimpse.glab import *
from glimpse.util import stats
#import trainer
#import validator
import os
import sys
import random as rand
import numpy as np
import matplotlib
import Image
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from sklearn.metrics import roc_curve, auc, confusion_matrix
"""
Given the positive and negative test data, make the ROC curve.
"""
def validate(pos,neg):
  pass


def c_matrix(truth,predicted):
    posCount = 0
    negCount = 0
    predPos = 0
    predNeg = 0

    for i in range(0,len(truth)):
      if truth[i] == 1 :
        posCount += 1
      elif truth[i] == -1:
        negCount += 1


    print posCount,negCount
    
# remove any images from the lists that are too small to process
def  check_img(pos,neg,pos_path,neg_path):
  print len(pos),len(neg)
  
  for p in pos:
    tmpImg = Image.open(os.path.join(pos_path,p))
    w,h = tmpImg.size
    if (w < 50 and h > 50) or (w > 50 and h < 50) or ( w < 50 or h < 50):
      print "removing"
      pos.remove(p)
  
  for n in neg:
    tmpImg = Image.open(os.path.join(neg_path,n))
    w,h = tmpImg.size
    if (w < 50 and h > 50) or (w > 50 and h < 50) or ( w < 50 or h < 50):
      print "Removing"
      neg.remove(n)
  print "Left over:",len(pos),len(neg)

def crop_exp(training_pos,trainig_neg):
  from glimpse.models import viz2
  params = viz2.Params()
  params.num_scales = 4
  SetModelClass(viz2.Model)
  SetLayer('c1')
  SetParams(params)

  # get all the contents of the directories
  pos = os.listdir(training_pos)
  neg = os.listdir(training_neg)
  check_img(pos,neg,training_pos,training_neg)
  
  total = len(pos)+len(neg)
  print "Total", total

  train_size =  int(len(pos)*(float(2)/3))
  #train_size = 1 * len(pos)
  training_splits_pos = list()
  training_splits_neg = list()

  test_split_pos = list()
  test_split_neg = list()

  test_size = total - (train_size*2)
  print "Train, Test",train_size,test_size

  # get 2/3 of the training set data
  for n in range(0,train_size):
    example = rand.randint(0,len(pos)-1)
    
    postemp = pos.pop(example)
    negtemp = neg.pop(example)
    #postemp = pos[n]
    #negtemp = neg[n]
    training_splits_pos.append(os.path.join(training_pos,postemp))
    training_splits_neg.append(os.path.join(training_neg,negtemp))
  if len(pos) == 0:
    test_split_pos = training_splits_pos
  if len(neg) == 0:
    test_split_neg = training_splits_neg
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
  #ImprintS2Prototypes(1000)

  print "training"
  TrainSvm()



  print "Testing"
  accuracy,results_dict = TestSvm()
  print results_dict

  decision_vals = results_dict['decision_values']
  tp_label = np.zeros(len(test_split_pos))
  tp_label.fill(1)
  tn_label = np.zeros(len(test_split_neg))
  tn_label.fill(-1)
  truth_labels = np.concatenate((tp_label,tn_label))
  fps,tps,thresholds = roc_curve(truth_labels,decision_vals)
  cm = confusion_matrix(truth_labels,results_dict['predicted_labels'])
  print cm
  c_matrix(truth_labels,results_dict['predicted_labels'])
  Reset()
  return (fps,tps,auc(fps,tps),accuracy,cm)

def main(training_pos,training_neg,n,obj_type):
  save_res = open(os.path.join('results',obj_type+'C1ROC.txt'),'wb') 
  # main loop of execution for the validation code
  accuracies_c1 = list()
  for i in range(0,n):
    accuracies_c1.append(crop_exp(training_pos,training_neg))
  print [x[2] for x in accuracies_c1]
  mean_auc = 0.0
  auc_list = list()
  for i in range(0, len(accuracies_c1)):
    save_res.write('AUC of Run '+str(i)+' : '+str(accuracies_c1[i][2])+'\n')
    save_res.write('Accuracy of Run '+str(i)+' : '+str(accuracies_c1[i][3])+'\n')
    save_res.write('Confusion Matrix of Run '+str(i)+' : '+str(accuracies_c1[i][4])+'\n')
    mean_auc += accuracies_c1[i][2]
    auc_list.append(accuracies_c1[i][2])
    pl.plot(accuracies_c1[i][0],accuracies_c1[i][1],label="Run "+str(i)+"")

  pl.xlabel("False Positive Rate")
  pl.ylabel("True Positive Rate")
  pl.title("ROC For "+obj_type+" C1")
  pl.legend(loc="lower right")
  pl.savefig(os.path.join('results',obj_type+'c1roc'))
  print "AVG AUC", float(mean_auc)/len(accuracies_c1)
  save_res.write("Mean AUC: "+str(float(mean_auc)/len(accuracies_c1)))
  
  auc_array = np.array(auc_list)
  save_res.write("STD : "+str(np.std(auc_array)))


if __name__ == '__main__':
  if len(sys.argv) < 4:
    #sys.exit("usage: %s XFORM-DIR IMAGE WINDOW-STEP-SIZE THRESHOLD OBJ-Type" % sys.argv[0])
    sys.exit("usage: %s Pos-Dir Neg-Dir Num-folds Obj-Type" % sys.argv[0])
  training_pos,training_neg,n,obj_type = sys.argv[1:5]
  main(training_pos,training_neg,int(n),obj_type)
