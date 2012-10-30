#!/usr/bin/python
from glimpse.glab import *
from glimpse.util import stats
from glimpse import util
#import trainer
#import validator
import os
import sys
import random as rand
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from sklearn.metrics import roc_curve, auc, confusion_matrix
import argparse
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

def crop_exp(training_pos,training_neg,num_proto,hist,uni,shuff,norm):
  from glimpse.models import viz2
  params = viz2.Params()
  params.num_scales = 4
  SetModelClass(viz2.Model)
  SetLayer('c2')
  SetParams(params)

  # get all the contents of the directories
  pos = os.listdir(training_pos)
  neg = os.listdir(training_neg)
  total = len(pos)+len(neg)
  print "Total", total

  train_size = int(len(pos)*(float(2)/3))
  #train_size = 1 * len(pos)
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


  print len(training_splits_pos),len(training_splits_neg)
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
  
  print "training"
  if uni:
    print "Learning Uniform Random"
    MakeUniformRandomS2Prototypes(num_proto)
  elif norm:
    print "Learning Normalized Random"
    MakeNormalRandomS2Prototypes(num_proto)
  elif shuff:
    print "Learning Shuffled Random"
    MakeShuffledRandomS2Prototypes(num_proto)
  elif hist:
    print "Learning Histogram Random"
    MakeHistogramRandomS2Prototypes(num_proto)
  else:
    print "Learning Imprinted Prototypes"
    ImprintS2Prototypes(num_proto)

  exp = GetExperiment()
  # concat c1 features with c2 features
  model = exp.model
  print model
  images = util.UngroupLists(exp.train_images+exp.test_images)
  states = [model.MakeStateFromFilename(fn) for fn in images ]
  output_layer = model.LayerClass.C2
  training_size = sum(map(len,exp.train_images)) 
  test_size = sum(map(len,exp.test_images))
  
  print "building states"
  builder = model.BuildLayerCallback(output_layer,save_all=True)
  #output_states = [model.BuildLayer(output_layer,state,save_all=True) for state in states]
  output_states = exp.pool.imap(builder,states)
  print "flattening features"
 # print len(util.FlattenArrays(output_states[0]['c1']))+len(util.FlattenArrays(output_states[0]['c2']))
  features = [np.concatenate((util.FlattenArrays(st['c1']),util.FlattenArrays(st['c2']))) for st in output_states ]
  
  # split features up by training and testing
  train_features,test_features = util.SplitList(features,[training_size,test_size])
  t = np.array(test_features)
  c = np.array(train_features)
  print "test feature shape,train", t.shape,c.shape
  # split features up by class
  train_features = util.SplitList(train_features,[len(training_splits_pos),len(training_splits_neg)])
  
  #split test features up by class
  test_features = util.SplitList(test_features,map(len,exp.test_images))
  

  
  exp.train_features = [ np.array(f, util.ACTIVATION_DTYPE) for f in train_features ]
  exp.test_features = [ np.array(f, util.ACTIVATION_DTYPE) for f in test_features ]
  print "training"
  exp.TrainSvm()
  print "done training"
  print "Testing"
  accuracy,results_dict = exp.TestSvm()

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

def testAll(training_pos,training_neg,num_proto,obj_type):
	label = 'All' + obj_type 
	save = 'C1C2 All' +obj_type+".txt"
	save_res = open(os.path.join('results',save),'wb')
	
	rocRes = list()
	# get all the contents of the directories
	for i in range(0,5):
	  from glimpse.models import viz2
	  params = viz2.Params()
	  params.num_scales = 4
	  SetModelClass(viz2.Model)
	  SetLayer('c2')
	  SetParams(params)
	  
	  print "STARTING RUN..."
	  pos = os.listdir(training_pos)
	  neg = os.listdir(training_neg)
	  total = len(pos)+len(neg)
	  print "Total", total
  
	  train_size = int(len(pos)*(float(2)/3))
	  #train_size = 1 * len(pos)
	  training_splits_pos = list()
	  training_splits_neg = list()
  
	  test_split_pos = list()
	  test_split_neg = list()
  
	  test_size = len(pos) - train_size
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
	  SetTrainTestSplit(train_split, test_split, classes)
	  print "training"
	  if i == 0:
		  ImprintS2Prototypes(num_proto)
	  elif i == 1:
		  print "Learning Uniform Random"
		  MakeUniformRandomS2Prototypes(num_proto)
	  elif i == 2:	
		  print "Learning Normalized Random"
		  MakeNormalRandomS2Prototypes(num_proto)
	  elif i == 3:
		  print "Learning Shuffled Random"
		  MakeShuffledRandomS2Prototypes(num_proto)
	  elif i==4:
		  print "Learning Histogram Random"
		  MakeHistogramRandomS2Prototypes(num_proto)
		  
	  exp = GetExperiment()
	  # concat c1 features with c2 features
	  model = exp.model
	  print model
	  images = util.UngroupLists(exp.train_images+exp.test_images)
	  states = [model.MakeStateFromFilename(fn) for fn in images ]
	  output_layer = model.LayerClass.C2
	  training_size = sum(map(len,exp.train_images)) 
	  test_size = sum(map(len,exp.test_images))
    
	  print "building states"
	  builder = model.BuildLayerCallback(output_layer,save_all=True)
	  #output_states = [model.BuildLayer(output_layer,state,save_all=True) for state in states]
	  output_states = exp.pool.imap(builder,states)
	  print "flattening features"
	  #print len(util.FlattenArrays(output_states[0]['c1']))+len(util.FlattenArrays(output_states[0]['c2']))
	  features = [np.concatenate((util.FlattenArrays(st['c1']),util.FlattenArrays(st['c2']))) for st in output_states ]
    
	  
	  
	  # split features up by training and testing
	  train_features,test_features = util.SplitList(features,[training_size,test_size])

	  # split features up by class
	  train_features = util.SplitList(train_features,[len(training_splits_pos),len(training_splits_neg)])
  
	  #split test features up by class
	  test_features = util.SplitList(test_features,map(len,exp.test_images))
    
	  exp.train_features = [ np.array(f, util.ACTIVATION_DTYPE) for f in train_features ]
	  exp.test_features = [ np.array(f, util.ACTIVATION_DTYPE) for f in test_features ]
	  print exp.test_features
	  print "training"
	  exp.TrainSvm()
	  print "done training"
	  print "Testing"
	  accuracy,results_dict = exp.TestSvm()
      
	  
	  decision_vals = results_dict['decision_values']
	  
	  save_res.write('accuracy at '+str(i)+' '+str(accuracy)+"\n")
	  tp_label = np.zeros(len(test_split_pos))
	  tp_label.fill(1)
	  tn_label = np.zeros(len(test_split_neg))
	  tn_label.fill(-1)
	  truth_labels = np.concatenate((tp_label,tn_label))
	  fps,tps,thresholds = roc_curve(truth_labels,decision_vals)
	  save_res.write('AUC at '+str(i)+' '+str(auc(fps,tps))+"\n")
	  rocRes.append((fps,tps,auc(fps,tps)))
	  Reset()

	# create plot and save to file
	for y in range(0,len(rocRes)):
	  if y == 0:
	    pl.plot(rocRes[y][0],rocRes[y][1],label="C1 + Imprinted")
	  elif y == 1:
		  
	    pl.plot(rocRes[y][0],rocRes[y][1],label="C1 + Uniform Random")
	  elif y == 2:	
		  
	    pl.plot(rocRes[y][0],rocRes[y][1],label="C1 + Normalized Random")
	  elif y == 3:

	    pl.plot(rocRes[y][0],rocRes[y][1],label="C1 + Shuffled Random")
	  elif y==4:
	    pl.plot(rocRes[y][0],rocRes[y][1],label="C1 + Histogram Random")
			
	caption = "Comparision of all C1C2 prototpying runs."
	pl.xlabel("False Positive Rate")
	pl.ylabel("True Positive Rate")
	pl.title(label)
	pl.legend(loc="lower right")
	pl.savefig(os.path.join('objRec',save))		
		
def main(training_pos,training_neg,num,numProto,h,u,s,n,obj_type):
  label = ''
  save = ''
  end = ''
  if h:
    end= obj_type+'_c1c2Histogram'
    label = 'ROC for '+ obj_type+' C1+C2 Histogram Prototypes'
  elif u:
    end= obj_type+'_c1c2Uniform'
    label ='ROC for '+obj_type+' C1+C2 Random Prototypes'
  elif s:
    end= obj_type + '_c1c2Shuffled'
    label = 'ROC for '+obj_type+' C1+C2 Shuffled Prototypes'
  elif n:
    end= obj_type+'_c1c2Normalized'
    label = 'ROC for '+obj_type+' C1+C2 Normalized Prototypes'
  else:
    end= obj_type+'_c1c2Imprinted'
    label = 'ROC for '+obj_type+' C1+C2 Imprinted Prototypes'

  save_res = open(os.path.join('ObjRec',end+'.txt'),'wb') 
  # main loop of execution for the validation code
  accuracies_c2 = list()
  for i in range(0,num):
    accuracies_c2.append(crop_exp(training_pos,training_neg,numProto,h,u,s,n))
    
  mean_auc = 0.0
  auc_list = list()
  for i in range(0, len(accuracies_c2)):
    save_res.write('AUC of Run '+str(i)+' : '+str(accuracies_c2[i][2])+'\n')
    save_res.write('Accuracy of Run '+str(i)+' : '+str(accuracies_c2[i][3])+'\n')
    save_res.write('Confusion Matrix of Run '+str(i)+' : '+str(accuracies_c2[i][4])+'\n')
    mean_auc += accuracies_c2[i][2]
    auc_list.append(accuracies_c2[i][2])
    pl.plot(accuracies_c2[i][0],accuracies_c2[i][1],label="Run "+str(i)+"")

  caption = "Comparision of all C1C2 prototpying runs."
  pl.xlabel("False Positive Rate")
  pl.ylabel("True Positive Rate")
  pl.title(label)
  #pl.text(.5,.5,text)
  pl.legend(loc="lower right")
  pl.savefig(os.path.join('ObjRec',end))
  save_res.write("Mean AUC: "+str(float(mean_auc)/len(accuracies_c2)))
  auc_array = np.array(auc_list)
  save_res.write("\nSTD: "+str(np.std(auc_array)))
  print np.std(auc_array)
  

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='C2 training script to allow training models using different prototype schemes.')
	parser.add_argument('posDir')
	parser.add_argument('negDir')
	parser.add_argument('numFolds',type=int)
	parser.add_argument('numProto',type=int)
	parser.add_argument('objType')
	parser.add_argument('-hist','--histogram',default=False,action='store_true',help='Build HMax model using histogram based prototypes.')
	parser.add_argument('-u','--uniform',default=False,action='store_true',help='Build HMax model using uniform random prototyppes.')
	parser.add_argument('-s','--shuffle',default=False,action='store_true',help='Build HMax model using shuffled random prototypes.')
	parser.add_argument('-n','--normal',default=False,action='store_true',help='Build HMax model using normal random prototypes.')
	parser.add_argument('-all','--all',default=False,action='store_true',help='Test HMax model for each prototype shema')
	args = parser.parse_args()
	if args.all:
		testAll(args.posDir,args.negDir,args.numProto,args.objType)
	else:
		main(args.posDir,args.negDir,args.numFolds,args.numProto,args.histogram,args.uniform,args.shuffle,args.normal,args.objType)
	"""
	  if len(sys.argv) < 3:
	    #sys.exit("usage: %s XFORM-DIR IMAGE WINDOW-STEP-SIZE THRESHOLD" % sys.argv[0])
	    sys.exit("usage: %s Pos-Dir Neg-Dir Num-folds" % sys.argv[0])
	  training_pos,training_neg,n = sys.argv[1:4]
	  main(training_pos,training_neg,int(n))
	"""
	
