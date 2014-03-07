#!/usr/bin/python
from glimpse.glab import *
from glimpse.util import stats
#import trainer
#import validator
import os
import sys
import Image
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

# remove any images from the lists that are too small to process
def  check_img(pos,neg,pos_path,neg_path):
  print len(pos),len(neg)
  
  for p in pos:
    tmpImg = Image.open(os.path.join(pos_path,p))
    w,h = tmpImg.size
    #if (w < 60 and h > 60) or (w > 60 and h < 50) or ( w < 50 or h < 50):
    if w < 70 or h < 70:
      print "removing"
      pos.remove(p)
  
  for n in neg:
    tmpImg = Image.open(os.path.join(neg_path,n))
    w,h = tmpImg.size
    if w < 70 or h < 70:
    #if (w < 50 and h > 50) or (w > 50 and h < 50) or ( w < 50 or h < 50):
      print "Removing"
      neg.remove(n)
  print "Left over:",len(pos),len(neg)

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

def test_prototypes(training_splits_pos,training_splits_neg,test_split_pos,test_split_neg,training_pos,training_neg,num_proto,hist,uni,shuff,norm):
  from glimpse.models import viz2
  params = viz2.Params()
  params.num_scales = 4
  SetModelClass(viz2.Model)
  SetLayer('c2')
  SetParams(params)
  print "HEY", num_proto,norm
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
  
  TrainSvm()
  print "Testing"
  accuracy,results_dict = TestSvm()

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
  print "False positive rate", fps
  Reset()
  return (fps,tps,auc(fps,tps),accuracy,cm)

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
  check_img(pos,neg,training_pos,training_neg)
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
  
  TrainSvm()
  print "Testing"
  accuracy,results_dict = TestSvm()

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
  print "False positive rate", fps
  Reset()
  return (fps,tps,auc(fps,tps),accuracy,cm)

def testAll(training_pos,training_neg,num_proto,obj_type):
	label = 'All'
	save = obj_type+'4000C2All.txt'
	save_res = open(os.path.join('ObjRec',save),'wb')
	
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
			
		TrainSvm()
		print "Testing"
		
		accuracy,results_dict = TestSvm()
		#save.write('accuracy at '+str(i)+' '+str(accuracy))
		
		decision_vals = results_dict['decision_values']
		tp_label = np.zeros(len(test_split_pos))
		tp_label.fill(1)
		tn_label = np.zeros(len(test_split_neg))
		tn_label.fill(-1)
		truth_labels = np.concatenate((tp_label,tn_label))
		fps,tps,thresholds = roc_curve(truth_labels,decision_vals)
		rocRes.append((fps,tps))
		Reset()

	# create plot and save to file
	for i in range(0,len(rocRes)):
		if i == 0:
			pl.plot(rocRes[i][0],rocRes[i][1],label="Imprinted")
		elif i == 1:
			
			pl.plot(rocRes[i][0],rocRes[i][1],label="Uniform Random")
		elif i == 2:	
			
			pl.plot(rocRes[i][0],rocRes[i][1],label="Normalized Random")
		elif i == 3:

			pl.plot(rocRes[i][0],rocRes[i][1],label="Shuffled Random")
		elif i==4:
			pl.plot(rocRes[i][0],rocRes[i][1],label="Histogram Random")
			
	caption = "Comparision of all "+obj_type+" C2 prototpying runs."
	pl.xlabel("False Positive Rate")
	pl.ylabel("True Positive Rate")
	pl.title(label)
	pl.legend(loc="lower right")
	pl.savefig(os.path.join('ObjRec',save))	

		
def main(training_pos,training_neg,num,numProto,h,u,s,n,obj_type):
  label = ''
  end = ''
  
  if h:
    end= obj_type+' c2Histogram'
    label = 'ROC for '+obj_type+' C2 Histogram Prototypes'
  elif u:
    end= obj_type+' c2Uniform'
    label ='ROC for '+obj_type+' C2 Random Prototypes'
  elif s:
    end= obj_type+' c2Shuffled'
    label = 'ROC for '+obj_type+' C2 Shuffled Prototypes'
  elif n:
    end= obj_type+' c2Normalized'
    label = 'ROC for '+obj_type+' C2 Normalized Prototypes'
  else:
    end= obj_type+' c2Imprinted'
    label = 'ROC for '+obj_type+' C2 Imprinted Prototypes'
    
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
  caption = "Comparision of all C2 prototpying runs for "+obj_type+"."
  pl.xlabel("False Positive Rate")
  pl.ylabel("True Positive Rate")
  pl.title(label)
  #pl.text(.5,.5,text)
  pl.legend(loc="lower right")
  pl.savefig(os.path.join('ObjRec',end))
  print "Mean AUC", float(mean_auc)/len(accuracies_c2)
  save_res.write("Mean AUC: "+str(float(mean_auc)/len(accuracies_c2)))
  auc_array = np.array(auc_list)
  save_res.write("STD: "+str(np.std(auc_array)))
  print np.std(auc_array)
  
def numProtoTest(posDir,negDir,numFolds,numProto,obj_type,h,u,s,norm):
	save_res = open(os.path.join('ObjRec',obj_type+'NumberProto.txt'),'wb') 
	for i in range(0,numFolds):
	    # get all the contents of the directories
	    pos = os.listdir(posDir)
	    neg = os.listdir(negDir)
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
		    training_splits_pos.append(os.path.join(posDir,postemp))
		    training_splits_neg.append(os.path.join(negDir,negtemp))
		
	    if len(pos) == 0:
		    test_split_pos = training_splits_pos
	    if len(neg) == 0:
		    test_split_neg = training_splits_neg
	    for p in pos:
		    test_split_pos.append(os.path.join(posDir,p))
	    for n in neg:
		    test_split_neg.append(os.path.join(negDir,n))
		
	    prototypes = (400,800,1200,1600,2000,2400,2800,3200,3600,4000)
	    acc_c2 = list()
	    auc_list = list()
	    accuracy_list = list()
	    avg_auc = list()
	    for numProto in prototypes:
		    C2_NumList = list()
		    print numProto
		    for i in range(0,5):
		      C2_NumList.append(test_prototypes(training_splits_pos,training_splits_neg,test_split_pos,test_split_neg,posDir,negDir,numProto,h,u,s,norm))
		    #acc_c2.append(test_prototypes(training_splits_pos,training_splits_neg,test_split_pos,test_split_neg,posDir,negDir,numProto,h,u,s,norm))
		    auc_avg_round = [ x[2] for x in C2_NumList]
		    avg = sum(auc_avg_round)/float(len(auc_avg_round))
		    save_res.write('AVg AUC '+str(avg)+' '+str(numProto)+'\n')
		    auc_list.append(avg)
	    #auc_list = [ x[2] for x in acc_c2]
	    """
	    for i in range(0,len(acc_c2)):
		    save_res.write('AUC of Run '+str(i)+' : '+str(acc_c2[i][2])+'\n')
		    save_res.write('Accuracy of Run '+str(i)+' : '+str(acc_c2[i][3])+'\n')
		    save_res.write('Confusion Matrix of Run '+str(i)+' : '+str(acc_c2[i][4])+'\n')
		    #mean_auc += accuracies_c2[i][2]
		    #auc_list.append(acc_c2[i][2])
		    accuracy_list.append(acc_c2)
	    """
	    pl.plot(prototypes,auc_list)
	pl.ylabel("AUC")
	pl.xlabel("Number of Prototypes")
	pl.title("Number of Prototypes used for classification "+obj_type)
	
	pl.savefig(os.path.join('ObjRec',obj_type+"_numProtoTest_2"))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='C2 training script to allow training models using different prototype schemes.')
	parser.add_argument('posDir')
	parser.add_argument('negDir')
	parser.add_argument('numFolds',type=int)
	parser.add_argument('numProto',type=int)
	parser.add_argument('objectType')
	parser.add_argument('-hist','--histogram',default=False,action='store_true',help='Build HMax model using histogram based prototypes.')
	parser.add_argument('-u','--uniform',default=False,action='store_true',help='Build HMax model using uniform random prototyppes.')
	parser.add_argument('-s','--shuffle',default=False,action='store_true',help='Build HMax model using shuffled random prototypes.')
	parser.add_argument('-n','--normal',default=False,action='store_true',help='Build HMax model using normal random prototypes.')
	parser.add_argument('-all','--all',default=False,action='store_true',help='Test HMax model for each prototype shema')
	parser.add_argument('-numT','--numT',default=False,action='store_true',help='Test HMax model for each prototype shema')
	args = parser.parse_args()
	if args.all:
		testAll(args.posDir,args.negDir,args.numProto,args.objectType)
	elif args.numT:
		print "Testing number of prototypes"
		numProtoTest(args.posDir,args.negDir,args.numFolds,args.numProto,args.objectType,args.histogram,args.uniform,args.shuffle,args.normal)
	else:
		main(args.posDir,args.negDir,args.numFolds,args.numProto,args.histogram,args.uniform,args.shuffle,args.normal,args.objectType)
	"""
	  if len(sys.argv) < 3:
	    #sys.exit("usage: %s XFORM-DIR IMAGE WINDOW-STEP-SIZE THRESHOLD" % sys.argv[0])
	    sys.exit("usage: %s Pos-Dir Neg-Dir Num-folds" % sys.argv[0])
	  training_pos,training_neg,n = sys.argv[1:4]
	  main(training_pos,training_neg,int(n))
	"""
	
