#!/usr/bin/python
import Windowing_C2 as windowing
import AnnotationParser as resultcheck
import AnnotationChecker as checker
import os
import sys
import Image
import ImageDraw
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import copy
"""
Test using 100 images so pull directly from test set folder. should they be random maybe?
"""

def testdb_loader(test_dir,anno_dir,obj_type):

  contents = os.listdir(test_dir)
  test_set = list()
  i = 0
  j = 0
  runningTotal = 0
  setSize = 100
  while i < setSize:

    num = len(resultcheck.getGroundTruths(contents[j],test_dir,anno_dir,obj_type))
    #num = checker.getGroundTruths(contents[i],test_dir,anno_dir,obj_type)
    runningTotal = runningTotal + num
    print "Number of objects",num
    if num > 0:
      print "adding"
      test_set.append(contents[j])
      i += 1
    j += 1
  print runningTotal,setSize,i
  return test_set,float(runningTotal)/setSize
  
def flattenResults(testResults):
	flatList = list()
	for results in testResults:
		flatList = flatList + results
	return flatList
  
def calcPrecisionRecall(flatList,truthDict,test_dir,savename,save_res):
  sortedDetections = sorted(flatList,key=lambda plot: plot[1],reverse=True)
  # get the number of objects in the set
  total = 0
  for x in truthDict:
    total += len(truthDict[x])
    
  print "Total number of objects", total
  
  td = 0
  fd = 0
  precision = list()
  recall = list()
  dvals = list()
  for elm in sortedDetections:
    print elm
    image = elm[2]
    dval = elm[1]
    detection = elm[0]
    print detection
    groundTruth = truthDict[image]
    # check the detection of this element
    img = Image.open(os.path.join(test_dir,image))
    res = resultcheck.test_detection(groundTruth,detection,img,image,dval)
    td += res[0]
    fd += res[1]
    img = res[2]
    p = float(td)/float(td+fd)
    r = float(td)/float(total)
    precision.append(p)
    recall.append(r)
    dvals.append(dval)
    save_res.write(str(p) +" "+str(r)+' '+str(dval)+"\n")
  pl.plot(recall,precision)
  pl.title("Precision Recall Graph C2 Cars")
  pl.xlabel("Recall")
  pl.ylabel("Precision")
  pl.xlim([0.0,1.0])
  pl.ylim([0.0,1.0])
  pl.savefig(savename+"_C2.png")
  print precision
  print recall
  
 
    
"""
Function to execute the experiment for C1 object detection.

Given test-image directory grab 100 images, potentially more if we want.
anno_dir is the directory of the xml annotations for each image.
c1 are the experiment files that contain our vision models.
"""
def main(test_dir,anno_dir,c1,obj_type,step_size,savename):

    test_set,runningTotal = testdb_loader(test_dir,anno_dir,obj_type)
    
    save_res = open(os.path.join('results',savename+'_detection_experiment_C2.txt'),'wb')
    save_res.write('Average Number of annotations per image: '+str(runningTotal)+'\n') 
    print "wrote"
    print runningTotal
    test_res = list()
    true = 0
    false = 0
    total_truth = 0
    precision = list()
    recall = list()
    all_objects = list()
    truthDict = resultcheck.getTruthDictionary(test_set,test_dir,anno_dir,obj_type)
    # get all ground truths for the test set
    for test in test_set:
      ground_truth = resultcheck.getGroundTruths(test,test_dir,anno_dir,obj_type)
      
      all_objects.append((ground_truth,test))

      # process all the images and save the tuples in form (bbox,dval,img)
      results = windowing.main(c1,os.path.join(test_dir,test),step_size,9,True)
      test_res.append(results[0])
    
    print test_res
    
    test_res = flattenResults(test_res)
    calcPrecisionRecall(test_res,truthDict,test_dir,savename,save_res)
"""
    # pass this image into the windowing.main() method to be processed. Do this for all images first
    for test in test_set:
      results = windowing.main(c1,os.path.join(test_dir,test),step_size,24,True)
      print "Updating ground truth numbers"
      ground_truth = resultcheck.getGroundTruths(results[0][2],test_dir,anno_dir,obj_type)
      img_name = test
      total_truth += len(ground_truth)
      print "Process detection results."
      tp,fp = resultcheck.check_detections(test,results,ground_truth,test_dir,anno_dir)
      true += tp
      false += fp
      precision.append(float(true)/float(true+false))
      recall.append(float(true)/float(total_truth))
      print "TRUE,FALSE Detections:", true, false
"""
    
"""
    def pr(ground_truth,detections):

      decisions = list()
      for det in detections:
        if len(det) > 0:
          for elm in det:

            decisions.append(elm[1])

      max_dec = max(decisions)
      min_dec = min(decisions)
      print "Max,min",max_dec,min_dec
      thresh_step = float(max_dec)/float(11)

      def drange(start, stop, step):
        r = start
        while r < stop:
          yield r
          r += step

      i0=drange(0, max_dec, thresh_step)
      thresholds = np.arange(0,max_dec,thresh_step)
      print thresholds
      total_relevant = sum([len(x[0]) for x in ground_truth])

      pr_plot = list()
      for threshold in thresholds:

        truePos = 0
        falsePos = 0
        # run through each detection and compare it to ground truths
        for i in range(0,len(detections)):

          # pass truths into
          if len(detections[i])>0:
            truth_boxes = copy.deepcopy(ground_truth[i][0])
            tp,fp = resultcheck.check_detections(detections[i][0][3],detections[i],truth_boxes,test_dir,anno_dir,threshold)
            truePos += tp
            falsePos += fp
            print "TP,FP,Thresh CHECK", truePos,falsePos,threshold

        precision = float(truePos)/float(truePos+falsePos)
        recall = float(truePos)/float(total_relevant)
        print "Precision, Recall", precision,recall
        pr_plot.append((precision,recall))
      return sorted(pr_plot,key=lambda plot: plot[0],reverse=False)

    # calculate PR results and graph
    pr_results = pr(all_objects,test_res)

    recall = [x[1] for x in pr_results]
    precision = [x[0] for x in pr_results]


    print precision
    print recall
    
    #recall = [x for x in range(0,11)]

    pl.plot(recall,precision,linestyle='--',marker='o')
    pl.title("Precision Recall Graph C1 Cars")
    pl.xlabel("Recall")
    pl.ylabel("Precision")
    pl.xlim([0.0,1.0])
    pl.ylim([0.0,1.0])
    pl.savefig(obj_type+"_prgraph9_18.png")
	"""
if __name__ == '__main__':
  if len(sys.argv) < 6:
    #sys.exit("usage: %s XFORM-DIR IMAGE WINDOW-STEP-SIZE THRESHOLD" % sys.argv[0])
    sys.exit("usage: %s test-dir-path annotation-dir-path xform obj-type Step-Size num-scales savename" % sys.argv[0])
  test_dir,anno_dir,xform,obj_type,step_size,savename = sys.argv[1:7]
  main(test_dir,anno_dir,xform,str(obj_type),int(step_size),savename)

