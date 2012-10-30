#!/usr/bin/python
import Windowing_C1 as windowing
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
import cPickle as pickle


"""
 calculate a gaussian value given a x,y point from -r to r
"""
def calc_gauss(x,y,r):
  pi = math.pi
  sigma = 1.3
  scale = 2*math.pi*((sigma*r)**2)
  gauss = (float(1)/float(2*pi*math.pow(sigma*r,2)))*math.exp(-(float(math.pow(x,2)+math.pow(y,2))/float(2*math.pow(sigma*r,2))))
  return 1-(gauss*scale)

"""
build the neighborhood suppression matrix such that it is the size of the global max bounding box.
find the "radius" of that box, 1/2 the width, and fill a NxN matrix with the values of 1-calc_gause(x,y). These will be our suppression values!
"""

def build_neighborhood(bbox):

  box_length = (bbox[2]-bbox[0])
  box_length = box_length + int((box_length*.35))
  #check to see if the bbox we found was less than the average vehicle size found in the database or not
  if box_length < 344:
    box_length = box_length*2
  
  r = int(box_length/2)
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


""" suppress regions of the image using Local Neighborhood suppresion."""
def suppress_neighborhood(bboxes,dvals):
  #print "Max box",bboxes[np.argmax(dvals)]
  
  """
  taking the global max bbox, suppress all bboxes within the max bounding box
  """
  global_max = bboxes[np.argmax(dvals)]
  
  
  global_center = find_center(global_max)
  
  neighborhood = build_neighborhood (global_max)
  
  x0 ,y0,x1,y1 = zip(global_max)
  
  gxc = global_center[0] - x0[0]
  gyc = global_center[1] - y0[0]
  
  decision_val = copy.deepcopy(dvals[np.argmax(dvals)][0])
  #print decision_val
  if decision_val <= 0.0 :
    return global_max,decision_val,-1
  
  
  #print "Max decision val", decision_val
  index = np.argmax(dvals)
  dvals[np.argmax(dvals)][0] = dvals[np.argmax(dvals)][0]*0
  
  for ind in range(0,len(bboxes)):
    tmp_box = zip(bboxes[ind])
  
    xp0,yp0,xp1,yp1 = tmp_box
  
    """
    Look to see if the center of the box is within the neighborhood by some pixel amount.
    If it is then suppress that crop in the neighborhood by figuring out where it falls within that matrix.
    """
    xc,yc = zip(find_center(bboxes[ind]))
  
  
    if xc[0] > x0[0] and xc[0] < x1[0] and yc[0] > y0[0] and yc[0] < y1[0]:
      xc_index = xc[0]-x0[0]
      yc_index = yc[0]-y0[0]
      
      if ind != index:
  
        dvals[ind][0] = dvals[ind][0]*neighborhood[yc_index-1][xc_index-1]
        #dvals[ind][0] = dvals[ind][0]*0.0
  
  return global_max,decision_val[0],index

"""
Given N testing images, we want to see the confusion matrix of all the crops that are extracted.

Get N test Images with at least 1 target object in it and extract all crops from that object.
The returned list of tuples, will contain (hasObj,bbox,decisionVal). We then will sort and count the number of positive and negative
instances and see which ones are true pos and true neg.
"""
"""
Draw the bounding box from a series of line objects.
"""
def draw_bbox(bbox,image,fill_color):
  x0,y0,x1,y1 = bbox
  p1 = (x0,y0)
  p2 = (x1,y0)
  p3 = (x0,y1)
  p4 = (x1,y1)

  image_copy = image.copy();
  draw = ImageDraw.Draw(image_copy)
  draw.line([p1,p2],fill_color,width= 2)
  draw.line([p1,p3],fill_color,width=2)
  draw.line([p2,p4],fill_color,width = 2)
  draw.line([p3,p4],fill_color, width = 2)

  return image_copy
"""
Test using 100 images so pull directly from test set folder. should they be random maybe?
"""
def testdb_loader(test_dir,anno_dir,obj_type):

  contents = os.listdir(test_dir)
  test_set = list()
  test_img = list()
  i = 0
  j = 0

  while i < 15:

    num = len(resultcheck.getGroundTruths(contents[j],test_dir,anno_dir,obj_type))
    print "Number of objects",num
    if num > 0:
      print "ADD",num
      test_set.append(contents[j])
      test_img.append(contents[j])
      i += 1
    j += 1

  return test_set,test_img

def c_matrix(truth,predicted):
  pass

def main(test_dir,anno_dir,c1,obj_type,step_size,savename):
  save_res = open(os.path.join('results',savename+'_OldSetConfusion.txt'),'wb') 
  testSet,testImages = testdb_loader(test_dir,anno_dir,obj_type)
  test_res = list()
  all_det = list()
  all_objects = list()
  # get all ground truths for the test set
  for test in testSet:
    ground_truth = resultcheck.getGroundTruths(test,test_dir,anno_dir,obj_type)
    all_objects.append((ground_truth,test))

    # process all the images and save the tuples in form (bbox,dval,img)
    results,all_test_results= windowing.main(c1,os.path.join(test_dir,test),step_size,24,True)
    test_res.append(results)
    all_det.append(all_test_results)
  
  print "saving off dval data"
  dvals_copy_file = open(os.path.join('pickledCrops','allDetectionsConfusion.dat'),'wb')
  pickle.dump(all_det,dvals_copy_file)
  dvals_copy_file.close()
  count = 0

  img_res = list()
  print "number of boxes found", count
  for a in range(0,len(all_det)):
    imgdet = list()
    for det in all_det[a]:
      detection_box = det
      print "ahhhh", all_objects[a][0]
      imgdet.append(resultcheck.c_matrix(all_objects[a][0],detection_box))
      #for labels in all_objects:
      #  pass
      #print "Labels", labels
    img_res.append(imgdet)
  tp=0
  fp=0
  tn=0
  fn=0
  
  print "Crops?", len(img_res) , len(all_det)
  for i in range(0,len(img_res)):
    img = Image.open(os.path.join(test_dir,testImages[i]))
    #draw = ImageDraw.Draw(img)
    
    for imgdata in img_res[i]:
      #tp instance
      imgdata = imgdata[0]
      
      if imgdata[1] > 0 and imgdata[2] == 1:
        #draw.rectangle((imgdata[0][0],imgdata[0][1],imgdata[0][2],imgdata[0][3]),outline='green')
        img = draw_bbox(imgdata[0],img,'green')
        print imgdata
        tp += 1

      elif imgdata[1] > 0 and imgdata[2] == 0:
        fp += 1

      elif imgdata[1] <= 0 and imgdata[2] == 1:
        #draw.rectangle((imgdata[0][0],imgdata[0][1],imgdata[0][2],imgdata[0][3]),outline='red')
        img = draw_bbox(imgdata[0],img,'red')
        
        fn += 1
      elif imgdata[1] <= 0 and imgdata[2] == 0:
        tn += 1
        
    #print i
    #draw.rectangle((0,0,10,10),outline='red')
    print "ALL", all_objects[i]
    #print i
    #for ground in all_objects[i]:
      #print ground
    for gtruth in all_objects[i][0]:
      print gtruth
      ground_tuple = gtruth
      print "GROUND", ground_tuple
      #img = draw_bbox(ground_tuple,img,'blue')
    #img.save(os.path.join('res_img',all_objects[i][1]))
    
  print "Confusion Matrix"

  save_res.write(str(tp)+" "+str(fn)+"\n"+str(fp)+" "+str(tn)+"\n") 
  print tp,fn
  print fp,tn
  

if __name__ == '__main__':
  if len(sys.argv) < 6:
    #sys.exit("usage: %s XFORM-DIR IMAGE WINDOW-STEP-SIZE THRESHOLD" % sys.argv[0])
    sys.exit("usage: %s test-dir-path annotation-dir-path xform obj-type Step-Size Save-Name" % sys.argv[0])
  test_dir,anno_dir,xform,obj_type,step_size,savename = sys.argv[1:7]
  main(test_dir,anno_dir,xform,str(obj_type),int(step_size),savename)
