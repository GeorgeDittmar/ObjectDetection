import xml.dom.minidom as mini
import os
import Image
import ImageDraw
import sys
import csv

#sys.path.append('../Shapely-1.2.14/')

from shapely.geometry import Polygon

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
  
# takes an object xml tag and extracts the nested points
def get_bounds(image,bounds):
  img = Image.open(image)
  w,h = img.size
  points = list()
  for pt in bounds:
    x_tag = pt.getElementsByTagName("x")
    y_tag = pt.getElementsByTagName("y")

    for i in range(0,len(x_tag)):
      point = ((int(x_tag[i].firstChild.nodeValue.strip())),int(y_tag[i].firstChild.nodeValue.strip()))
      points.append(point)

  truth = Polygon(points)
  truth_bound = truth.bounds
  truth = [(truth_bound[0],truth_bound[1]),(truth_bound[2],truth_bound[1]),(truth_bound[2],truth_bound[3]),(truth_bound[0],truth_bound[3])]

  truth_poly = Polygon(truth)
  aspect_corrected = setAspectRatio(w,h,truth_poly.bounds)
  return aspect_corrected

def setAspectRatio(img_width,img_height,bounds):
  c = 5
  x_length =  (bounds[2]+c)-(bounds[0]-c)
  y_length =  (bounds[3]+c)-(bounds[1]-c)

  #print "XY PAIR", int(center.x),int(center.y),x_length,y_length
  bbox = ()
  xmin = 0
  ymin = 0
  xmax = 0
  ymax = 0

  if x_length >= y_length:
      diffs = x_length - y_length
      # divide in half so we expand the bbox around the object in both directions
      diffs = diffs/2

      xmin = bounds[0]-c
      ymin = (bounds[1]-diffs)-c
      xmax = bounds[2]+c
      ymax = bounds[3]+diffs+c

  elif y_length >= x_length:

      diffs = y_length - x_length
      diffs = diffs/2

      xmin = (bounds[0]-diffs)-c
      ymin = bounds[1]-c
      xmax = (bounds[2]+diffs)+c
      ymax = bounds[3]+c


  if xmin < 0:
    diff = 0 - xmin

    xmin = 0
    xmax = xmax + diff
  if xmax > img_width:

    diff = xmax - img_width
    xmax = img_width
    xmin = xmin - diff
  #test if top of bounding box is passed the ymin of the image
  if ymin < 0 :
    diff = 0-ymin

    #set ymin to 0 since we are looking at the top of the image where 0 corresponds to the top
    ymin = 0
    ymax = ymax + diff
  if ymax > img_height:

    diff = ymax - img_height
    ymax = img_height
    ymin = ymin-diff

  bbox = (xmin,ymin,xmax,ymax)
  return bbox
"""
Function that takes the truth detections for an object in an image and compares that
to the classifiers possible detections.
"""
def test_detection(truth_detections,detection,img,image,thresh):
  #detection = detection[0]
  #print "Testing", detection
  # append the first element to the end to have a closed polygon as per shapely's request
  detection_points = [(detection[0],detection[1]),(detection[2],detection[1]),(detection[2],detection[3]),(detection[0],detection[3])]
  detect_poly = Polygon(detection_points)
  td = 0
  fd = 0
  relevent = len(truth_detections)
  #print img
  if len(truth_detections) == 0:
    fd += 1
    return td,fd,img
  else:
    # check all detections against the ground truths.
    for truth_bound in truth_detections:
      truth_points = [(truth_bound[0],truth_bound[1]),(truth_bound[2],truth_bound[1]),(truth_bound[2],truth_bound[3]),(truth_bound[0],truth_bound[3])]
      #truth = Polygon(truth_points)
      #truth_bound = truth.bounds
      #truth = [(truth_bound[0],truth_bound[1]),(truth_bound[2],truth_bound[1]),(truth_bound[2],truth_bound[3]),(truth_bound[0],truth_bound[3])]

      truth_poly = Polygon(truth_points)
      #draw.polygon(detection_points,outline="red")
      if detect_poly.intersects(truth_poly):
        inter_poly = detect_poly.intersection(truth_poly)
        union_poly = detect_poly.union(truth_poly)
        det_result = float(inter_poly.area)/float(union_poly.area)
        #print det_result
        #print "intersection"
        if det_result >= .5:
          print "Correct detection"
          td += 1
          index = truth_detections.index(truth_bound)
          truth_detections.pop(index)
          img = draw_bbox(detection,img,"green")
          break
        elif det_result < .5:
          print "No detection"
          img = draw_bbox(detection,img,"red")
          fd += 1

      elif detect_poly.contains(truth_poly):
        print "object contained in detection!"
        # this checks to see if the the ground truth is contained within the detection. If so break out of the loop and add 1 to td.
        td += 1
        index = truth_detections.index(truth_bound)
        truth_detections.pop(index)
        #draw.polygon(detection_points,outline="blue")
        img = draw_bbox(detection,img,"green")
        break

      elif detect_poly.within(truth_poly):

        print "Detection is within ground truth"
        # this checks to see if the detection is within the ground truth box. If so check that the area is greater than 50% of the ground truth box.
        area = float(detect_poly.area)/(truth_poly.area)
        if area >= .5:
          td += 1
          index = truth_detections.index(truth_bound)
          truth_detections.pop(index)
          #draw.polygon(detection_points,outline="blue")
          img = draw_bbox(detection,img,"green")
          break
        else:
          fd += 1
          img = draw_bbox(detection,img,"red")
      else:
        fd += 1
        img = draw_bbox(detection,img,"red")
  print "DONE"
  #img.save('DUPLICATE'+str(thresh)+'s.jpg')
  return td,fd,img



"""
Function to check if the results found match with the annotations in the image.
"""
"""
def check_detections(car,imgfile,test_dir,anno_dir):

    file_list = os.listdir(anno_dir)
    end = "_LMformat.xml"
    #for f in file_list:
    cars_truth = list()
    bicycles_truth = list()
    pedestrians_truth = list()

    #file = open("Anno_XML/"+f)
    #print f

    #read the corresponding image annotation file
    xmldoc = mini.parse(os.path.join(anno_dir,os.path.splitext(imgfile)[0]+end))

    # load corresponding image and find objects
    img = Image.open(os.path.join(test_dir,imgfile))
    objects = xmldoc.getElementsByTagName("object")
    #if len(objects) == 0:
    #	break

    # parse the xml for the objects that contain a car
    for obj in objects:
	    name = obj.getElementsByTagName("name")
	    bounds = obj.getElementsByTagName("polygon")
	    if len(name) == 0:
		    break
	    obj_type = name[0].firstChild.nodeValue.strip()
	    if obj_type == 'car':
		    cars_truth.append(get_bounds(bounds))
		    #print "CARS list", len(cars)
	    elif obj_type == 'bicycle':
		    bicycles_truth.append(get_bounds(bounds))
		    #print "BIKES", len(bicycles)
	    elif obj_type == 'pedestrian':
		    pedestrians_truth.append(get_bounds(bounds))
		    #print "PEOPLE", len(bicycles)

    # loop through each detection and see if it overlaps with any of the known ground truths.

    true_detections = 0
    false_detections = 0

    
    print os.path.join(anno_dir,os.path.splitext(imgfile)[0]+end)

    #imgdraw = Image.open(os.path.join(test_dir,imgfile))
    #draw = ImageDraw.Draw(imgdraw)
    #truth = Polygon(cars_truth[0])
    #truth_bound = truth.bounds
    #truth = [(truth_bound[0],truth_bound[1]),(truth_bound[2],truth_bound[1]),(truth_bound[2],truth_bound[3]),(truth_bound[0],truth_bound[3])]

    #truth_poly = Polygon(truth)
    #draw.polygon(truth)
    #imgdraw.save("AnnotationCheck.JPG")

    for detection in car:
	    td,fd = test_detection(cars_truth,detection)
	    true_detections += td
	    false_detections += fd
	    if td == len(cars_truth):
	      print "FOUND ALL VEHICLES"
	      break

    precision = td/float(td+fd)
    recall =  td/len(cars_truth)

    return precision,recall

"""
"""
Take a list of ground truths and a list of tuples containing detections and decision values.
  Return:   List of tuples (bbox,dval,detected)

Function will loop through each detection for an image and compare that to the truthLabels given.

If a detection is overlapping,contains,  or within a truth label AND the dval is positive then we set the tpFlag to 1.
If a detection is overlapping etc AND dval is negative then set fnflag to 1.
If not detection is overlapping etc and dval is positive then increment fpFlag etc
"""
def c_matrix(truthLabels,img_detection):
  print "TRUTH,DETECTION",truthLabels,img_detection
  #what we return
  detection_results = list()

  #for detection in img_detections:

  bbox = img_detection[0]
  dval = img_detection[1]
  # append the first element to the end to have a closed polygon as per shapely's request
  detection_points = [(bbox[0],bbox[1]),(bbox[2],bbox[1]),(bbox[2],bbox[3]),(bbox[0],bbox[3])]
  detect_poly = Polygon(detection_points)

  # check all detections against the ground truths.
  for truth in truthLabels:

    print "Truth Label", truth
    truth_points = [(truth[0],truth[1]),(truth[2],truth[1]),(truth[2],truth[3]),(truth[0],truth[3])]
    #truth = Polygon(truth_points)
    #truth_bound = truth.bounds
    #truth = [(truth_bound[0],truth_bound[1]),(truth_bound[2],truth_bound[1]),(truth_bound[2],truth_bound[3]),(truth_bound[0],truth_bound[3])]

    truth_poly = Polygon(truth_points)
    #draw.polygon(detection_points,outline="red")
    if detect_poly.intersects(truth_poly):
      inter_poly = truth_poly.intersection(detect_poly)
      union_poly = truth_poly.union(detect_poly)
      det_result = float(inter_poly.area)/float(union_poly.area)

      if det_result >= .5:
        print "Correct detection"
        #td += 1
        #index = truth_detections.index(truth_bound)
        #truth_detections.pop(index)
        #draw.polygon(detection_points,outline="blue")
        detection_results.append((bbox,dval,1))
        break
      elif det_result < .5:
        detection_results.append((bbox,dval,0))

    elif detect_poly.contains(truth_poly):
      print "object contained in detection!"
      # this checks to see if the the ground truth is contained within the detection. If so break out of the loop and add 1 to td.
      #td += 1
      #index = truth_detections.index(truth_bound)
      #truth_detections.pop(index)
      #draw.polygon(detection_points,outline="blue")
      detection_results.append((bbox,dval,1))
      break

    elif detect_poly.within(truth_poly):

      print "Detection is within ground truth"
      # this checks to see if the detection is within the ground truth box. If so check that the area is greater than 50% of the ground truth box.
      area = float(detect_poly.area)/(truth_poly.area)
      if area >= .5:
        #td += 1
        #index = truth_detections.index(truth_bound)
        #truth_detections.pop(index)
        #draw.polygon(detection_points,outline="blue")
        detection_results.append((bbox,dval,1))
        break
      else:
        detection_results.append((bbox,dval,0))
    else:
      detection_results.append((bbox,dval,0))
  print "image", len(detection_results)
  return detection_results
"""
given an image its detections and ground truths, determine the number of true vs false detections
"""
def check_detections(image,image_result,ground_truth,test_dir,anno_dir,thresh):
  td = 0
  fd = 0
  print image
  total_truth = len(ground_truth)
  img = Image.open(os.path.join(test_dir,image))
  
  for truth_bound in ground_truth:
    truth = [(truth_bound[0],truth_bound[1]),(truth_bound[2],truth_bound[1]),(truth_bound[2],truth_bound[3]),(truth_bound[0],truth_bound[3])]
    draw.polygon(truth,outline="green")
    img = draw_bbox(truth_bound,img,'blue')
  #loop through all detections for the image and find the ones that match. stop checking either when we run out of detections or we run find all of the objects
  for detection in image_result:
    d = detection[0]
    
    if total_truth == td:
			pass
			
    if detection[1] >= thresh:
      # test the detection to the ground truths available
      a = test_detection(ground_truth,detection,img,image,thresh)
      guess = [(d[0],d[1]),(d[2],d[1]),(d[2],d[3]),(d[0],d[3])]
      #draw.polygon(guess,outline="red")
      print len(a)
      tp = a[0]
      fp =a [1]
      img = a[2]
      td += tp
      fd += fp
      #print "TP,FP", td,fd
  img.save(os.path.join('../gdittmar/experimentResults/Windowing/',image+'_'+str(thresh)+'s.jpg'))
  return td,fd

  
# return all the ground truths for a test image for a particular object class.
def getGroundTruths(image,test_dir,anno_dir,obj_type):
  end = "_LMformat.xml"
  #read the corresponding image annotation file
  xmldoc = mini.parse(os.path.join(anno_dir,os.path.splitext(image)[0]+end))
  t = obj_type.strip()
  ground_truths = list()

  # load corresponding image and find objects

  objects = xmldoc.getElementsByTagName("object")
  # parse the xml for the objects that contain a car
  for obj in objects:
    name = obj.getElementsByTagName("name")
    bounds = obj.getElementsByTagName("polygon")
    if len(name) == 0:
      break
    truth = name[0].firstChild.nodeValue.strip()
    if truth == t:
      ground_truths.append(get_bounds(os.path.join(test_dir,image),bounds))
  print len(ground_truths)
  return ground_truths

def getTruthLocations(test_set,test_dir,anno_dir):
  label = '.labl'
  truth_locations = {}
  for t in test_set:
    ground_truths = list()
    label_name = os.path.splitex(t)[0]+label
    f = open(os.path.join(anno_dir,label_name),'rb')
    reader = csv.reader(f,delimeter='|')
    for readline in reader:
      numBoxInd = 2
      print readline[2]
  

def getTruthDictionary(test_set,test_dir,anno_dir,obj_type):
  end = "_LMformat.xml"
	#read the corresponding image annotation file
  truthDict = {}
  for image in test_set:
    xmldoc = mini.parse(os.path.join(anno_dir,os.path.splitext(image)[0]+end))
    t = obj_type.strip()
    groundTruths = list()
    
    
    objects = xmldoc.getElementsByTagName("object")
    for obj in objects:
      name = obj.getElementsByTagName("name")
      bounds = obj.getElementsByTagName("polygon")
      if len(name) == 0:
        break
      truth = name[0].firstChild.nodeValue.strip()
      if truth == t:
        groundTruths.append(get_bounds(os.path.join(test_dir,image),bounds))
    truthDict[image] = groundTruths
    
  print truthDict
  return truthDict
