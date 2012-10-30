#!/usr/bin/python
import urllib2
import os
import sys
import Image

#import easy to use xml parser called minidom:
import xml.dom.minidom as minidom

def main(annotations,image):

  file_basename = os.path.basename(os.path.splitext(image)[0])
  extension = "_LMformat.xml"
  xml_doc = os.path.join(annotations,file_basename + extension)
  # used to grab all the elements from the document
  xml_root = minidom.parse(xml_doc)
  xml = xml_root.documentElement

  xml = xml_root.getElementsByTagName("object")[0]
  print xml[0]

class Annotations:
  def __init__(self,name):
    bob = 1

if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit("usage: %s ANNOTATION-DIR IMAGE" % sys.argv[0])
  annotations, image = sys.argv[1:3]
  # Use a bounding box size (in C1 coordinates) of 24 units, which is equivalent
  # to 128 pixels in image space.

  main(annotations,image)

