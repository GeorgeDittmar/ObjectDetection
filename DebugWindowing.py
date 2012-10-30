#!/usr/bin/python

import pickle as pkl
import sys
import os
import Windowing_C1 as windowing


def main(xform,test_set,dictionary):
  object_dict = open(dictionary,'rb')
  objects = pkl.load(object_dict)
  object_dict.close()

  print len(objects)

  for key in objects.iterkeys():

    crops = objects[key]
    image = os.path.join(test_set,key)
    windowing.sanity_check(xform,image,25,24,crops,debug=False)
    print key


if __name__ == '__main__':
  if len(sys.argv) < 3:
    """
    sys.exit("usage: %s XFORM-DIR POS-DIR NEG-DIR" % sys.argv[0])
  xform_dir, pos_image_dir, neg_image_dir = sys.argv[1:4]
  main(xform_dir, pos_image_dir, neg_image_dir)
    """
    sys.exit("usage: %s xform test-set dictionary-file" % sys.argv[0])
  xform,test_set,dictionary= sys.argv[1:4]
  main(xform,test_set,dictionary)









