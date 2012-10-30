import AnnotationParser as an
import os

def loadTestSet(test_dir,anno_dir,obj_type):
  
  contents = os.listdir(test_dir)
  test_set = list()
  
  an.getTruthLocations(test_set,test_dir,anno_dir)
    


