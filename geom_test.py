import geom

bbox = (1,1,3,3)
bbox2 = (.5,1,2,4)
print geom.calcArea(bbox)
print geom.intersects(bbox,bbox2)
print geom.intersects(bbox,(0,1,1.5,2.6))
print geom.intersects((0,0,15,2.6),bbox)
print geom.intersects((1.5,0,2,1.5),bbox)
print geom.intersects(bbox,(1.5,0,2,1.5))
print "+++"
print geom.intersects(bbox,(1,1,3,2))
print geom.intersects((1,1,3,2),bbox)
print geom.intersects((1,1,3,3),(0,0,5,2))

#print geom.contains(bbox,bbox2)
#geom.intersection(bbox,bbox2)
