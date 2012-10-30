#!/usr/bin/python 

# takes in a bbox tuple in the form of (x0,y0,x1,y1)
def calcArea(bbox):
	x0,y0,x1,y1 = bbox
	width = x1-x0
	height = y1-y0
	return width*height

# helper function to see if a point is contained inside a bounding box.
def pointWithin(point,bbox):
	x0,y0,x1,y1 = bbox
	if (point[0] >= x0 and point[0] <= x1) and (point[1]>=y0 and point[1] <= y1):
		return True


# test to see if bbox1 intersects with bbox2 by seeing if any of the points are contained within the bounds of bbox2.
# returns if it is true and which point
def intersects(bbox1,bbox2):
	x0,y0,x1,y1 = bbox1
	x0_prime,y0_prime,x1_prime,y1_prime = bbox2
	
	"""	
	cords = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
	for point in cords:
		if pointWithin(point,bbox2):
			return True
			
	"""
	
	# see if bbox1 right hand side intersects with the left hand side of bbox 2
	if x1 > x0_prime :
		
		# check if intersection is at bottom right corner
		if y1 > y0_prime and y1 <= y1_prime:
		
			return True
			
		# check if intersection is at top right cornder
		if y0 > y1_prime and y0 <= y0_prime:
			print "b"
			return True
			
		# check if whole right side is contained within bbox 2
		if y0 >= y0_prime and y0 <= y1_prime and y1 >= y0_prime and y1 <= y1_prime:
		
			return True
		
	# see if bbox2 right side intersects with left side of bbox1. This is the same as treating bbox 1 left as intersecting with bbox2 right. Kind of confusing.
	if x1_prime > x0:
		
		# check if intersection is at bottom right corner
		if y1 > y0_prime and y1 <= y1_prime:
			return True
		
		# check if intersection at top left corner
		if y0_prime > y1 and y0_prime <= y0:
			return True
			
		if y0_prime >= y0 and y0_prime <= y1 and  y1_prime >= y0 and y1_prime <= y1:
			return True 
			
	# check if bbox1 top is inside bbox2
	if y1 > y0_prime:

		if x0 >= x0_prime and x0 <= x1_prime:
			print 'eh'
			print (x0,y0,x1,y1_prime)
			return True
	
	# check if bbox2 bottom is inside bbox1
	if y1_prime > y0:
		print y1_prime
		if x0_prime >= x0 and x0_prime <= x1:
			print "hi"
			print (x0_prime,y0,x1_prime,y1_prime)
			return True
	return False

"""
Tests to see if bbox1 contains bbox2 completely. if so return true.
"""
def contains(bbox1,bbox2):

	x0,y0,x1,y1 = bbox2
	cords = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
	count = 0
	for point in cords:
		if pointWithin(point,bbox1):
			count += 1
	if count == len(cords):
		return True
	else:
		return False
"""
calculate the new intersection box of bbox1 to bbox2
"""
def intersection(bbox1,bbox2):
	x0,y0,x1,y1 = bbox1

	
	x0_prime,y0_prime,x1_prime,y1_prime = bbox2
	

	# check if intersection from bottom left
	if bbox1[0] <= bbox2[0] and bbox1[3] >= bbox2[3]:
		bbox = (x0_prime,y0,x1,y1_prime)
		return bbox
	
	#check if intersection from top right
	if x1 >= x1_prime and y0 <= y0_prime:
		bbox = (x0,y0_prime,x1_prime,y1)
		return bbox
	
	# check if intersection from top left
	if x0 <= x0_prime and y0 <= y0_prime:

		if x1 >= x1_prime:
			return (x0_prime,y0_prime,x1,y1)

		return (x0_prime,y0_prime,x1,y1)

	#check if intersection from bottom right
	if x1 >=x1_prime and y1 >= y1_prime:
		bbox = (x0,y0,x1_prime,y1_prime)
		return bbox
	# check if intersection from top
	if y0 <= y0_prime and (x0 >= x0_prime and x1 <= x1_prime):
		bbox = ( x0, y0_prime,x1,y1)
		return bbox
	# check if intersection from bottom

					
					






	
	
