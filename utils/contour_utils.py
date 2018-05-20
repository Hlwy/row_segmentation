# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO
from matplotlib import pyplot as plt
import utils as ut
import numpy as np
import os
import cv2

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

def combine_contours(contours):
	LENGTH = len(contours)
	status = np.zeros((LENGTH,1))
	for i,cnt1 in enumerate(contours):
		x = i
		if i != LENGTH-1:
			for j,cnt2 in enumerate(contours[i+1:]):
				x = x+1
				dist = find_if_close(cnt1,cnt2)
				if dist == True:
					val = min(status[i],status[x])
					status[x] = status[i] = val
				else:
					if status[x]==status[i]:
						status[x] = i+1

	unified = []
	maximum = int(status.max())+1
	for i in xrange(maximum):
		pos = np.where(status==i)[0]
		if pos.size != 0:
			cont = np.vstack(contours[i] for i in pos)
			hull = cv2.convexHull(cont)
			unified.append(hull)

	cv2.drawContours(tmpFil,unified,-1,(255,255,255),2)


def show_hulls(img_left,img_right,cL,cR,parent):
	hullL = cv2.convexHull(cL)
	hullR = cv2.convexHull(cR)
	# print(hullL)
	# print(hullR)
	cv2.drawContours(img_left, [hullL], -1, (128, 0, 0), 1)
	cv2.drawContours(img_right, [hullR], -1, (0, 0, 128), 1)

	rows,cols = parent.shape[:2]
	[vx,vy,x,y] = cv2.fitLine(cL, cv2.DIST_L2,0,0.01,0.01)
	lefty = int((-x*vy/vx) + y)
	righty = int(((cols-x)*vy/vx)+y)
	cv2.line(img_left,(cols-1,righty),(0,lefty),(255,0,0),2)

	[vx,vy,x,y] = cv2.fitLine(cR, cv2.DIST_L2,0,0.01,0.01)
	lefty = int((-x*vy/vx) + y)
	righty = int(((cols-x)*vy/vx)+y)
	cv2.line(img_right,(cols-1,righty),(0,lefty),(0,0,255),2)
	combined = np.hstack((img_left, img_right))
	plt.figure(4)
	plt.clf()
	plt.imshow(combined)
	plt.show()

	return combined


def highlight_all_contours(contours,display):
	i = 0
	for cnt in contours:
		i+=1
		M = cv2.moments(cnt)
		hull = cv2.convexHull(cnt)
		cv2.drawContours(display, [hull], -1, (0, 0, 255), 1)
		try:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			cv2.circle(display, (cX, cY), 2, (255, 255, 255), -1)
			cv2.putText(display, "center"+str(i), (cX - 20, cY - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		except:
			print("ERROR: Moments not found")
			pass
	return display
