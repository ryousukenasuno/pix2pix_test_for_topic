import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np

def callback(image_sub):
	#print(depth)	
	try:
		#depth_image = CvBridge.imgmsg_to_cv2(image_sub,"CV_16UC1")
		depth_image = CvBridge.imgmsg_to_cv2(image_sub,"32FC1")
		#print(depth_image)
		#print(np.shape(image_sub))
		#cv2.imwrite("tmp.png",image_sub)
	except CvBridgeError as e:
		print(e)
	

def main():
	node_name = "/camera/depth/image_raw"
	rospy.init_node('listener', anonymous=True)
	image_sub = message_filters.Subscriber(node_name, Image)
	#ts = message_filters.TimeSynchronizer([image_sub], 10)
	#ts.registerCallback(callback)
	image_sub.registerCallback(callback)
	rospy.spin()

if __name__== '__main__':
	main()
