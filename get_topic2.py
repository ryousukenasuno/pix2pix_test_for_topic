import sys
import rospy
import cv2
import message_filters
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime

from models.test_model import TestModel
from options.test_options import TestOptions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from data.data_loader import CreateDataLoader


class get_images:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_pub = rospy.Publisher("/test/depth/test_image",Image)
        self.input_pub = rospy.Publisher("/test/depth/input_image",Image)
        self.original_pub = rospy.Publisher("/test/depth/org_image",Image)
        self.image_sub = rospy.Subscriber("/camera/depth/image_raw",Image,self.callback)
	self.opt = TestOptions().parse()
        self.opt.nThreads = 1
        self.opt.batchSize = 1
        self.model = TestModel()
	self.model.initialize(self.opt)
        self.x,self.width = 192,256
        self.y,self.height = 112,256
        self.loss = 10000

    #def callback(self, rgb_data, depth_data):
    def callback(self, depth_data):
        try:
            #depth_image_raw = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
            depth_image_raw = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
            depth_image = np.zeros((1,1,256,256),"float32")
            tmp = depth_image_raw[self.y:self.y+self.height,self.x:self.x+self.width].astype("float32")
            depth_image[0,0,:,:] = (tmp - tmp.mean())/np.std(tmp)
            test = torch.utils.data.TensorDataset(torch.from_numpy(depth_image),torch.from_numpy(depth_image))
            test_loader = torch.utils.data.DataLoader(test, batch_size=self.opt.batchSize, shuffle=False)
            #test_loader = torch.utils.data.DataLoader(depth_image, batch_size=self.opt.batchSize, shuffle=False)
            #data_loader = CreateDataLoader(self.opt)
            #dataset = data_loader.load_data()
            for i, data in enumerate(test_loader):
                x,y = data
                self.model.set_input(x)
                self.model.test()
                visuals = self.model.get_current_visuals()
        except CvBridgeError as e:
            print(e)
	try:
            #print((visuals['real_A'].dtype))
            
            loss = abs(depth_image[0,0,:,:] - visuals['fake_B'][:,:,0]).mean()
            depth_image = (depth_image[0,0,:,:])*np.std(tmp)+tmp.mean()
            visuals['real_A'] = (visuals['real_A'][:,:,0])*np.std(tmp)+tmp.mean()
            visuals['fake_B'] = (visuals['fake_B'][:,:,0])*np.std(tmp)+tmp.mean()
            depth_image = np.uint16(depth_image)
            visuals_in = np.uint16(visuals['real_A'])
            visuals_tar = np.uint16(visuals['fake_B'])
            #loss = abs(depth_image - visuals['fake_B']).mean()
            if loss < self.loss:
                print(self.loss)
                self.loss = loss
            	cv2.imwrite("original.png",depth_image)
            	cv2.imwrite("input.png",visuals_in)
            	cv2.imwrite("target.png",visuals_tar)
            self.input_pub.publish(self.bridge.cv2_to_imgmsg(visuals_in, "mono16"))
            self.depth_pub.publish(self.bridge.cv2_to_imgmsg(visuals_tar, "mono16"))
            self.original_pub.publish(self.bridge.cv2_to_imgmsg(depth_image, "mono16"))
        except CvBridgeError as e:
            print(e)



def main(args):
    fp = get_images()
    rospy.init_node('get_images', anonymous=True)
    #image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
    #depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)

    #ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.5)
    #ts = message_filters.ApproximateTimeSynchronizer([depth_sub], 1, 0.5)
    #ts.registerCallback(fp.callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
