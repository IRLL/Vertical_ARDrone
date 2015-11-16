#!/usr/bin/env python

import rospy
import numpy as np
import os
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import LinkStates
import time

class GoForward():
	def __init__(self):
		rospy.init_node('GoForward', anonymous=False)

		rospy.loginfo("To stop TurtleBot CTRL + C")  
		rospy.on_shutdown(self.shutdown)
		self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
		rospy.Subscriber('/scan', LaserScan, self.laser_callback)
		rospy.Subscriber('/gazebo/link_states', LinkStates, self.globalstate_callback)
		
		self.r = rospy.Rate(10);
		self.move_cmd = Twist()
		self.move_cmd.linear.x = 0
		self.move_cmd.angular.z = 0
		self.globalposition = [0,0,0]
		self.goal = [1.0, -1.0, np.pi/2]
		self.disturbance = 0.1 #np.random.uniform(low=-0.1, high=0.1, size=None)
		
		while not rospy.is_shutdown():
			# set up controller
			k1 = .5
			k2 = .5
			k3 = .5
			k4 = -.1*k3
			K = [[k1, 0, 0, 0],[0, k2, k3, k4]]
			
			# get state
			print("position: %1.2f" % self.globalposition[0]), 
			print (" %1.2f" % self.globalposition[1]),
			print(" %1.2f" % self.globalposition[2]), #,self.goal
			e = np.subtract(self.goal, self.globalposition)
			print ("  error: %1.2f" % e[0]), 
			print (" %1.2f" % e[1]),
			print(" %1.2f" % e[2])

			r = np.sqrt(e[0]**2 + e[1]**2)
			psi = np.arctan2(e[1],e[0])
			beta = self.globalposition[2]
			kappa = self.goal[2]
			theta = (kappa+psi)
			alpha = (psi-beta)
			phi = (kappa+beta)

			#gamma = np.arctan2(e[1],e[0]) - self.globalposition[2] #+self.goal[2] #+ np.pi
			#delta = np.arctan2(e[1],e[0]) #gamma + self.globalposition[2]
			#print "r:",r
			eps = .0001
			div = np.sign(alpha)*max(abs(alpha),eps)
			#phi = [r*np.cos(gamma), gamma, np.sin(gamma)*np.cos(gamma)/div*(gamma + delta),1] 
			#print "phi",phi
			#print "div",div
			state = [r*np.cos(alpha), alpha, np.sin(div)*np.cos(div)/div*(div + theta) + self.disturbance, 1]
			# get action

			action = np.dot(K,state)#[k1*np.cos(alpha)*r, k2*div+k3*np.cos(div)*np.sin(div)/div*(div+theta)] 
			print alpha, theta
			#print "action:",action 

			if e[0]*e[0]+e[1]*e[1]<.01: #np.dot(e,e)<.01:
				action = [0,0]
				print ("victory! resetting...")
				os.system("rosservice call /gazebo/set_model_state '{model_state: { model_name: mobile_base, pose: { position: { x: 0, y: 0 ,z: 0 }, orientation: {x: 0, y: 0.0, z: 0, w: 1 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'")

			self.move_cmd.linear.x = action[0]#np.sign(action[0])*min(1, .1*abs(action[0]))  #action[0]
			self.move_cmd.angular.z = action[1]#np.sign(action[1])*min(1, .1*abs(action[1])) #action[1]
			self.cmd_vel.publish(self.move_cmd)
			self.r.sleep()
			
	def shutdown(self):
		rospy.loginfo("Stop TurtleBot")
		self.move_cmd.linear.x = 0.0
		self.cmd_vel.publish(self.move_cmd)
		rospy.sleep(1)

	def globalstate_callback(self, data):
		for n in range(len(data.name)):
			#print data.name[n]
			if data.name[n]=='mobile_base::base_footprint':
				robotid = n
		#print data.pose[robotid]
		
		#convert quaternion
		qw = data.pose[robotid].orientation.w
		qx = data.pose[robotid].orientation.x
		qy = data.pose[robotid].orientation.y
		qz = data.pose[robotid].orientation.z
		thetap = 2*np.arctan2(np.sqrt(1-qw**2),qw)
		#thetan = 2*np.arctan2(-np.sqrt(1-qw**2),qw)
		den = np.sqrt(np.complex(1-qw**2))
		kp = [qx/den, qy/den, qz/den]
		#kn = -kp
		
		#since I know rotation is around z, I only need atan2 from rotation mat
		vt = 1-np.cos(thetap)
		ct = np.cos(thetap)
		st = np.sin(thetap)
		R00 = kp[0]**2*vt + ct
		R10 = kp[0]*kp[1]*vt + kp[2]*st
		theta = np.arctan2(np.real(R10),np.real(R00))
		
		robox = data.pose[robotid].position.x
		roboy = data.pose[robotid].position.y
		
		tmp = theta#-np.pi/2
		#if abs(tmp)>np.pi:
		#	if tmp>np.pi:
		#		tmp = tmp - np.pi
		#	else:
		#		tmp = tmp + np.pi

		self.globalposition = [robox,roboy,tmp]
		#print self.globalposition

	def laser_callback(self, scan):
		depths = []
		max_scan = -1
		too_close = 1
		for dist in scan.ranges:
			if np.isnan(dist):
				depths.append(max_scan)
			else:
				depths.append(dist)
				if dist<too_close:
					pass
	
	def wrapTo2Pi(self, num):
		num = np.mod(num,2*np.pi)
		return num

	def wrapToPi(self, num):
		outsidePi = (num < -np.pi) | (np.pi < num)
		if outsidePi:
			tmp = num+np.pi
			num = self.wrapTo2Pi(tmp) - np.pi
		return num

if __name__ == '__main__':
	try:
		GoForward()
	except:
		rospy.loginfo("GoForward node terminated.")

