#!/usr/bin/env python

import roslib
import rospy
import readchar
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from kbreader import KBreader


class DroneKeyboard():
	def __init__:
		self.takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty)
		self.land_pub = rospy.Publisher('/ardrone/land', Empty)
		self.control_pub = rospy.Publisher('/cmd_vel', Empty)
		self.kr = KBreader()
		self.running = True
		self.generate_cmds()


	def run():
		while self.running:	
			key = self.kr.getkey()	
			cmd = self.moves[key]
			self.control_pub.publish(cmd)
			

	def generate_cmds(self):
		self.moves = dict()
		cmd = Twist()
		amount = .2

		cmd = Twist()
		cmd.linear.y = amount
		self.moves['a'] = cmd
		
		cmd = Twist()
		cmd.linear.y = -amount
		self.moves['d'] = cmd
		
		cmd = Twist()
		cmd.linear.x = amount
		self.moves['w'] = cmd
		
		cmd = Twist()
		cmd.linear.x = -amount
		self.moves['s'] = cmd

		cmd = Twist()
		cmd.angular.z = -amount
		self.moves['q'] = cmd

		cmd = Twist()
		cmd.angular.z = -amount
		self.moves['e'] = cmd


	def move



	def test_quit(cmd):
		if cmd == char(13):
			return 1
		return 0
