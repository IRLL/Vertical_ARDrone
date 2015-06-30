#!/usr/bin/python

import rospy
from xboxdrv.xboxdrv_parser import Controller
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from dronedrv import dronedrv
from time import sleep
from sys import exit 
import signal


class xboxcontroller():
	def __init__(self):
		pass

	def hysteresis(self, value, threshold = .1):
		if (abs(value) > threshold):
			return value
		else: return 0

	def run(self):
		global running
		button_threshold = -0.5

		cmd = Twist()
		drone = dronedrv()
		# Get input from the two analog sticks as yaw, throttle, roll, and pitch. Take the (0 - 255) input value and
		# map it to a (-1 - 1) range.
		controller = Controller (["X2", "Y2", "X1", "Y1", "L2", "R2", "X", "/\\", "[]", "select"], ["yaw", "throttle", "roll", "pitch", "descend", "ascend", "takeover", "takeoff", "land", "kill"], (0, 255), (-1, 1))
		#controller = Controller (["X1", "Y1", "X2", "Y2"])

		print "wating for controller to init"
		while(controller.get_values() == {}):
			sleep(.5)
		print "done"

		print "running!"
		while True:
			control_packet = controller.get_values()
			if(control_packet == {}):
				print "dropped control signal: all"
			
			try:	
				if (control_packet["kill"] > -1):
					drone.kill()
					print "toggling reset"
					sleep(1)
				if (control_packet["takeoff"] > button_threshold):
					drone.takeoff()
					print "taking off!"
					sleep(1)
				if (control_packet["land"] > button_threshold):
					drone.land()
					print "landing!"
					sleep(1)

				if (control_packet["takeover"] > button_threshold):
					cmd = Twist()
					drone.cmd(cmd)
					controller.kill_controller()
					sleep(1)
					break	

				cmd.linear.x = self.hysteresis(-control_packet["pitch"])
				cmd.linear.y = self.hysteresis(-control_packet["roll"])
				cmd.angular.z = self.hysteresis(-control_packet["yaw"])

				cmd.linear.z = -control_packet["descend"] - -control_packet["ascend"] 
			except KeyError, e:
				print "dropped control signal: ", e
				cmd = Twist() #clear out message if bad control signal

			drone.cmd(cmd)

			sleep (.1)
		print "exiting xbox controller"

if __name__ == '__main__':
	rospy.init_node('xbox_controller', anonymous=True)
	controller = xboxcontroller()

	controller.run()
