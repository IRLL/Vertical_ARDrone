#!/usr/bin/env python

from geometry_msgs.msg import Twist


#struct definitions
class xy():
	def __init__(self, x, y):
		self.x = x
		self.y = y

class pi_controller():
	def __init__(self, proportional, integral):
		self.proportional_factor = proportional
		self.integral_factor = integral
		self.error_integral = 0

	def run(self, error):
		correction_p = error * self.proportional_factor
		
		self.error_integral = self.error_integral + error)
		
		correction_i = self.error_integral * self.integral_factor

		correction = correction_p + correction_i
			
		return correction

class pi_controller_2d():
	def __init__(self, proportional, integral, timestep):
		self.x_controller = pi_controller(proportional, integral)
		self.y_controller = pi_controller(proportional, integral)

	def run(self, error):
		correction = xy()
			
		correction.x = self.x_controller.run(error.x)
		correction.y = self.y_controller.run(error.y)
		
		return correction

class Stabilizer():
	def __init__(self, proportional=1, integral=.1):
		self.controller = pi_controller(proportional, integral) 
		self.error_sub = rospy.Subscriber('v_controller/control_state', Float32MultiArray, self.run)
		self.control_pub = rospy.Publisher('v_controller/control_cmd', Twist)
		rospy.init_node('stabilizer', anonymous=False)

	def run(self, data):
		command = Twist()		

		error = xy(data[0], data[1])
		correction = controller.run(error)
		
		command.linear.y = correction.x

		command.linear.x = correction.y
		self.control_pub.publish(command)
		


if __name__ == "__main__":
	stabilizer = Stabilizer()
	rospy.spin() 
