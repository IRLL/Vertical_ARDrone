#!/usr/bin/env python

#struct definitions
class xy():
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y

class pi_controller():
	def __init__(self, proportional, integral):
		self.proportional_factor = proportional
		self.integral_factor = integral
		self.error_integral = 0

	def run(self, error):
		correction_p = error * self.proportional_factor
		
		self.error_integral = self.error_integral + error
		
		correction_i = self.error_integral * self.integral_factor

		correction = correction_p + correction_i
			
		return correction

class pi_controller_2d():
	def __init__(self, proportional, integral):
		self.x_controller = pi_controller(proportional, integral)
		self.y_controller = pi_controller(proportional, integral)

	def run(self, error):
		correction = xy()
			
		correction.x = self.x_controller.run(error.x)
		correction.y = self.y_controller.run(error.y)
		
		return correction

