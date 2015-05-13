#!/usr/bin/env python

class pi_controller():
	def __init__(self, proportional, integral, timestep):
		self.proportional_factor = proportional
		self.integral_factor = integral
		self.timestep = timestep
		self.error_integral = 0

	def run(self, error):
		correction_p = error * self.proportional_factor
		
		self.error_integral = self.error_integral + (error * self.timestep)
		
		correction_i = self.error_integral * self.integral_factor

		correction = correction_p + correction_i
			
		return correction

class pi_controller_2d():
	def __init__(self, proportional, integral, timestep):
		self.x_controller = pi_controller(proportional, integral, timestep)
		self.y_controller = pi_controller(proportional, integral, timestep)

	def run(self, error):
		class correction:
			x=0
			y=0
		correction.x = self.x_controller.run(error.x)
		correction.y = self.y_controller.run(error.y)
		
		return correction
