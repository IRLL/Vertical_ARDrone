from xboxdrv_parser import Controller
from time import sleep
import os

def main ():
    # Get input from the two analog sticks as yaw, throttle, roll, and pitch. Take the (0 - 255) input value and
    # map it to a (-1 - 1) range.
    controller = Controller (["X1", "Y1", "X2", "Y2", "L2", "R2", "X", "/\\", "[]"], ["yaw", "throttle", "roll", "pitch", "descend", "ascend", "takeover", "takeoff", "land"], (0, 255), (-1, 1))
    #controller = Controller (["X1", "Y1", "X2", "Y2"])

    while True:
        control_packet = controller.get_values ()
	os.system("clear")
	for i in control_packet:
		print i, ": ", control_packet[i]
	
        # Update at 1000 messages a second
        sleep (.01)

if __name__ == '__main__':
    main()
