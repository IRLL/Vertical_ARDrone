# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:54:40 2015

@author: brownyoda
"""

import rospy
import sys
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetModelStateRequest

from xbox_controller import xboxcontroller
from structs import Data
#from drawStartState import drawStartState
from drawAction import drawAction
#from rewardFnc import rewardFnc
#from drawNextState import drawNextState

from math import sqrt, isinf

import random

_reset_pos = None
_enable_controller = None
_soft_reset_pub = None
_takeoff_pub = None
_action_pub = None
_state_sub = None
_visible_sub = None
_threshold = None

_state_x = 0.0
_state_y = 0.0
_state_z = 0.0
_visible = 0

def getState(data):
    global _state_x, _state_y, _state_z
    _state_x = data.data[0]
    _state_y = data.data[1]
    _state_z = data.data[2] - 1.5 #dc bias so that it floats at 1.5 meters

def visible_callback(visible):
    global _visible
    _visible = visible.data

def reset_sim(x, y, z=0, angle=0):
    _enable_controller.publish(Bool(0))
    _takeoff_pub.publish(Empty())
    rospy.sleep(.1)
    a = SetModelStateRequest()
    a.model_state.model_name = 'quadrotor'
    a.model_state.pose.position.z = 2 + z
    a.model_state.pose.position.x = 3 + x
    a.model_state.pose.position.y = 0 + y
    _reset_pos(a)
    a.model_state.model_name = 'unit_sphere_4'
    a.model_state.pose.position.z = 0.052
    a.model_state.pose.position.x = 3
    a.model_state.pose.position.y = 0
    _reset_pos(a)
    rospy.sleep(.5)
    _soft_reset_pub.publish(Empty())

def startEpisode():
    #x = random.uniform(-.23, .23)
    #y = random.uniform(-.23, .23)
    x = random.uniform(-.8, .8)
    y = random.uniform(-.23, .23)
    reset_sim(x,y,0)
    rospy.sleep(.5)

def obtainData(policy, L, H, param):
    global _reset_pos, _enable_controller, \
           _soft_reset_pub, _takeoff_pub, \
           _action_pub, _state_sub, \
           _visible_sub, _threshold

    print "waiting for service"
    rospy.wait_for_service('/v_control/reset_world')
    print "done"
    _reset_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    _enable_controller = rospy.Publisher('v_controller/move_enable', Bool)
    _soft_reset_pub = rospy.Publisher('v_controller/soft_reset', Empty)
    _takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty)
    _action_pub = rospy.Publisher('v_controller/agent_cmd', Twist)
    _state_sub = rospy.Subscriber('v_controller/state', Float32MultiArray, getState)
    _visible_sub = rospy.Subscriber('v_controller/visible', Bool, visible_callback)
    _threshold = rospy.get_param("v_controller/threshold")

    N = param.param.N
    M = param.param.M
    data = [Data(N, M, L) for i in range(H)]

    # Based on Jan Peters; codes.
    # Perform H

    for trials in range(H):
        startEpisode()

        # Save associated policy
        data[trials].policy = policy

        # Draw the first state
        data[trials].x[:,0] = np.array([[-_state_x/4.0, -_state_y/3.0, _state_z/3.0]])

        # Perform a trial of length L
        for steps in range(L):
            if rospy.is_shutdown():
                sys.exit()
            # Draw an action from the policy
            xx = data[trials].x[:,steps]
            data[trials].u[:,steps] = drawAction(policy, xx, param)


            for j in range(M):
                data[trials].u[:,steps][j] = round(data[trials].u[:,steps][j], 5) + param.param.disturbance[j]
                print "action ", j, ":", data[trials].u[:,steps][j]

                if _visible == 0:# or (sqrt(xx[0][0]**2) <= .5 and sqrt(xx[0][1]**2) <= .5):
                    data[trials].u[:,steps][j] = 0.0
                elif data[trials].u[:,steps][j] > 1.0: # saturate
                    data[trials].u[:,steps][j] = 1.0
                elif data[trials].u[:,steps][j] < -1.0:
                    data[trials].u[:,steps][j] = -1.0


            command = Twist()
            command.linear.x = data[trials].u[:,steps][0]
            command.linear.y = data[trials].u[:,steps][1]
            command.linear.z = data[trials].u[:,steps][2]
            _action_pub.publish(command)
            rospy.sleep(.2)

            print "action: ", data[trials].u[:,steps]

            # Draw next state from environment
            #data[trials].x[:,steps+1] = drawNextState(data[trials].x[:,steps], data[trials].u[:,steps], param, i)
            state = np.array([[-_state_x/4.0, -_state_y/3.0, _state_z/3.0]])
            #state = np.array([[-_state_x/4.0, -_state_y/3.0]])
            data[trials].x[:,steps+1] = state

            # Obtain the reward from the
            #data[trials].r[0][steps] = rewardFnc(data[trials].x[:,steps], data[trials].u[:,steps]) # FIXME: Not similar to original

            u = data[trials].u[:,steps]
            u_p = u.conj().T

            reward = -sqrt(np.dot(np.dot(state, np.eye(N) * 10), state.conj().T)) - \
                      sqrt(np.dot(np.dot(u, np.eye(M) * 5), u_p))
            print "reward: ", reward

            if isinf(reward):
                print "Error: INFINITY"
                sys.exit(1)

            data[trials].r[0][steps] = reward

    return data
