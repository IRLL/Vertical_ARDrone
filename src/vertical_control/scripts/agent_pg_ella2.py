#!/usr/bin/env python
import rospy
import time
import sys
from datetime import datetime
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetModelStateRequest

from xbox_controller import xboxcontroller

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from math import sqrt, exp, isinf

from createSys import createSys
from constructPolicies import constructPolicies
from episodicREINFORCE import episodicREINFORCE
from episodicNaturalActorCritic import episodicNaturalActorCritic
from drawAction import drawAction
from initPGELLA import initPGELLA
from computeHessian import computeHessian
from updatePGELLA import updatePGELLA
from structs import Data, Hessianarray, Parameterarray, PGPolicy, Policy

import cPickle as pickle


class Agent():

    def __init__(self, n_systems, learning_rate, gamma):
        rospy.init_node('agent_pg_ella', anonymous=False)

        print "waiting for service"
        rospy.wait_for_service('/v_control/reset_world')
        print "done"

        self._n_systems = n_systems
        self._learning_rate = learning_rate
        self._gamma = gamma

        self._reset_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._enable_controller = rospy.Publisher('v_controller/move_enable', Bool)
        self._soft_reset_pub = rospy.Publisher('v_controller/soft_reset', Empty)
        self._takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty)
        self._action_pub = rospy.Publisher('v_controller/agent_cmd', Twist)
        self._state_sub = rospy.Subscriber('v_controller/state', Float32MultiArray, self.getState)
        self._visible_sub = rospy.Subscriber('v_controller/visible', Bool, self.visible_callback)
        self._threshold = rospy.get_param("v_controller/threshold")

    def startPg(self, poli_type='Gauss', base_learner='NAC',
                traj_length=100, num_rollouts=40, num_iterations=150,
                task_file='task.p', policy_file='policy.p',
                is_load=False):
        if not is_load:
            # Creating Tasks
            self.Tasks = createSys(self._n_systems, poli_type, base_learner, self._gamma)

            # Constructing policies
            self.Policies = constructPolicies(self.Tasks)

            # Calculating theta
            self.Policies = self.calcThetaStar(self.Tasks, self.Policies,
                                               self._learning_rate, traj_length,
                                               num_rollouts, num_iterations)

            self.savePolicies(task_file, policy_file)
        else:
            self.loadPolicies(task_file, policy_file)

    def savePolicies(self, task_file, policy_file):
        # Save PG policies and tasks to a file
        pickle.dump(self.Tasks, open(task_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Policies, open(policy_file, 'wb'), pickle.HIGHEST_PROTOCOL)

    def loadPolicies(self, task_file, policy_file):
        # Load PG policies and tasks to a file
        self.Tasks = pickle.load(open(task_file, 'rb'))
        self.Policies = pickle.load(open(policy_file, 'rb'))

    def getState(self, data):
        self._state_x = data.data[0]
        self._state_y = data.data[1]
        self._state_z = data.data[2] - 1.5 #dc bias so that it floats at 1.5 meters
        #print "x:%s y:%s z:%s" %(_state_x, _state_y, _state_z)

    def visible_callback(self, visible):
        self._visible = visible.data

    def resetSim(self, x, y, z=0, angle=0):
        self._enable_controller.publish(Bool(0))
        self._takeoff_pub.publish(Empty())
        rospy.sleep(.5)
        a = SetModelStateRequest()
        a.model_state.model_name = 'quadrotor'
        a.model_state.pose.position.z = 2 + z
        a.model_state.pose.position.x = 3 + x
        a.model_state.pose.position.y = 0 + y
        self._reset_pos(a)
        a.model_state.model_name = 'unit_sphere_4'
        a.model_state.pose.position.z = 0.052
        a.model_state.pose.position.x = 3
        a.model_state.pose.position.y = 0
        self._reset_pos(a)
        rospy.sleep(.5)
        self._soft_reset_pub.publish(Empty())

    def startEpisode(self):
        #x = random.uniform(-.23, .23)
        #y = random.uniform(-.23, .23)
        x = random.uniform(-.8, .8)
        y = random.uniform(-.23, .23)
        self.resetSim(x,y,0)
        rospy.sleep(1)


    def obtainData(self, policy, L, H, param):
        N = param.param.N
        M = param.param.M
        data = [Data(N, M, L) for i in range(H)]

        time.sleep(.5)


        # Based on Jan Peters; codes.
        # Perform H

        for trials in range(H):
            print "      Trial #", trials+1
            self.startEpisode()

            # Save associated policy
            data[trials].policy = policy

            # Draw the first state
            data[trials].x[:,0] = np.array([[-self._state_x/4.0, -self._state_y/3.0, self._state_z/1.5]])

            # Perform a trial of length L
            for steps in range(L):
                if rospy.is_shutdown():
                    sys.exit()
                # Draw an action from the policy
                xx = data[trials].x[:,steps]
                #print "current state: ", xx
                data[trials].u[:,steps] = drawAction(policy, xx, param)
                #print "action: ", data[trials].u[:,steps]


                if not self._visible:
                    data[trials].u[:,steps] *= 0.0
                else:
                    for j in range(M):
                        data[trials].u[:,steps][j] += param.param.disturbance[j]
                        data[trials].u[:,steps][j] = round(data[trials].u[:,steps][j], 5)

                        if data[trials].u[:,steps][j] > 1.0: # saturate
                            data[trials].u[:,steps][j] = 1.0
                        elif data[trials].u[:,steps][j] < -1.0:
                            data[trials].u[:,steps][j] = -1.0

                if self._visible and self._state_z > 2.8:
                    data[trials].u[:,steps][2] = -0.1
                '''
                if self._visible and xx[2] >= .9:
                    data[trials].u[:,steps][2] = -0.1
                elif self._visible and xx[2] <= -.7:
                    data[trials].u[:,steps][2] = 0.1
                '''


                command = Twist()
                command.linear.x = data[trials].u[:,steps][0]
                command.linear.y = data[trials].u[:,steps][1]
                command.linear.z = data[trials].u[:,steps][2]
                self._action_pub.publish(command)
                rospy.sleep(.2)

                #print "action: ", data[trials].u[:,steps]

                # Draw next state from environment
                state = np.array([[-self._state_x/4.0, -self._state_y/3.0, self._state_z/1.5]])
                data[trials].x[:,steps+1] = state

                # Obtain the reward
                u = data[trials].u[:,steps]
                u_p = u.conj().T
                #imp_eye_state = np.zeros((3,3))
                #imp_eye_act = np.zeros((3,3))
                #np.fill_diagonal(imp_eye_state, [10, 10, 5])
                #np.fill_diagonal(imp_eye_act, [5, 5, 2.5])
                #reward = -sqrt(np.dot(np.dot(state, imp_eye_state), state.conj().T)) - \
                #          sqrt(np.dot(np.dot(u, imp_eye_act), u_p))
                reward = -sqrt(np.dot(np.dot(state, np.eye(N) * 10), state.conj().T)) - \
                          sqrt(np.dot(np.dot(u, np.eye(M) * 5), u_p))
                #print "reward: ", reward

                if isinf(reward):
                    print "Error: INFINITY"
                    sys.exit(1)

                data[trials].r[0][steps] = reward

        return data


    def calcThetaStar(self, Params, Policies, rates,
                      trajlength, rollouts, numIterations):

        plt.ion()

        nSystems = np.shape(Params)[0]
        r = np.zeros(shape=(1, rollouts))

        tasks_time = [0] * nSystems

        for i in range(nSystems):
            # TODO: Clear screen
            print "@ Task: ", i
            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            ax.grid()

            policy = Policies[i].policy # Resetting policy IMP
            start_time = datetime.now()

            for k in range(numIterations):
                print "@ Iteration: ", k
                print "    Initial Theta: ", policy.theta
                print "    Sigma: ", policy.sigma
                print "    Learning Rate: ", rates
                data = self.obtainData(policy, trajlength, rollouts, Params[i])
                dJdtheta = None
                if Params[i].param.baseLearner == "REINFORCE":
                    dJdtheta = episodicREINFORCE(policy, data, Params[i])
                else:
                    dJdtheta = episodicNaturalActorCritic(policy, data, Params[i])

                policy.theta = policy.theta + rates*dJdtheta.reshape(9, 1)

                for z in range(rollouts):
                    r[0, z] = np.sum(data[z].r)

                if np.isnan(r.any()):
                    print "System has NAN"
                    print "exiting..."
                    import sys
                    sys.exit(1)

                print "Mean: ", np.mean(r)
                time.sleep(1)
                ax.scatter(k, np.mean(r), marker=u'x', c='blue', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.05)

            stop_time = datetime.now()
            tasks_time[i] = str(stop_time - start_time)

            Policies[i].policy = policy # Calculating theta
        print "Task completion times: ", tasks_time
        plt.show(block=True)

        return Policies

    def startElla(self, traj_length, num_rollouts, mu1, mu2, k=1):
        self.modelPGELLA = initPGELLA(self.Tasks, 1, mu1, mu2, self._learning_rate)

        print "Learn PGELLA"
        self.modelPGELLA = self.learnPGELLA(self.Tasks, self.Policies, self._learning_rate,
                                            traj_length, num_rollouts, self.modelPGELLA)

    def learnPGELLA(self, Tasks, Policies, learningRate,
                    trajLength, numRollouts, modelPGELLA):

        counter = 1
        tasks_size = np.shape(Tasks)[0]
        ObservedTasks = np.zeros((tasks_size, 1))
        limitOne = 0
        limitTwo = tasks_size


        HessianArray = [Hessianarray() for i in range(tasks_size)]
        ParameterArray = [Parameterarray() for i in range(tasks_size)]

        while not np.all(ObservedTasks):  # Repeat until all tasks are observed

            taskId = np.random.randint(limitOne, limitTwo, 1)  # Pick a random task
            print "Task ID: ", taskId

            if ObservedTasks[taskId] == 0:  # Entry is set to 1 when corresponding task is observed
                ObservedTasks[taskId] = 1

            # Policy Gradientts on taskId
            data = self.obtainData(Policies[taskId].policy, trajLength, numRollouts, Tasks[taskId])

            if Tasks[taskId].param.baseLearner == 'REINFORCE':
                dJdTheta = episodicREINFORCE(Policies[taskId].policy, data, Tasks[taskId])
            else:
                dJdTheta = episodicNaturalActorCritic(Policies[taskId].policy, data, Tasks[taskId])

            # Updating theta*
            Policies[taskId].policy.theta = Policies[taskId].policy.theta + learningRate * dJdTheta.reshape(9, 1)

            # Computing Hessian
            data = self.obtainData(Policies[taskId].policy, trajLength, numRollouts, Tasks[taskId])

            D = computeHessian(data, Policies[taskId].policy.sigma)
            HessianArray[taskId].D =  D

            ParameterArray[taskId].alpha = Policies[taskId].policy.theta

            # Perform Updating L and S
            modelPGELLA = updatePGELLA(modelPGELLA, taskId, ObservedTasks, HessianArray, ParameterArray)  # Perform PGELLA for that Group

            print 'Iterating @: ', counter
            counter = counter + 1

        print 'All Tasks observed @: ', counter-1

        return modelPGELLA

    def startTest(self, traj_length, num_rollouts, num_iterations):
        # Creating new PG policies
        PGPol = constructPolicies(self.Tasks)

        print "Test Phase"
        # Testing and comparing PG and PG-ELLA
        self.testPGELLA(self.Tasks, PGPol, self._learning_rate, traj_length,
                        num_rollouts, num_iterations, self.modelPGELLA)

    def testPGELLA(self, Tasks, PGPol, learningRate, trajLength,
                   numRollouts, numIterations, modelPGELLA):

        plt.ion()

        tasks_size = np.shape(Tasks)[0]

        PolicyPGELLAGroup = [PGPolicy() for i in range(tasks_size)]

        for i in range(tasks_size):  # loop over all tasks
            theta_PG_ELLA = modelPGELLA.L * modelPGELLA.S[:,i] # TODO: double check
            policyPGELLA = Policy()
            policyPGELLA.theta = theta_PG_ELLA
            policyPGELLA.sigma = PGPol[i].policy.sigma
            PolicyPGELLAGroup[i].policy = policyPGELLA

        Avg_rPGELLA = np.zeros((numIterations, tasks_size))
        Avg_rPG = np.zeros((numIterations, tasks_size))
        for k in range(tasks_size): # Test over all tasks
            print "@ Task: ", k
            fig = plt.figure(k)
            ax = fig.add_subplot(111)
            ax.grid()
            for m in range(numIterations): # Loop over Iterations
                print "    @ Iteration: ", m
                # PG
                data = self.obtainData(PGPol[k].policy, trajLength, numRollouts, Tasks[k])

                if Tasks[k].param.baseLearner == 'REINFORCE':
                    dJdTheta = episodicREINFORCE(PGPol[k].policy, data, Tasks[k])
                else:
                    dJdTheta = episodicNaturalActorCritic(PGPol[k].policy, data, Tasks[k])

                # Update Policy Parameters
                PGPol[k].policy.theta = PGPol[k].policy.theta + learningRate * dJdTheta.reshape(9, 1)
                print "PG policy: ", PGPol[k].policy.theta

                # PG-ELLA
                dataPGELLA = self.obtainData(PolicyPGELLAGroup[k].policy, trajLength, numRollouts, Tasks[k])

                if Tasks[k].param.baseLearner == 'REINFORCE':
                    dJdThetaPGELLA = episodicREINFORCE(PolicyPGELLAGroup[k].policy, dataPGELLA, Tasks[k])
                else:
                    dJdThetaPGELLA = episodicNaturalActorCritic(PolicyPGELLAGroup[k].policy, dataPGELLA, Tasks[k])

                # Update Policy Parameters
                PolicyPGELLAGroup[k].policy.theta = PolicyPGELLAGroup[k].policy.theta + learningRate * dJdThetaPGELLA.reshape(9, 1)
                print "PGELLA policy: ", PolicyPGELLAGroup[k].policy.theta

                # Computing Average in one System per iteration
                data_size = np.shape(data)[0]
                Sum_rPG = np.zeros((data_size, 1))
                for z in range(data_size):
                    Sum_rPG[z,:] = np.sum(data[z].r)

                dataPG_size = np.shape(dataPGELLA)[0]
                Sum_rPGELLA = np.zeros((dataPG_size, 1))
                for z in range(dataPG_size):
                    Sum_rPGELLA[z,:] = np.sum(dataPGELLA[z].r)

                Avg_rPGELLA[m, k] = np.mean(Sum_rPGELLA)
                Avg_rPG[m, k] = np.mean(Sum_rPG)

                # TODO: Plot graph
                ax.scatter(m, Avg_rPGELLA[m, k], marker=u'*', color='r', cmap=cm.jet)
                ax.scatter(m, Avg_rPG[m, k], marker=u'*', color='b', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()
        plt.show(block=True)


if __name__ == "__main__":
    n_systems = 2  # Integer number of tasks
    learning_rate = .35  # Learning rate for stochastic gradient descent
    gamma = 0.9  # Discount factor gamma

    # Parameters for policy
    poli_type = 'Gauss'  # Policy Type (Only supports Gaussian Policies)
                        # 'Gauss' => Gaussian Policy
    base_learner = 'NAC'  # Base Learner
                               # 'REINFORCE' => Episodic REINFORCE
                               # 'NAC' => Episodic Natural Actor Critic

    traj_length = 150 # Number of time steps to simulate in the cart-pole system
    num_rollouts = 40 # Number of trajectories for testing
    num_iterations = 120 # Number of learning episodes/iterations # 200

    agent = Agent(n_systems, learning_rate, gamma)
    time.sleep(.5)

    # Learning PG
    agent.startPg(poli_type, base_learner, traj_length, num_rollouts, num_iterations,
                   task_file='task.p', policy_file='policy.p', is_load=False)

    # Loading PG policies from file
    #agent.startPg(task_file='task.p', policy_file='policy.p', isLoadPolicy=True)


    # Learning ELLA
    traj_length = 150
    num_rollouts = 40 # 200
    mu1 = exp(-5)  # Sparsity coefficient
    mu2 = exp(-5)  # Regularization coefficient
    k = 1  # Number of inner layers

    agent.startElla(traj_length, num_rollouts, mu1, mu2, k)


    # Testing Phase
    traj_length = 100
    num_rollouts = 5 # 100
    num_iterations = 500 # 200

    agent.startTest(traj_length, num_rollouts, num_iterations)

    rospy.spin()