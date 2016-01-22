#!/usr/bin/env python
from __future__ import print_function
import rospy
import time
import sys
import types
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
import copy
from matplotlib import cm
from math import sqrt, exp, isinf
from sys import stdout

from createSys import createSys
from constructPolicies import constructPolicies
from episodicREINFORCE import episodicREINFORCE
from episodicNaturalActorCritic import episodicNaturalActorCritic
from finiteDifferences import finiteDifferences
from drawAction import drawAction
from initPGELLA import initPGELLA
from computeHessian import computeHessian
from updatePGELLA import updatePGELLA
from structs import Data, Hessianarray, Parameterarray, PGPolicy, Policy

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Agent():

    def __init__(self, n_systems, learning_rate, gamma):
        rospy.init_node('agent_pg_ella', anonymous=False)

        print("waiting for service")
        rospy.wait_for_service('/v_control/reset_world')
        print("done")

        self._n_systems = n_systems
        self._learning_rate = learning_rate
        self._gamma = gamma

        self._reset_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._enable_controller = rospy.Publisher('v_controller/move_enable', Bool, queue_size=10)
        self._soft_reset_pub = rospy.Publisher('v_controller/soft_reset', Empty, queue_size=10)
        self._takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=10)
        self._land_pub = rospy.Publisher('/ardrone/land', Empty, queue_size=10)
        self._action_pub = rospy.Publisher('v_controller/agent_cmd', Twist, queue_size=10)
        self._state_sub = rospy.Subscriber('v_controller/state', Float32MultiArray, self.getState)
        self._visible_sub = rospy.Subscriber('v_controller/visible', Bool, self.visible_callback)
        self._threshold = rospy.get_param("v_controller/threshold")

        self._hover_height = .4

    def startPg(self, poli_type='Gauss', base_learner='NAC',
                traj_length=100, num_rollouts=40, num_iterations=150,
                task_file='task.p', policy_file='policy.p',
                avg_file='average.p', is_continue=False, is_load=False):
        if is_continue:
            # Continue learning from a loaded policy
            # num_iterations treated as additional iterations

            # Load policy from file
            self.loadPolicies(task_file, policy_file, avg_file)

            # Ensure number of systems are equal
            if (self.Tasks[0].nSystems != self._n_systems):
                import sys
                print("Error: File has different number of systems")
                sys.exit(1)

            # Continue Learning
            self.Policies, self.Avg_rPG = self.calcThetaStar(self.Tasks, self.Policies,
                                                        self._learning_rate, traj_length,
                                                        num_rollouts, num_iterations,
                                                        avgRPG=self.Avg_rPG)

            self.savePolicies(task_file, policy_file, avg_file)
        elif not is_load:
            # Creating Tasks
            self.Tasks = createSys(self._n_systems, poli_type, base_learner, self._gamma)

            # Constructing policies
            self.Policies = constructPolicies(self.Tasks)

            # Calculating theta
            self.Policies, self.Avg_rPG = self.calcThetaStar(self.Tasks, self.Policies,
                                                             self._learning_rate, traj_length,
                                                             num_rollouts, num_iterations)

            self.savePolicies(task_file, policy_file, avg_file)
        else:
            self.loadPolicies(task_file, policy_file, avg_file)

    def savePolicies(self, task_file, policy_file, avg_file):
        # Save PG policies and tasks to a file
        pickle.dump(self.Tasks, open(task_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Policies, open(policy_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Avg_rPG, open(avg_file, 'wb'), pickle.HIGHEST_PROTOCOL)

    def loadPolicies(self, task_file, policy_file, avg_file):
        # Load PG policies and tasks to a file
        self.Tasks = pickle.load(open(task_file, 'rb'))
        self.Policies = pickle.load(open(policy_file, 'rb'))
        self.Avg_rPG = pickle.load(open(avg_file, 'rb'))

    def getState(self, data):
        self._state_x = data.data[0]
        self._state_y = data.data[1]
        self._state_z = data.data[2] - self._hover_height #dc bias so that it floats at .240 meters
        # .235 ultrasonic read still works, using .240
        # -.240 is too low

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
        #x = 0
        #y = 0
        #self.resetSim(x,y,0)
        while True:
            x = random.uniform(-.8, .8)
            y = random.uniform(-.23, .23)
            self.resetSim(x,y,0)
            rospy.sleep(1)
            if self._visible:
                return

    def land(self):
        self.land_pub.publish(Empty())

    def obtainData(self, policy, L, H, param):
        N = param.param.N
        M = param.param.M
        data = [Data(N, M, L) for i in range(H)]

        time.sleep(.5)


        # Based on Jan Peters; codes.
        # Perform H
        noiseval = []
        landed = 0

        stdout.write("\r    Trial %2d of %2d | Drone Landed %2d of %2d" % (0, H, 0, H))
        stdout.flush()
        for trials in range(H):
            #print("      Trial #", trials+1)
            self.startEpisode()

            # Save associated policy
            data[trials].policy = copy.deepcopy(policy)

            # Finite Differences
            if trials != 0:
                if np.mod(trials, 2) == 1:
                    noiseval = np.random.uniform(-0.0001, 0.0001, np.shape(data[trials].policy.theta)) + 0.0
                else:
                    noiseval = -noiseval + 0.0
                data[trials].policy.theta = copy.deepcopy(data[0].policy.theta) + noiseval


            # Draw the first state
            data[trials].x[:,0] = np.array([[-self._state_x/4.0, -self._state_y/3.0, self._state_z/3.0]])

            hasLanded = False

            # Perform a trial of length L
            for steps in range(L):
                if rospy.is_shutdown():
                    sys.exit()
                # Draw an action from the policy
                xx = copy.deepcopy(data[trials].x[:,steps])
                #print("current state: ", xx)
                data[trials].u[:,steps] = drawAction(policy, xx, param)
                #print("action: ", data[trials].u[:,steps])


                if not self._visible or hasLanded:
                    data[trials].u[:,steps] *= 0.0
                else:
                    for j in range(M):
                        data[trials].u[:,steps][j] += param.param.disturbance[j]
                        data[trials].u[:,steps][j] = round(data[trials].u[:,steps][j], 5)

                        if data[trials].u[:,steps][j] > 1.0: # saturate
                            data[trials].u[:,steps][j] = 1.0
                        elif data[trials].u[:,steps][j] < -1.0:
                            data[trials].u[:,steps][j] = -1.0

                # ensures quadrotor doesn't go off
                # the max altitude of 3.0
                if self._visible and self._state_z >= (2.8 - self._hover_height): #2.8
                    data[trials].u[:,steps][2] = -0.1

                if not hasLanded:
                    if self._state_z <= 0 and self._visible and \
                       self._state_x < .7 and self._state_x > -.7 and \
                       self._state_y < .6 and self._state_y > -.8:
                        data[trials].u[:,steps][0] = 0.0
                        data[trials].u[:,steps][1] = 0.0
                        data[trials].u[:,steps][2] = 0.0
                        hasLanded = True
                        self._land_pub.publish(Empty())
                        landed += 1
                        rospy.sleep(3)
                    else:
                        command = Twist()
                        command.linear.x = data[trials].u[:,steps][0]
                        command.linear.y = data[trials].u[:,steps][1]
                        command.linear.z = data[trials].u[:,steps][2]
                        self._action_pub.publish(command)
                        rospy.sleep(.2)

                #print("action: ", data[trials].u[:,steps])

                # Draw next state from environment
                if hasLanded:
                    state = np.array([[0.0, 0.0, 0.0]])
                else:
                    #data[trials].x[:,steps+1] = drawNextState(data[trials].x[:,steps], data[trials].u[:,steps], param, i)
                    state = np.array([[-self._state_x/4.0, -self._state_y/3.0, self._state_z/3.0]])
                    #state = np.array([[-_state_x/4.0, -_state_y/3.0]])

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

                if isinf(reward):
                    print("Error: INFINITY")
                    sys.exit(1)

                data[trials].r[0][steps] = reward
            stdout.write("\r    Trial %2d of %2d | Drone Landed %2d of %2d" % (trials+1, H, landed, H))
            stdout.flush()
        print()
        return data


    def calcThetaStar(self, Params, Policies, rates,
                      trajlength, rollouts, numIterations,
                      avgRPG=None):

        plt.ion()

        nSystems = self._n_systems
        print("Number of Systems", nSystems)
        r = np.zeros(shape=(1, rollouts))

        tasks_time = [0] * nSystems
        start_it = [0] * nSystems

        if type(avgRPG) == types.NoneType:
            Avg_rPG = [np.zeros((numIterations, 1)) for i in range(nSystems)]
            #Avg_rPG = np.zeros((numIterations, nSystems))
        elif type(avgRPG) == np.ndarray:
            #Convert numpy array to list
            tmp_avg_r = []
            n_iterations = np.shape(avgRPG)[0]
            for j in range(nSystems):
                start_it[j] = n_iterations
                tmp = np.append(avgRPG[:,j].reshape(n_iterations, 1),
                                np.zeros((numIterations, 1)), axis=0)
                tmp_avg_r.append(tmp)
            Avg_rPG = tmp_avg_r
        elif type(avgRPG) == list:
            for j in range(nSystems):
                start_it[j] = np.shape(avgRPG[j])[0]
                avgRPG[j] = np.append(avgRPG[j],
                                      np.zeros((numIterations, 1)), axis=0)
            Avg_rPG = avgRPG

        for i in range(nSystems):
            # TODO: Clear screen
            print("@ Task: ", i + 1)
            print("    Perturbation: ", Params[i].param.disturbance)
            fig = plt.figure("PG Task " + str(i+1))
            ax = fig.add_subplot(111)
            ax.grid()
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Reward')

            if type(avgRPG) != types.NoneType:
                # plot the rewards from previous learning session
                for k in range(start_it[i]):
                    ax.scatter(k, Avg_rPG[i][k], marker=u'x', c='green', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()

            policy = copy.deepcopy(Policies[i].policy) # Resetting policy IMP
            start_time = datetime.now()

            for k in range(start_it[i], numIterations+start_it[i]):
                print("@ Iteration: ", k)
                print("    Initial Theta: ", policy.theta)
                print("    Sigma: ", policy.sigma)
                print("    Learning Rate: ", rates)
                print("    Perturbation: ", Params[i].param.disturbance)
                data = self.obtainData(policy, trajlength, rollouts, Params[i])
                dJdtheta = None
                if Params[i].param.baseLearner == "REINFORCE":
                    dJdtheta = episodicREINFORCE(policy, data, Params[i])
                elif Params[i].param.baseLearner == "NAC":
                    dJdtheta = episodicNaturalActorCritic(policy, data, Params[i])
                elif Params[i].param.baseLearner == "FD":
                    dJdtheta = finiteDifferences(data, rollouts)

                policy.theta = policy.theta + rates*dJdtheta.reshape(9, 1)

                for z in range(rollouts):
                    r[0, z] = np.sum(data[z].r)

                if np.isnan(r.any()):
                    print("System has NAN")
                    print("exiting...")
                    import sys
                    sys.exit(1)

                avg_r = np.mean(r)
                print("    Mean: ", avg_r)
                Avg_rPG[i][k] = np.mean(avg_r)

                time.sleep(1)
                ax.scatter(k, avg_r, marker=u'x', c='blue', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.05)

            stop_time = datetime.now()
            tasks_time[i] = str(stop_time - start_time)

            Policies[i].policy = policy # Calculating theta
            print("    Learned Theta: ", Policies[i].policy.theta)
            plt.savefig("PG Task " + str(i+1) + ".png", bbox_inches='tight')
            plt.close(fig)
        print("Task completion times: ", tasks_time)
        #plt.show(block=True)

        return Policies, Avg_rPG

    def startElla(self, traj_length=100, num_rollouts=40,
                  learning_rate=0.1, mu1=0.1, mu2=0.1, k=1,
                  model_file='model.p', is_load=False):
        if not is_load:
            self._learning_rate = learning_rate
            self.modelPGELLA = initPGELLA(self.Tasks, k, mu1, mu2, self._learning_rate)

            print("Learn PGELLA")
            #self.Policies = constructPolicies(self.Tasks) # NEW -> learn from random policies
            self.modelPGELLA = self.learnPGELLA(self.Tasks, self.Policies, self._learning_rate,
                                                traj_length, num_rollouts, self.modelPGELLA)

            self.saveModelPGELLA(model_file)
        else:
            self.loadModelPGELLA(model_file)

    def saveModelPGELLA(self, model_file):
        # Save PG policies and tasks to a file
        pickle.dump(self.modelPGELLA, open(model_file, 'wb'), pickle.HIGHEST_PROTOCOL)

    def loadModelPGELLA(self, model_file):
        # Load PG policies and tasks to a file
        self.modelPGELLA = pickle.load(open(model_file, 'rb'))

    def getSeedRandom(self, size, n_sets):
        self.randNumbers = []
        for i in range(n_sets):
            temp = range(size)
            random.shuffle(temp)
            self.randNumbers += temp

    def getNextRandom(self):
        if len(self.randNumbers) > 0:
            return self.randNumbers.pop(0)
        return None

    def learnPGELLA(self, Tasks, Policies, learningRate,
                    trajLength, numRollouts, modelPGELLA):
        counter = 1
        tasks_size = np.shape(Tasks)[0]
        ObservedTasks = np.zeros((tasks_size, 1))
        limitOne = 0
        limitTwo = tasks_size


        HessianArray = [Hessianarray() for i in range(tasks_size)]
        ParameterArray = [Parameterarray() for i in range(tasks_size)]

        #self.getSeedRandom(tasks_size, 4) # each task will be observed two times
        #taskId = 0 # FIXME to get rid of random choice of task
        while not np.all(ObservedTasks):  # Repeat until all tasks are observed
        #while len(self.randNumbers) > 0:

            taskId = np.random.randint(limitOne, limitTwo, 1)  # Pick a random task
            #taskId = self.getNextRandom()
            #if counter == 1:
            #    self.randNumbers.append(taskId)
            print("Task ID: ", taskId)

            if ObservedTasks[taskId] == 0:  # Entry is set to 1 when corresponding task is observed
                ObservedTasks[taskId] = 1

            # Policy Gradientts on taskId
            print("    Learn Theta")
            data = self.obtainData(Policies[taskId].policy, trajLength, numRollouts, Tasks[taskId])

            if Tasks[taskId].param.baseLearner == 'REINFORCE':
                dJdTheta = episodicREINFORCE(Policies[taskId].policy, data, Tasks[taskId])
            elif Tasks[taskId].param.baseLearner == 'NAC':
                dJdTheta = episodicNaturalActorCritic(Policies[taskId].policy, data, Tasks[taskId])
            elif Tasks[taskId].param.baseLearner == 'FD':
                dJdTheta = finiteDifferences(data, numRollouts)

            # Updating theta*
            Policies[taskId].policy.theta = Policies[taskId].policy.theta + learningRate * dJdTheta.reshape(9, 1)

            if False:
                # Computing Hessian
                print("    Data for Hessian")
                data = self.obtainData(Policies[taskId].policy, trajLength, numRollouts, Tasks[taskId])

                D = computeHessian(data, Policies[taskId].policy.sigma)
            else:
                # Use the identity matrix
                D = np.identity(np.shape(dJdtheta)[0])

            HessianArray[taskId].D =  D

            ParameterArray[taskId].alpha = copy.deepcopy(Policies[taskId].policy.theta)

            # Perform Updating L and S
            modelPGELLA = updatePGELLA(modelPGELLA, taskId, ObservedTasks, HessianArray, ParameterArray)  # Perform PGELLA for that Group

            print("Iterating @: ", counter)
            counter = counter + 1

            #taskId += 1 # FIXME to get rid of random choice of task

        print("All Tasks observed @: ", counter-1)

        return modelPGELLA

    def startTest(self, traj_length=100, num_rollouts=40,
                  num_iterations=150, learning_rate=0.1,
                  test_file='test.p', is_load=False, is_continue=False):
        if is_continue:
            # Continue learning from a loaded policy
            # num_iterations treated as additional iterations

            # Load policy from file
            self.loadModelTest(test_file)
            avg_RPGELLA = self.Test_Avg_rPGELLA
            avg_RPG = self.Test_Avg_rPG

            # Ensure number of systems are equal
            if (self.Tasks[0].nSystems != self._n_systems):
                import sys
                print("Error: File has different number of systems")
                sys.exit(1)

        elif not is_load:
            self._learning_rate = learning_rate
            # Creating new PG policies
            #self.PGPol = constructPolicies(self.Tasks)
            self.PGPol = self.Policies # NEW

            self.PolicyPGELLAGroup = [PGPolicy() for i in range(self._n_systems)]

            for i in range(self._n_systems):  # loop over all tasks
                theta_PG_ELLA = np.dot(self.modelPGELLA.L, self.modelPGELLA.S[:,i].reshape(self.modelPGELLA.k, 1))
                policyPGELLA = Policy()
                policyPGELLA.theta = theta_PG_ELLA
                policyPGELLA.sigma = self.PGPol[i].policy.sigma
                self.PolicyPGELLAGroup[i].policy = policyPGELLA

            avg_RPGELLA = self.Test_Avg_rPGELLA = None
            avg_RPG = self.Test_Avg_rPG = None
            self.saveModelTest(test_file)

        else:
            self.loadModelTest(test_file)
            avg_RPGELLA = self.Test_Avg_rPGELLA
            avg_RPG = self.Test_Avg_rPG

        print("Test Phase")
        # Testing and comparing PG and PG-ELLA
        self.testPGELLA(self.Tasks, self.PGPol,
                        self._learning_rate, traj_length, num_rollouts,
                        num_iterations, self.PolicyPGELLAGroup,
                        avgRPGELLA=avg_RPGELLA, avgRPG=avg_RPG)
        #print(self.PGPol)
        #print(self.Test_Avg_rPG)
        #print(self.PolicyPGELLAGroup)
        #print(self.Test_Avg_rPGELLA)
        self.saveModelTest(test_file)

    def saveModelTest(self, test_file):
        test = {}
        test['alpha'] = self._learning_rate
        test['pg_pol'] = self.PGPol
        test['pgella_pol'] = self.PolicyPGELLAGroup
        test['avg_RPG'] = self.Test_Avg_rPG
        test['avg_RPGELLA'] = self.Test_Avg_rPGELLA
        # Save PG policies and tasks to a file
        pickle.dump(test, open(test_file, 'wb'), pickle.HIGHEST_PROTOCOL)

    def loadModelTest(self, test_file):
        # Load PG policies and tasks to a file
        test = pickle.load(open(test_file, 'rb'))
        self._learning_rate = test['alpha']
        self.PGPol = test['pg_pol']
        self.PolicyPGELLAGroup = test['pgella_pol']
        self.Test_Avg_rPG = test['avg_RPG']
        self.Test_Avg_rPGELLA = test['avg_RPGELLA']


    def testPGELLA(self, Tasks, PGPol, learningRate, trajLength,
                   numRollouts, numIterations, PolicyPGELLAGroup,
                   avgRPGELLA=None, avgRPG=None):

        plt.ion()

        tasks_size = self._n_systems

        tasks_time = [0] * tasks_size
        start_it = [0] * tasks_size

        if type(avgRPGELLA) == types.NoneType or type(avgRPG) == types.NoneType:
            Test_Avg_rPGELLA = [np.zeros((numIterations, 1)) for i in range(tasks_size)]
            Test_Avg_rPG = [np.zeros((numIterations, 1)) for i in range(tasks_size)]
        elif type(avgRPGELLA) == list or type(avgRPG) == list:
            for j in range(tasks_size):
                start_it[j] = np.shape(avgRPG[j])[0]
                avgRPG[j] = np.append(avgRPG[j],
                                      np.zeros((numIterations, 1)), axis=0)
                avgRPGELLA[j] = np.append(avgRPGELLA[j],
                                      np.zeros((numIterations, 1)), axis=0)
            Test_Avg_rPG = avgRPG
            Test_Avg_rPGELLA = avgRPGELLA
        #Avg_rPGELLA = np.zeros((numIterations, tasks_size))
        #Avg_rPG = np.zeros((numIterations, tasks_size))
        for k in range(tasks_size): # Test over all tasks
            print("@ Task: ", k + 1)
            fig = plt.figure("PG-ELLA Task " + str(k+1))
            ax = fig.add_subplot(111)
            ax.grid()
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Reward')

            if type(avgRPG) != types.NoneType:
                # plot the rewards from previous learning session
                for m in range(start_it[k]):
                    ax.scatter(m, Test_Avg_rPGELLA[k][m], marker=u'*', color='r', cmap=cm.jet)
                    ax.scatter(m, Test_Avg_rPG[k][m], marker=u'*', color='b', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()

            start_time = datetime.now()

            for m in range(start_it[k], numIterations+start_it[k]): # Loop over Iterations
                print("    @ Iteration: ", m+1)
                # PG
                data = self.obtainData(PGPol[k].policy, trajLength, numRollouts, Tasks[k])

                if Tasks[k].param.baseLearner == 'REINFORCE':
                    dJdTheta = episodicREINFORCE(PGPol[k].policy, data, Tasks[k])
                elif Tasks[k].param.baseLearner == 'NAC':
                    dJdTheta = episodicNaturalActorCritic(PGPol[k].policy, data, Tasks[k])
                elif Tasks[k].param.baseLearner == 'FD':
                    dJdTheta = finiteDifferences(data, numRollouts)

                # Update Policy Parameters
                PGPol[k].policy.theta = PGPol[k].policy.theta + learningRate * dJdTheta.reshape(9, 1)
                print("PG policy: ", PGPol[k].policy.theta)

                # PG-ELLA
                dataPGELLA = self.obtainData(PolicyPGELLAGroup[k].policy, trajLength, numRollouts, Tasks[k])

                if Tasks[k].param.baseLearner == 'REINFORCE':
                    dJdThetaPGELLA = episodicREINFORCE(PolicyPGELLAGroup[k].policy, dataPGELLA, Tasks[k])
                elif Tasks[k].param.baseLearner == 'NAC':
                    dJdThetaPGELLA = episodicNaturalActorCritic(PolicyPGELLAGroup[k].policy, dataPGELLA, Tasks[k])
                elif Tasks[k].param.baseLearner == 'FD':
                    dJdThetaPGELLA = finiteDifferences(dataPGELLA, numRollouts)

                # Update Policy Parameters
                PolicyPGELLAGroup[k].policy.theta = PolicyPGELLAGroup[k].policy.theta + learningRate * dJdThetaPGELLA.reshape(9, 1)
                print("PGELLA policy: ", PolicyPGELLAGroup[k].policy.theta)

                # Computing Average in one System per iteration
                data_size = np.shape(data)[0]
                Sum_rPG = np.zeros((data_size, 1))
                for z in range(data_size):
                    Sum_rPG[z,:] = np.sum(data[z].r)

                dataPG_size = np.shape(dataPGELLA)[0]
                Sum_rPGELLA = np.zeros((dataPG_size, 1))
                for z in range(dataPG_size):
                    Sum_rPGELLA[z,:] = np.sum(dataPGELLA[z].r)

                Test_Avg_rPGELLA[k][m] = np.mean(Sum_rPGELLA)
                Test_Avg_rPG[k][m] = np.mean(Sum_rPG)

                # Plot graph
                ax.scatter(m, Test_Avg_rPGELLA[k][m], marker=u'*', color='r', cmap=cm.jet)
                ax.scatter(m, Test_Avg_rPG[k][m], marker=u'*', color='b', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()

            stop_time = datetime.now()
            tasks_time[k] = str(stop_time - start_time)

            plt.savefig("PG-ELLA Task " + str(k+1) + ".png", bbox_inches='tight')
            plt.close(fig)

        print("Task completion times: ", tasks_time)
        self.PGPol = PGPol
        self.Test_Avg_rPG = Test_Avg_rPG
        self.PolicyPGELLAGroup = PolicyPGELLAGroup
        self.Test_Avg_rPGELLA = Test_Avg_rPGELLA

if __name__ == "__main__":
    np.random.seed(10)
    n_systems = 5  # Integer number of tasks 4
    learning_rate = 0.000001 #.05  # Learning rate for stochastic gradient descent
    gamma = 0.998 #.9 # Discount factor gamma

    # Parameters for policy
    poli_type = 'Gauss'  # Policy Type (Only supports Gaussian Policies)
                        # 'Gauss' => Gaussian Policy
    base_learner = 'FD'  # Base Learner
                          # 'REINFORCE' => Episodic REINFORCE
                          # 'NAC' => Episodic Natural Actor Critic
                          # 'FD' => Finite Differences

    traj_length = 150 # Number of time steps to simulate in the cart-pole system
    num_rollouts = 15 #40 # Number of trajectories for testing
    num_iterations = 50 # Number of learning episodes/iterations # 120 600

    agent = Agent(n_systems, learning_rate, gamma)
    time.sleep(.5)

    # Learning PG
    agent.startPg(poli_type, base_learner, traj_length,
                  num_rollouts, num_iterations, task_file='task_new_50.p',
                  policy_file='policy_new_50.p', avg_file='average_new_50.p',
                  is_load=False)

    # Continue Learning PG
    # NOTE: Make a Backup of the files before running to ensure
    #       you have a copy of the original policy
    #agent.startPg(poli_type, base_learner, traj_length,
    #          num_rollouts, num_iterations, task_file='task_new_50.p',
    #          policy_file='policy_new_50.p', avg_file='average_new_50.p',
    #          is_continue=True)

    # Loading PG policies from file
    #agent.startPg(task_file='task_new_50.p', policy_file='policy_new_50.p',
    #              avg_file='average_new_50.p', is_load=True)

    '''
    # Learning ELLA
    traj_length = 150
    num_rollouts = 40 # 200
    learning_rate = 0.000001 #.00003
    mu1 = 0.0001 #exp(-3)  # Sparsity coefficient
    mu2 = 0.00000001 #exp(-3)  # Regularization coefficient
    k = 4 #2 # Number of inner layers

    # Learning PGELLA
    agent.startElla(traj_length, num_rollouts, learning_rate, mu1, mu2, k,
                    model_file='model_new_50.p', is_load=False)
    # Load PGELLA Model from file
    #agent.startElla(model_file='model_10_rand_pol.p', is_load=True)

    # Testing Phase
    traj_length = 150
    num_rollouts = 45 #40 # 100
    num_iterations = 10 # 200
    learning_rate = 0.03 #.05

    agent.startTest(traj_length, num_rollouts, num_iterations, learning_rate,
                    test_file='test_new_50.p', is_load=False)

    #agent.startTest(traj_length, num_rollouts, num_iterations, learning_rate,
    #                test_file='test_10_rand_pol.p', is_continue=True)
    '''
    rospy.spin()
