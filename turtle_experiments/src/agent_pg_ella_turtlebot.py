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
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import LinkStates

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from math import sqrt, exp, isinf
from sys import stdout
import os

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
        rospy.loginfo("To stop TurtleBot CTRL + C")

        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.globalstate_callback)

        self._n_systems = n_systems
        self._learning_rate = learning_rate
        self._gamma = gamma

        self._reset_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.r = rospy.Rate(10);
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0
        self.globalposition = [0,0,0]
        self.state = [0,0,0]
        self.goal = [1.0, -1.0, 0]
        #self.disturbance = np.random.uniform(-0.1,0.1,size=None)

        print "finished init" 
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
            if (np.shape(self.Avg_rPG)[1] != self._n_systems):
                import sys
                print "Error: File has different number of systems"
                sys.exit(1)

            # Continue Learning
            self.Policies, self.Avg_rPG = self.calcThetaStar(self.Tasks, self.Policies,
                                                        self._learning_rate, traj_length,
                                                        num_rollouts, num_iterations,
                                                        avgRPG=self.Avg_rPG)

            self.savePolicies(task_file, policy_file, avg_file)
        elif not is_load:
            # Creating Tasks #################################################NUMBER OF PARAMS
            self.Tasks = createSys(self._n_systems, poli_type, base_learner, self._gamma)

            # Constructing policies, random init
            self.Policies = constructPolicies(self.Tasks)

            # Calculating theta
            self.Policies, self.Avg_rPG = self.calcThetaStar(self.Tasks, self.Policies,
                                                             self._learning_rate, traj_length,
                                                             num_rollouts, num_iterations)

            self.savePolicies(task_file, policy_file, avg_file)
        else:
            self.loadPolicies(task_file, policy_file, avg_file)
        print "finished startPG"

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

    '''def getState(self, data): ######################################
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

        eps = .0001
        div = np.sign(alpha)*max(abs(alpha),eps)
        
        state = [r*np.cos(alpha), alpha, np.sin(div)*np.cos(div)/div*(div + theta) , 1]
    '''

    def visible_callback(self, visible):
        self._visible = visible.data

    def resetSim(self, x, y, z=0, angle=0):
        #self._enable_controller.publish(Bool(0))
        #self._takeoff_pub.publish(Empty())
        rospy.sleep(.5)
        a = SetModelStateRequest()
        a.model_state.model_name = 'mobile_base'
        a.model_state.pose.position.z = 0 
        a.model_state.pose.position.x = 0 
        a.model_state.pose.position.y = 0 
        a.model_state.pose.orientation.z = 0 
        a.model_state.pose.orientation.x = 0 
        a.model_state.pose.orientation.y = 0 
        a.model_state.pose.orientation.y = 1 
        self._reset_pos(a)
        rospy.sleep(.5)

    def startEpisode(self):
        #x = random.uniform(-.23, .23)
        #y = random.uniform(-.23, .23)
        x = random.uniform(-.8, .8)
        y = random.uniform(-.23, .23)
        self.resetSim(x,y,0)
        rospy.sleep(1)

    def land(self):
        self.land_pub.publish(Empty())

    def obtainData(self, policy, L, H, param):
        N = param.param.N
        M = param.param.M
        data = [Data(N, M, L) for i in range(H)]

        rospy.sleep(.5)


        # Based on Jan Peters; codes.
        # Perform H

        for trials in range(H): #rollout
            stdout.write("\r    Trial %d of %d" % (trials+1, H))
            stdout.flush()
            #print "      Trial #", trials+1
            self.startEpisode()

            # Save associated policy
            data[trials].policy = policy

            # Draw the first state #######################################################
            os.system("rosservice call /gazebo/set_model_state '{model_state: { model_name: mobile_base, pose: { position: { x: 0, y: 0 ,z: 0 }, orientation: {x: 0, y: 0.0, z: 0, w: 1 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'")
            rospy.sleep(.1)
            
            data[trials].x[:,0] = self.state 

            # Perform a trial of length L trajlength
            for steps in range(L): 
                if rospy.is_shutdown():
                    sys.exit()
                ''' Draw an action from the policy '''
                xx = data[trials].x[:,steps]
                #print "current state: ", xx
                data[trials].u[:,steps] = drawAction(policy, xx, param) + param.param.disturbance #####################
                #print "action: ", data[trials].u[:,steps]

                ''' Get next state '''
                command = Twist()
                self.move_cmd.linear.x = data[trials].u[:,steps][0]
                self.move_cmd.angular.z = data[trials].u[:,steps][1]
                #self.move_cmd.linear.x = action[0]#np.sign(action[0])*min(1, .1*abs(action[0]))  #action[0]
                #self.move_cmd.angular.z = action[1]#np.sign(action[1])*min(1, .1*abs(action[1])) #action[1]
                self.cmd_vel.publish(self.move_cmd)
                rospy.sleep(.2)
                data[trials].x[:,steps+1] = self.state


                ''' Obtain the reward '''
                u = data[trials].u[:,steps]
           
                #imp_eye_state = np.zeros((3,3))
                #imp_eye_act = np.zeros((3,3))
                #np.fill_diagonal(imp_eye_state, [10, 10, 5])
                #np.fill_diagonal(imp_eye_act, [5, 5, 2.5])
                #reward = -sqrt(np.dot(np.dot(state, imp_eye_state), state.conj().T)) - \
                #          sqrt(np.dot(np.dot(u, imp_eye_act), u_p))
		
		t1 = np.subtract(self.goal, self.globalposition)[1:2] #, np.eye(N) * 10)
		#t2 = np.array(self.state).conj().T
                #print t1 #,t2
                tmp1 = np.dot(t1, t1.conj().T)
                tmp2 = np.dot(u, u.conj().T)
                #print 'arg:',tmp1,tmp2
		#print 'sqrt:',sqrt(tmp1),sqrt(tmp2)
		
                goalweight = 10
                controlweight = 1
                reward = -goalweight*tmp1 -controlweight*tmp2
                #print "reward: ", reward, "  state:", tmp1, "  action",tmp2

                if isinf(reward):
                    print "Error: INFINITY"
                    sys.exit(1)

                data[trials].r[0][steps] = reward
        print
        return data


    def calcThetaStar(self, Params, Policies, rates,
                      trajlength, rollouts, numIterations,
                      avgRPG=None):
        plt.ion() #interactive plotting

        nSystems = np.shape(Params)[0]
        r = np.zeros(shape=(1, rollouts))

        tasks_time = [0] * nSystems
        start_it = 0
        if avgRPG == None:
            Avg_rPG = np.zeros((numIterations, nSystems))
        else:
            start_it = np.shape(avgRPG)[0]
            print "Start: ", start_it
            Avg_rPG = np.zeros((numIterations, nSystems))
            Avg_rPG = np.append(avgRPG, Avg_rPG, axis=0)
            numIterations += start_it

        for i in range(nSystems):
            # TODO: Clear screen
            print "@ Task: ", i
            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            ax.grid()
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Reward')

            if avgRPG != None:
                # plot the rewards from previous learning session
                for k in range(start_it):
                    ax.scatter(k, Avg_rPG[k, i], marker=u'x', c='cyan', cmap=cm.jet)
                    ax.figure.canvas.draw()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            policy = Policies[i].policy # Resetting policy IMP
	    
            start_time = datetime.now()

            for k in range(start_it, numIterations):
                print "@ Iteration: ", k
                print "    Initial Theta: ", policy.theta
                print "    Sigma: ", policy.sigma
                print "    Learning Rate: ", rates
                print "    Perturbation: ", Params[i].param.disturbance
		
		
                data = self.obtainData(policy, trajlength, rollouts, Params[i])
                dJdtheta = None
                if Params[i].param.baseLearner == "REINFORCE":
                    dJdtheta = episodicREINFORCE(policy, data, Params[i])
                else:
                    dJdtheta = episodicNaturalActorCritic(policy, data, Params[i])
                
                policy.theta = policy.theta + rates*dJdtheta.reshape(np.shape(policy.theta))  #########fixed reshape
                
                for z in range(rollouts):
                    r[0, z] = np.sum(data[z].r)

                if np.isnan(r.any()):
                    print "System has NAN"
                    print "exiting..."
                    import sys
                    sys.exit(1)

                avg_r = np.mean(r)
                print "    Mean: ", avg_r
                Avg_rPG[k, i] = np.mean(avg_r)

                rospy.sleep(1)
                ax.scatter(k, avg_r, marker=u'x', c='blue', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()
                rospy.sleep(0.05)

            stop_time = datetime.now()
            tasks_time[i] = str(stop_time - start_time)

            Policies[i].policy = policy # Calculating theta
            print "    Learned Theta: ", Policies[i].policy.theta
        print "Task completion times: ", tasks_time
        #plt.show(block=True)

        return Policies, Avg_rPG

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
            fig = plt.figure(k+1000)
            ax = fig.add_subplot(111)
            ax.grid()
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Reward')

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
        
        tmp = theta

        self.globalposition = [robox,roboy,tmp]
 
        e = np.subtract(self.goal, self.globalposition)
        r = np.sqrt(e[0]**2 + e[1]**2)
        psi = np.arctan2(e[1],e[0])
        beta = self.globalposition[2]
        kappa = self.goal[2]
        theta = (kappa+psi)
        alpha = (psi-beta)
        phi = (kappa+beta)

        eps = .0001
        div = np.sign(alpha)*max(abs(alpha),eps)
        
        self.state = [r*np.cos(alpha), alpha, np.sin(div)*np.cos(div)/div*(div + theta), 1]


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


if __name__ == "__main__":
    n_systems = 4  # Integer number of tasks 4
    learning_rate = 0.2  # Learning rate for stochastic gradient descent
    gamma = 0.99  # Discount factor gamma

    # Parameters for policy
    poli_type = 'Gauss'  # Policy Type (Only supports Gaussian Policies)
                        # 'Gauss' => Gaussian Policy
    base_learner = 'NAC'  # Base Learner
                               # 'REINFORCE' => Episodic REINFORCE
                               # 'NAC' => Episodic Natural Actor Critic

    traj_length = 100 # Number of time steps to simulate in the cart-pole system
    num_rollouts = 10 # Number of trajectories for testing
    num_iterations = 500 # Number of learning episodes/iterations # 120 600

    agent = Agent(n_systems, learning_rate, gamma)
    rospy.sleep(.5)
    # Learning PG
    agent.startPg(poli_type, base_learner, traj_length,
                  num_rollouts, num_iterations, task_file='task.p',
                  policy_file='policy.p', avg_file='average.p',
                  is_load=False)

    # Continue Learning PG
    # NOTE: Make a Backup of the files before running to ensure
    #       you have a copy of the original policy
    #agent.startPg(poli_type, base_learner, traj_length,
    #          num_rollouts, num_iterations, task_file='task.p',
    #          policy_file='policy.p', avg_file='average.p',
    #          is_continue=True)

    # Loading PG policies from file
    #agent.startPg(task_file='task.p', policy_file='policy.p', isLoadPolicy=True)

    '''
    # Learning ELLA
    traj_length = 150
    num_rollouts = 40 # 200
    mu1 = exp(-5)  # Sparsity coefficient
    mu2 = exp(-5)  # Regularization coefficient
    k = 1  # Number of inner layers

    agent.startElla(traj_length, num_rollouts, mu1, mu2, k)


    # Testing Phase
    traj_length = 150
    num_rollouts = 40 # 100
    num_iterations = 200 # 200

    agent.startTest(traj_length, num_rollouts, num_iterations)
    '''

    rospy.spin()
