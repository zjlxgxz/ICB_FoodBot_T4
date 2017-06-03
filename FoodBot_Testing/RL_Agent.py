from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
import sys

import math
import time
import json


sys.path.append('../FoodBot_GRPC_Server/')
import FoodBotRLAgent_pb2
import grpc
from concurrent import futures


#from gridworld import gameEnv
#env = gameEnv(partial=False,size=5)

class FoodBotRLAgent(FoodBotRLAgent_pb2.FoodBotRLRequestServicer):
  def GetRLResponse (self, request, context):
      #do something
      currentState = list(request.currentState) #11-D
      formerState = list(request.formerState)
      rewardForTheFormer = request.rewardForTheFormer
      #0 1 2 3
      #0 2 5 10
      if(rewardForTheFormer == 0):
          rewardForTheFormer = 0
      elif  (rewardForTheFormer == 2):
          rewardForTheFormer = 20  
      elif  (rewardForTheFormer == 5):
          rewardForTheFormer = 50
      elif  (rewardForTheFormer == 10):
          rewardForTheFormer = 100
      formerAction = request.formerAction
      # Runmodel has check the start state..
      #if len(set(formerState)) ==1 and formerState[0]!=0:#[0,0,0,0,...] [1,1,1,1,1,...]
      #    formerAction = -1
      #if ((QTable[indexOfState(currentState),9]!=0 and currentState == [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) or formerState == [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] ):
      #if True:
      #  print ("============================================================")
      #def hasNewTurn(formerAction,formerReward,currentState,d,formerState):
      policy = hasNewTurn(formerAction,rewardForTheFormer,currentState,False,formerState) 
      if formerAction == 9:
      #if True:
        print ("NowQTable:",QTable[indexOfState(currentState),])
        print ("NowAction: ",policy)   
        print ("currentState: ",currentState)
        print ("formerState: ",formerState)
        print ("rewardForTheFormer: ",rewardForTheFormer)
        print ("formerAction: ",formerAction)
        print ("============================================================")
      return FoodBotRLAgent_pb2.Policy(policyNumber = policy)


class Qnetwork():
    def __init__(self):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.scalarInput =  tf.placeholder(shape=[None,11],dtype=tf.float32)

        #Just use one or two layer of fc
        self.dense1 = slim.fully_connected(self.scalarInput,1024,activation_fn=tf.nn.relu)
        #self.dropout1 = tf.layers.dropout(inputs=self.dense1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
        
        self.dense2 = slim.fully_connected(self.dense1, 1024, activation_fn=tf.nn.relu)
        #self.dropout2 = tf.layers.dropout(inputs=self.dense2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

        self.dense3 = slim.fully_connected(self.dense2, 11, activation_fn=tf.nn.relu)

        self.Qout = self.dense3
        self.predict = tf.argmax(self.Qout,1)
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,11,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])

def processState(states):
    return np.reshape(states,[21168])

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10 #How many episodes of game environment to train network with.
pre_train_steps = 100 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

#global variable
sess = tf.InteractiveSession()
#tf.reset_default_graph()
mainQN = Qnetwork()
targetQN = Qnetwork()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)
myBuffer = experience_buffer()
sess.run(init)

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps
#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0
#Make a path for our model to be saved in.
#if not os.path.exists(path):
#    os.makedirs(path)
#sess.run(init)
#if load_model == True:
#    print('Loading Model...')
#    ckpt = tf.train.get_checkpoint_state(path)
#    saver.restore(sess,ckpt.model_checkpoint_path)
#updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.

QTable = np.zeros([2**11,10])


episodeBuffer = experience_buffer()
#Reset environment and get first new observation
s = [2,2,2,2,2,2,2,2,2,2,2]
#s = processState(s)
d = False
rAll = 0
j = 0
diagNumber = 0

def indexOfState(state):
    index = 0
    for i in range(len(state)):
        index = (2**i)*state[i]
    if state == [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]:
        index = 2047
    if state == [2,2,2,2,2,2,2,2,2,2,2]:
        index = 0
    return int(index)

def newDialogSetup():
    global episodeBuffer,s,d,rAll,j

    #pisodeBuffer = experience_buffer()
    #Reset environment and get first new observation
    s = [2,2,2,2,2,2,2,2,2,2,2]
    #s = processState(s)
    d = False
    rAll = 0
    j = 0

def newDialogSetupDoubleQNN():
    global episodeBuffer,s,d,rAll,j

    episodeBuffer = experience_buffer()
    #Reset environment and get first new observation
    s = [2,2,2,2,2,2,2,2,2,2,2]
    #s = processState(s)
    d = False
    rAll = 0
    j = 0
    a = -1

def hasNewTurn(formerAction,formerReward,currentState,d,formerState):
    lr = 0.8
    y = 0.9
    s = formerState
    a = formerAction
    r = formerReward
    #r = 1
    s1 = currentState
    d = False # The termination indiction 
    #In our case, the termiantion states are: [0,0,0,0,0,0,0,0,0,0,0],[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    if currentState == [0,0,0,0,0,0,0,0,0,0,0] or currentState == [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]:
        d = True

    global j,total_steps,episodeBuffer,mainQN,targetQN,e,diagNumber,rAll,QTable
    j+=1
    total_steps = total_steps+1
    
    currentStateIndex = indexOfState(s1)
    if(a != -1):# start state: 22222 won't be accounted.
        formerStateIndex = indexOfState(s)
        QTable[formerStateIndex,a] = QTable[formerStateIndex,a] + lr*(r + y*np.max(QTable[currentStateIndex,:]) - QTable[formerStateIndex,a])
        #print (QTable[formerStateIndex,])
    print("dailog total turn,total turn",j,total_steps)
    #Choose an action by greedily (with e chance of random action) from the Q-network

    if total_steps<5000:
        print ("Random pick")
        a = np.random.randint(0, 10)
    elif(np.random.random_sample()>0.2):
        print ("Pick max in Q")
        a = np.argmax(QTable[currentStateIndex,:])
    else:
        print ("Random pick")
        a = np.random.randint(0, 10)
    rAll += r
    
    if d == True: # initial the dialog and reset the buffers and Accumulated Q
        newDialogSetup()
        diagNumber = diagNumber + 1
        #print('\n\n New Dialog:',diagNumber)
        #print('Dialog total reward:',rAll)
        rList.append(rAll)

    #if len(rList) % 10 == 0:
    #    print(total_steps,np.mean(rList[-10:]), e)
    #saver.save(sess,path+'/model-'+str(i)+'.cptk')
    return a


def hasNewTurnDoubleQNN(formerAction,formerReward,currentState,d,formerState):
    s = formerState
    a = formerAction
    r = formerReward
    s1 = currentState
    d = False # The termination indiction 
    #In our case, the termiantion states are: [0,0,0,0,0,0,0,0,0,0,0],[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    if currentState == [0,0,0,0,0,0,0,0,0,0,0] or currentState == [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]:
        d = True

    global j,total_steps,episodeBuffer,mainQN,targetQN,e,diagNumber,rAll
    j+=1
    total_steps = total_steps+1
    
    print("dailog total turn,total turn",j,total_steps)

    #Choose an action by greedily (with e chance of random action) from the Q-network
    if np.random.rand(1) < e or total_steps < pre_train_steps:
        a = np.random.randint(0,11)
        print("randomly pick an action:",a)
    else:
        a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
        print("NN picks an action:",a)

    if(len(set(formerState))!=1):
        episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
        print ("Add the turn to buffer: ",[s,a,r,s1,d])
    
    if total_steps > pre_train_steps:
        print("Get into",total_steps % (update_freq))
        if e > endE:
            e -= stepDrop
        
        if total_steps % (update_freq) == 0:
            print ("Training NN turn:",total_steps)
            trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
            #Below we perform the Double-DQN update to the target Q-values
            Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
            Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
            #print ("Q1,Q2: ",Q1,Q2)
            end_multiplier = -(trainBatch[:,4] - 1)
            doubleQ = Q2[range(batch_size),Q1]
            targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
            #Update the network with our target values.
            _ = sess.run(mainQN.updateModel,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
            
            updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
        rAll += r
    
    if(len(set(formerState))!=1):
        myBuffer.add(episodeBuffer.buffer)
    #jList.append(j)
    #rList.append(rAll)
    #Periodically save the model. 
    if j % 50 == 0:
        saver.save(sess,path+'/model-'+str(j)+'.cptk')
        print("Saved Model")

    if d == True: # initial the dialog and reset the buffers and Accumulated Q
        newDialogSetup()
        diagNumber = diagNumber + 1
        print('\n\n New Dialog:',diagNumber)
        print('Dialog total reward:',rAll)
        rList.append(rAll)

    #if len(rList) % 10 == 0:
    #    print(total_steps,np.mean(rList[-10:]), e)
    #saver.save(sess,path+'/model-'+str(i)+'.cptk')
    return a

def main():
    tf.reset_default_graph()
    mainQN = Qnetwork(h_size)
    targetQN = Qnetwork(h_size)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables,tau)

    myBuffer = experience_buffer()

    #Set the rate of random action decrease. 
    e = startE
    stepDrop = (startE - endE)/anneling_steps

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0

    #Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        sess.run(init)
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
        for i in range(num_episodes):
            episodeBuffer = experience_buffer()
            #Reset environment and get first new observation
            s = [0,0,0,0,0,0,0,0,0,0,0]
            #s = processState(s)
            d = False
            rAll = 0
            j = 0
            #The Q-Network
            while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a = np.random.randint(0,11)
                else:
                    a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
                # Get the reward, new state.
                print (s1,r,d)
                #s1 = processState(s1)
                # Get the new state

                total_steps += 1
                episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
                
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop
                    
                    if total_steps % (update_freq) == 0:
                        trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                        #Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                        Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                        end_multiplier = -(trainBatch[:,4] - 1)
                        doubleQ = Q2[range(batch_size),Q1]
                        targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                        #Update the network with our target values.
                        _ = sess.run(mainQN.updateModel,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                        
                        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
                rAll += r
                s = s1
                
                if d == True:
                    break
            
            myBuffer.add(episodeBuffer.buffer)
            jList.append(j)
            rList.append(rAll)
            #Periodically save the model. 
            if i % 1000 == 0:
                saver.save(sess,path+'/model-'+str(i)+'.cptk')
                print("Saved Model")
            if len(rList) % 10 == 0:
                print(total_steps,np.mean(rList[-10:]), e)
        saver.save(sess,path+'/model-'+str(i)+'.cptk')
    print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")


if __name__ == "__main__":
  if not os.path.exists(path):
    os.makedirs(path)

    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.

  # The model has been loaded.
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
  #Service_OpenFace_pb2.add_openfaceServicer_to_server(Servicer_openface(), server)
  FoodBotRLAgent_pb2.add_FoodBotRLRequestServicer_to_server(FoodBotRLAgent(),server)
  server.add_insecure_port('[::]:50053')
  server.start()
  print ("GRCP Server is running. Press any key to stop it.")
  try:
    while True:
      # openface_GetXXXXXX will be responsed if any incoming request is received.
      time.sleep(48*60*60)
  except KeyboardInterrupt:
    server.stop(0)



