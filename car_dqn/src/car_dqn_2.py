#!/usr/bin/env python
# -*- coding: utf-8 -*-

## ================================================================ ##
## ====      Updated with Tensorflow 2.0 Keras Support         ==== ##
## ====      and Publish ROS message for Ackermann Msg         ==== ##
## ================================================================ ##
import rospy
import roslib; roslib.load_manifest('car_dqn')
import tf
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import String

import random
import numpy as np
import cv2
import datetime
import os
import signal
import sys
import math
import time
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.callbacks import TensorBoard, Callback
import airsim
from multiprocessing import Value
from ctypes import c_bool, c_int, c_float

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Float32

## ROS Joystick Parameters ##
ButtonBrake = 1
ButtonReverse = 0
ButtonJoyPadControl = 7
ButtonReset = 8
AxisSteer = 0
AxisThrottle = 4
ScaleSteer = 1.0
ScaleThrottle = 1.0
ThrottleReverse = 1.0
SteerReverse = 1.0

## DQN Constants ##
EPISODES = 10000
FRAME_NUM = 4
RESIZE = 84
SKIP_FRAME = 4
SHOOT_FRAME = 5
ACTION_SPACE = 7
MAX_STEP = 2000 #EPISODES * 10
BATCH_SIZE = 32
REWARD_BASE = 1#5000
SCREEN_SIZE = (640, 480)
NEAR = 3.0
CLOSE = 6.0

global car_controls
car_controls = None

# for logging training loss history
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, batch_size = 32, concat = True):
        self.state_size = state_size
        self.k_state_size = (1, state_size[0], state_size[1], state_size[2])
        self.kb_state_size = (batch_size, state_size[0], state_size[1], state_size[2])
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.concat = concat
        self.gamma = 0.9    # discount rate
        self.epsilon_max = 1.0  # initial exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.001
        self.learning_rate = 0.001
        self.tau = 0.125 #256
        self.tau_step = 200
        self.batch_size = batch_size
        self.model = self._build_concat_model()
        self.target_model = clone_model(self.model)
        self.observe = 0
        self.epsilon = 1.0
        #self.losses = []
        self.random_action = 0

        #self.tensorboard = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, activation = 'elu', input_shape = self.state_size, padding='valid', kernel_initializer='glorot_normal'))
        #model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (4, 4), strides=2, activation = 'elu', padding='valid', kernel_initializer='glorot_normal'))
        #model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), strides=2, activation = 'elu', padding='valid', kernel_initializer='glorot_normal'))
        model.add(Flatten())
        #model.add(Dropout(0.5))
        model.add(Dense(512, activation = 'elu', kernel_initializer='glorot_normal'))
        #model.add(Dropout(0.5))
        model.add(Dense(self.action_size, activation = None))

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

        model.summary()

        return model

    def _build_concat_model(self):
        Img = Input(shape = self.state_size, name = 'car_image')
        conv1 = Conv2D(32, (4, 4), activation = 'elu', padding='valid', kernel_initializer='glorot_normal')(Img)
        max1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(64, (4, 4), activation = 'elu', padding='valid', kernel_initializer='glorot_normal')(max1)
        max2 = MaxPooling2D((2, 2))(conv2)
        conv3 = Conv2D(64, (3, 3), activation = 'elu', padding='valid', kernel_initializer='glorot_normal')(max2)
        max3 = MaxPooling2D((2, 2))(conv3)
        flat = Flatten()(max3)
        h1 = Dense(256, activation = 'elu', kernel_initializer='glorot_normal')(flat)

        V = Input(shape = (3, ), name = 'velocity') 
        v1 = Dense(256, activation='linear')(V) 

        concat = concatenate([h1, v1])
        concat2 =  Dense(128, activation='elu')(concat) 
        out = Dense(self.action_size, activation = None)(concat2)

        model = Model(inputs=[Img, V], outputs=out)

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

        model.summary()

        return model

    def remember(self, state, vel, action_id, reward, next_state, next_vel, done):
        self.memory.append((state, vel, action_id, reward, next_state, next_vel, done))

    def move(self, state, vel, decay_step):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * decay_step)
    
        if (self.epsilon > np.random.rand()):
            # Make a random action (exploration)
            action_id = random.randint(0,self.action_size-1)
            self.random_action = 1
        else:
            # Get action from Q-network (exploitation)
            action_id = np.argmax(self.model.predict([np.reshape(state, self.k_state_size), np.reshape(vel, (1, 3))]))
            self.random_action = 0
        return action_id

    def trained_move(self, state, vel):
        return np.argmax(self.model.predict([np.reshape(state, self.k_state_size), np.reshape(vel, (1, 3))]))

    def learn(self): 
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states_mb = np.array([each[0] for each in batch], ndmin=3)
        vel_mb = np.array([each[1] for each in batch], ndmin=1)
        actions_mb = np.array([each[2] for each in batch])
        rewards_mb = np.array([each[3] for each in batch]) 
        next_states_mb = np.array([each[4] for each in batch], ndmin=3)
        next_vel_mb = np.array([each[5] for each in batch], ndmin=1)
        dones_mb = np.array([each[6] for each in batch])

        target_Qs_batch = []

        # Get Q values for next_state 
        Qs_next_state = np.squeeze(self.target_model.predict([np.reshape(next_states_mb, self.kb_state_size), np.reshape(next_vel_mb, (self.batch_size, 3))]))
        targets_mb = np.squeeze(self.target_model.predict([np.reshape(states_mb, self.kb_state_size), np.reshape(vel_mb, (self.batch_size, 3))]))

        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
        for i in range(self.batch_size):
            done = dones_mb[i]
            if done:
                targets_mb[i,actions_mb[i]] = rewards_mb[i]
            else:
                targets_mb[i,actions_mb[i]] = rewards_mb[i] + self.gamma * np.max(Qs_next_state[i])

        self.history = self.model.fit([states_mb, vel_mb], targets_mb, epochs = 1, verbose = False)
        print("loss = ", self.history.history['loss'])

    def train_target_model(self, decay_step):
        if decay_step % self.tau_step == 0 and decay_step > self.tau_step:
            weights = self.model.get_weights()
            self.target_model.set_weights(weights)
        print("Target network weights updated")

    def load_model(self, name):
        if os.path.isfile(name):
            self.model.load_weights(name)
            self.target_model.load_weights(name)
            print("==============  Successfully loaded model weights ==============")
        else:
            print("=============== Can't load the model weights! ================")

    def save_model(self, name):
        self.model.save_weights(name)
        print("Successfully saved model weights")

    def memory_length(self):
        return len(self.memory)
    
def preprocess_frame(response):
    np_img = np.asarray(cv2.imdecode(airsim.string_to_uint8_array(response.image_data_uint8), cv2.IMREAD_UNCHANGED))
    return cv2.resize(cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY), (RESIZE, RESIZE), interpolation=cv2.INTER_CUBIC) / 255.0

def frames_to_state(frame_queue):
    state = np.zeros((RESIZE, RESIZE, FRAME_NUM))
    # do frame skipping and find maximum of ith and i-1th frame
    for i in range (0, FRAME_NUM):
        state[:,:,i] = np.maximum(frame_queue[(i+1)*SKIP_FRAME-1], frame_queue[(i+1)*SKIP_FRAME-2])
    return state

def interpret_action(action_id):
    #car_controls.brake = 0
    #car_controls.throttle = 1
    if action_id == 0:
        car_controls.throttle = 0
        car_controls.brake = 1
    elif action_id == 1:
        car_controls.steering = 0
    elif action_id == 2:
        car_controls.steering = 0.5
    elif action_id == 3:
        car_controls.steering = -0.5
    elif action_id == 4:
        car_controls.steering = 0.25
    elif action_id == 5:
        car_controls.steering = -0.25
    elif action_id == 6:
        car_controls.brake = 0
        car_controls.throttle = 1

    return car_controls

def compute_reward(car_state, car_controls, collision_info, pos):
    MAX_SPEED = 5.0
    MIN_SPEED = 1.0
    thresh_dist = 3.5
    beta = 3

    z = 0
    pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]), np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]), np.array([0, -1, z])]
    pd = car_state.kinematics_estimated.position
    car_pt = np.array([pd.x_val, pd.y_val, pd.z_val])

    dist = 10000000
    for i in range(0, len(pts)-1):
        dist = min(dist, np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

    #print(dist)
    if dist > thresh_dist:
        reward = -3
    else:
        reward_dist = math.exp(-beta*dist) * 1.5 #- 0.5
        reward_speed = (((car_state.speed - MIN_SPEED)/(MAX_SPEED - MIN_SPEED)) * 10)
        reward = reward_dist + reward_speed

    if collision_info.has_collided:
        print("collision")
        reward = -100

   
    #if car_state.speed > MAX_SPEED:
    #    print("overspeed")
    #    if car_controls.brake == 0:
    #        reward = -10
    #    else:
    #        reward += 3
    
    # check closing in goal
    goal_dist = getDist(pos, [120, 0])
    if goal_dist < NEAR:
        reward += goal_dist / 120 * 100
    #else:
    #    reward += 100 - goal_dist

    #if car_state.speed < 0:
    #    print("move backward")
    #    reward = -10
    #print("reward = "+str(reward))
    return reward

def isDone(car_state, car_controls, reward, frame_cnt):
    done = 0
    if reward < -50:
        done = 1

    if frame_cnt > MAX_STEP:
        done = 1
    return done

def getDist(pos, wp):
    return math.sqrt(pow((wp[0] - pos.x_val),2) + pow((wp[1] - pos.y_val), 2))

def train():
    decay_step = 0 
    # begin training episodes
    for e in range(EPISODES):
        # initialize the environment for every episode
        responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Segmentation)])
        frame_queue.append(preprocess_frame(responses[0]))

        # initialize the state (with initial frame)
        state = frames_to_state(frame_queue)
        v = client.getCarState().kinematics_estimated.linear_velocity
        vel = np.array([v.x_val, v.y_val, v.z_val])

        # reset frame counter for counting frame
        frame_cnt = 1        

        if manualDrive.value:
            car_controls.brake = brake.value
            car_controls.steering = steering.value
            car_controls.throttle = throttle.value
            car_controls.manual_gear = manual_gear.value
            client.setCarControls(car_controls)
            #pubCar.publish(rosmsg(brake.value, steering.value, throttle.value, manual_gear.value))

            ## Publish AckermannDriveStamp in ROS
            #print(float(car_controls.steering), float(v.y_val))
            ackermann_drive = AckermannDrive(steering_angle = float(car_controls.steering), speed = float(v.y_val)/20.0)
            ackermann_drive_stamp = AckermannDriveStamped(drive = ackermann_drive)
            ackermann_drive_stamp.header.frame_id = "car_dqn"
            ackermann_drive_stamp.header.stamp = rospy.Time.now()
            pubAckermann.publish(ackermann_drive_stamp)

            if reset.value:
                client.reset()



        while not manualDrive.value: #True:
            # get action on the current frame using the last known state
            action_id = agent.move(state, vel, decay_step )
            car_controls = interpret_action(action_id)
            pubCar.publish(rosmsg(car_controls.brake, car_controls.steering, car_controls.throttle, car_controls.manual_gear))
            client.setCarControls(car_controls)
            #print(car_controls.steering, car_controls.throttle)

            ## Publish AckermannDriveStamp in ROS
            print(float(car_controls.steering), float(v.y_val))
            ackermann_drive = AckermannDrive(steering_angle = float(car_controls.steering), speed = float(v.y_val)/20.0)
            ackermann_drive_stamp = AckermannDriveStamped(drive = ackermann_drive)
            ackermann_drive_stamp.header.frame_id = "car_dqn"
            ackermann_drive_stamp.header.stamp = rospy.Time.now()
            pubAckermann.publish(ackermann_drive_stamp)


            # get next observation based on the action above
            # Note: the reward obtained from env.step() in the game is not the actual reward
            # done is not used
            car_state = client.getCarState()
            collision_info = client.simGetCollisionInfo()

            reward = compute_reward(car_state, car_controls, collision_info, pos) 
            done = isDone(car_state, car_controls, reward, frame_cnt)

            # append new observation into frame_queue            
            next_obs = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Segmentation)])
            frame_queue.append(preprocess_frame(next_obs[0]))

            # get game state for every SKIP_FRAME-th (e.g. 4) frame
            if frame_cnt % SKIP_FRAME == 0:
                next_state = frames_to_state(frame_queue) 
                v = car_state.kinematics_estimated.linear_velocity
                next_vel = np.array([v.x_val, v.y_val, v.z_val])   
                  
                # save into agent's memory
                agent.remember(state, vel, action_id, reward, next_state, next_vel, done)

                # get experience replay and train agent's and target's model
                agent.learn()
                agent.train_target_model(decay_step)
                
                print("Training>> frame:{}, reward:{}, done:{}, action_id:{}, epsilon:{}, is_random:{}".format(frame_cnt, reward, done, action_id, agent.epsilon, agent.random_action))
                state = next_state
                vel = next_vel
                decay_step += 1
                
                if done:
                    client.reset()
                    car_control = interpret_action(1)
                    client.setCarControls(car_control)
                    time.sleep(1)
                    print("{} episode {}/{} ends. reward: {}, total frame:{}, action_id:{}" .format(datetime.datetime.now, e, EPISODES, reward, frame_cnt, action_id))
                    break

            frame_cnt += 1

        if e % 10 == 0 and e > 10 and not manualDrive.value:
            agent.save_model("/home/oscar/catkin/src/car_dqn/src/cardqn.h5")

        if not globalRun.value:
            print("break from loop")
            break

def rosmsg(brake, steering, throttle, manual_gear):
    msg_str = repr(brake)+" "+repr(steering)+" "+repr(throttle)+" "+repr(manual_gear)
    return msg_str

def ReceiveJoystickMessage(data):
    if data.buttons[ButtonJoyPadControl]==1:
        if manualDrive.value:
            manualDrive.value = False
            rospy.loginfo("============ Manual Drive Disengaged ================")
        else:
            manualDrive.value = True
            rospy.loginfo("============= Manual Drive Engaged ==================")

    if data.buttons[ButtonReset]==1:
        rospy.loginfo("Reset Car")
        if manualDrive.value:
            reset.value = True
        else:
            reset.value = False
    
    brake.value = data.buttons[ButtonBrake]
    print(float(data.axes[AxisSteer]/ScaleSteer), float(data.axes[AxisThrottle]/ScaleThrottle))

    steering.value = float(data.axes[AxisSteer]/ScaleSteer*SteerReverse)
    #throttle.value = float(data.axes[AxisThrottle]/ScaleThrottle*ThrottleReverse)
    throttle.value = (float(data.axes[AxisThrottle]) * -1 + 1)/2.0/ScaleThrottle*ThrottleReverse

    if data.buttons[ButtonReverse] == 1:
        if car_controls.manual_gear == 1:
            rospy.loginfo("Reverse Gear")
            manual_gear.value = -1
        else:
            rospy.loginfo("Drive Gear")
            manual_gear.value = 1

def ReceiveCarMessage(data):
    return

# to shut down the ROS Service gracefully after pressing Ctrl-C
def sigint_handler(signum, data):
    rospy.loginfo('Ctrl-C is pressed.')
    globalRun.value = False
    rospy.signal_shutdown('Shutting Down...')  
    sys.exit(0)

if __name__ == "__main__":
    #global car_controls
    rospy.init_node('car_dqn')
    rate = rospy.Rate(10) # 10hz

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    pubCar = rospy.Publisher("carDQNAction", String, queue_size=1)

    pubAckermann = rospy.Publisher("/dqn_ackermann_cmd", AckermannDriveStamped, queue_size=1) 
    #pubBrake = rospy.Publisher("/dqn_brake", Float32, queue_size=1)

    subJoystick = rospy.Subscriber('/joy', Joy, ReceiveJoystickMessage)
    #subJoystick = rospy.Subscriber('/rosCar', String, ReceiveCarMessage)

    # create a game environment and initialize it
    client = airsim.CarClient()
    client.confirmConnection()
    client.reset()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()
    car_controls.is_manual_gear = False #True

    ButtonReverse = int (   rospy.get_param("~ButtonReverse",ButtonReverse) )
    ButtonJoyPadControl = int (   rospy.get_param("~ButtonJoyPadControl",ButtonJoyPadControl) )
    ButtonBrake = int (   rospy.get_param("~ButtonBrake",ButtonBrake) )
    ButtonReset = int (   rospy.get_param("~ButtonReset",ButtonReset) )
    AxisSteer        = int (   rospy.get_param("~AxisSteer",AxisSteer) )
    AxisThrottle        = int (   rospy.get_param("~AxisThrottle",AxisThrottle) )

    pos = client.simGetVehiclePose().position

    manualDrive = Value(c_bool, False)
    globalRun = Value(c_bool, True)
    manual_gear = Value(c_int, 1)
    brake = Value(c_int, 0)
    steering = Value(c_float, 0.0)
    throttle = Value(c_float, 0.0)
    reset = Value(c_bool, False)

    # obtain state parameters
    state_size = (RESIZE, RESIZE, FRAME_NUM)
    action_size = ACTION_SPACE

    # create game frame container
    frame_queue = deque(maxlen=FRAME_NUM * SKIP_FRAME) # 16 frames at most for sampling 1 every 4 frames
    for i in range (0, FRAME_NUM * SKIP_FRAME):
        frame_queue.append(np.zeros((RESIZE, RESIZE)))

    # initialize learning agent
    agent = DQNAgent(state_size, action_size, batch_size = BATCH_SIZE)
    agent.load_model("/home/oscar/catkin/src/car_dqn/src/cardqn.h5")

    try:
        train()
    except rospy.ROSInterruptException, KeyboardInterrupt:
        globalRun.value = False
        pass

