# Deep Determinstic Policy Gradient
# DDPG Class
# v0.0 @ 7/26/2017 
# Yuanda Wang

# GOD! IT WORKS

import numpy as np
import tensorflow as tf
import random

class DDPG_Agent_Conv():
    def __init__(self, settings, session):
        # parameters 
        self.buffer_size = settings['DDPG']['replay_buffer_size']
        self.batch_size = settings['DDPG']['batch_size']
        self.gamma = settings['DDPG']['gamma']
        self.dim_s = settings['num_state']
        self.dim_a = settings['num_action']
        self.rl_a = settings['DDPG']['learning_rate_A']
        self.rl_c = settings['DDPG']['learning_rate_C']
        self.tau = settings['DDPG']['soft_update_rate']
        
        self.structure = settings['DDPG']['net_structure']
        self.conv1num = settings['DDPG']['num_conv1']
        self.conv2num = settings['DDPG']['num_conv2']
        self.fc1num = settings['DDPG']['num_fc1']
        self.fc2num = settings['DDPG']['num_fc2']

        # other parameters
        # ....
        
        # TensorFlow Network
        self.sess = session
        # input place holders
        self.S = tf.placeholder(tf.float32, [None, self.dim_s*self.dim_s], 'State')
        self.S1 = tf.placeholder(tf.float32, [None, self.dim_s*self.dim_s], 'newState')
        self.R = tf.placeholder(tf.float32, [None, 1], 'reward')

        # resize the input [batch_size, shape, in_channels]
        self.Sconv = tf.reshape(self.S, shape=[-1, self.dim_s, self.dim_s, 1])
        self.S1conv = tf.reshape(self.S1, shape=[-1, self.dim_s, self.dim_s, 1])

        self.A   = self.new_actor_convnet(self.Sconv, 'actorMain')
        self.At  = self.new_actor_convnet(self.S1conv, 'actorTarget')
        self.Q   = self.new_critic_convnet(self.Sconv, self.A, 'criticMain')
        self.Qt  = self.new_critic_convnet(self.S1conv, self.At, 'criticTarget')

        # param group
        # should be tf.GraphKeys.GLOBAL_VARIABLES in future version of TF
        self.A_param  = tf.get_collection(tf.GraphKeys.VARIABLES, scope='actorMain')
        self.At_param = tf.get_collection(tf.GraphKeys.VARIABLES, scope='actorTarget')
        self.Q_param  = tf.get_collection(tf.GraphKeys.VARIABLES, scope='criticMain')
        self.Qt_param = tf.get_collection(tf.GraphKeys.VARIABLES, scope='criticTarget')
        
        # loss functions of Critic
        target_Q = self.R + self.gamma * self.Qt
        #td_error = tf.losses.mean_squared_error(labels=target_Q, predictions=self.Q)
        td_error = self.Q - target_Q
        self.loss_c = tf.reduce_mean(tf.square(td_error))
        self.trainer_critic = tf.train.AdamOptimizer(self.rl_c).minimize(self.loss_c, var_list=self.Q_param)
        
        # loss function of Actor
        loss_a =  tf.reduce_mean(-self.Q)
        self.trainer_actor = tf.train.AdamOptimizer(self.rl_a).minimize(loss_a, var_list=self.A_param)
        
        # soft update to target networks operations
        # tf.multiply in future version
        # for Actor
        self.update_actorTarget = \
        [self.At_param[i].assign(tf.mul(self.A_param[i], self.tau) + tf.mul(self.At_param[i], (1-self.tau)))
         for i in range(len(self.At_param))]
        # for Critic
        self.update_criticTarget = \
        [self.Qt_param[i].assign(tf.mul(self.Q_param[i], self.tau) + tf.mul(self.Qt_param[i], (1-self.tau)))
         for i in range(len(self.Qt_param))]
        
        #self.all_vars = tf.trainable_variables()
        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        
        # replay buffer
        self.exp_buffer = []
        self.buffer_ready = False

    
    # functions:
    # training critic and actor and soft update
    def train_it(self):
        s, a, r, s1, d = self.buffer_sample()
        # train critic network
        # here the A should be assign with stored batch a
        _, loss=self.sess.run([self.trainer_critic, self.loss_c], feed_dict={self.S:s, self.A:a, self.R:r, self.S1:s1})
        # train actor network
        self.sess.run(self.trainer_actor, feed_dict={self.S:s})
        # soft update target network from main network
        self.sess.run(self.update_actorTarget)
        self.sess.run(self.update_criticTarget)
        
        return loss
        
        
    # critic networks and actor networks
    def new_critic_net(self, s, a, scope):
        with tf.variable_scope(scope):
            W1 = tf.Variable(tf.random_normal([self.dim_s, self.fc1num] ,stddev=0.01))
            b1 = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            W1_a = tf.Variable(tf.random_normal([self.dim_a, self.fc1num] ,stddev=0.01))
            b1_a = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            
            W2 = tf.Variable(tf.random_normal([self.fc1num, self.fc2num] ,stddev=0.01))
            b2 = tf.Variable(tf.random_normal([self.fc2num] ,stddev=0.01))
            W3 = tf.Variable(tf.random_normal([self.fc2num, 1] ,stddev=0.01))
            b3 = tf.Variable(tf.random_normal([1] ,stddev=0.01))
            
            # !! s and a should be linked together in the critic network, or it will not work
            fc1 = tf.nn.relu(tf.matmul(s, W1) + b1 + tf.matmul(a, W1_a) + b1_a)
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
            out = tf.matmul(fc2, W3) + b3
            return out
            
            
    def new_actor_net(self, s, scope):
        with tf.variable_scope(scope):
            W1 = tf.Variable(tf.random_normal([self.dim_s, self.fc1num] ,stddev=0.01))
            b1 = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            
            W2 = tf.Variable(tf.random_normal([self.fc1num, self.fc2num] ,stddev=0.01))
            b2 = tf.Variable(tf.random_normal([self.fc2num] ,stddev=0.01))
            W3 = tf.Variable(tf.random_normal([self.fc2num, self.dim_a] ,stddev=0.01))
            b3 = tf.Variable(tf.random_normal([self.dim_a] ,stddev=0.01))            
            
            fc1 = tf.nn.relu(tf.matmul(s, W1) + b1)
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
            out = tf.tanh(tf.matmul(fc2, W3) + b3)
            return out
        
    def new_critic_convnet(self, s, a, scope):
        with tf.variable_scope(scope):
            # convolutions acti:relu
            self.conv1 = tf.contrib.layers.convolution2d(
                inputs=s,
                num_outputs = self.conv1num,
                kernel_size=[5,5], stride=[2,2], padding='VALID', biases_initializer=None)

            self.conv2 = tf.contrib.layers.convolution2d(
                inputs=self.conv1,
                num_outputs = self.conv2num,
                kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
            convout = tf.contrib.layers.flatten(self.conv2)
            convoutnum = convout.get_shape().as_list()[1]
            
            # full connected
            W1 = tf.Variable(tf.random_normal([convoutnum, self.fc1num] ,stddev=0.01))
            b1 = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            W1_a = tf.Variable(tf.random_normal([self.dim_a, self.fc1num] ,stddev=0.01))
            W2 = tf.Variable(tf.random_normal([self.fc1num, self.fc2num] ,stddev=0.01))
            b2 = tf.Variable(tf.random_normal([self.fc2num] ,stddev=0.01))
            # deeper
            W3 = tf.Variable(tf.random_normal([self.fc2num, 1], stddev=0.01))
            b3 = tf.Variable(tf.random_normal([1], stddev=0.01))
            
            # !! s and a should be linked together in the critic network, or it will not work
            fc1 = tf.nn.relu(tf.matmul(convout, W1) + b1 + tf.matmul(a, W1_a))
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
            out = tf.matmul(fc2, W3) + b3
            return out
            
            
    
    def new_actor_convnet(self, s, scope):
        with tf.variable_scope(scope):
            # convolutions acti:relu
            self.conv1 = tf.contrib.layers.convolution2d(
                inputs=s,
                num_outputs = self.conv1num,
                kernel_size=[5,1], stride=[2,1], padding='VALID', biases_initializer=None)

            self.conv2 = tf.contrib.layers.convolution2d(
                inputs=self.conv1,
                num_outputs = self.conv2num,
                kernel_size=[3,1], stride=[1,1], padding='VALID', biases_initializer=None)
            convout = tf.contrib.layers.flatten(self.conv2)
            convoutnum = convout.get_shape().as_list()[1]
            
            # full connected
            W1 = tf.Variable(tf.random_normal([convoutnum, self.fc1num] ,stddev=0.01))
            b1 = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            W2 = tf.Variable(tf.random_normal([self.fc1num, self.dim_a] ,stddev=0.01))
            b2 = tf.Variable(tf.random_normal([self.dim_a] ,stddev=0.01))
            
            fc1 = tf.nn.relu(tf.matmul(convout, W1) + b1)
            out = tf.tanh(tf.matmul(fc1, W2) + b2)
            return out
            
            
            
    def buffer_add(self, s, a, r, s1, d):
        experience = np.reshape(np.array([s, a, r, s1, d]),[1,5])
        if len(self.exp_buffer) + len(experience) >= self.buffer_size:
            self.exp_buffer[0:(len(experience) + len(self.exp_buffer))-self.buffer_size] = []
        if len(self.exp_buffer) > 100 * self.batch_size:
            self.buffer_ready = True
        self.exp_buffer.extend(experience)
               
    def buffer_sample(self):
        batch_size = self.batch_size
        if self.batch_size > len(self.exp_buffer):
            print 'Not enough experience, now we only have', len(self.exp_buffer)
            batch_size = 1
        batch = np.reshape(np.array(random.sample(self.exp_buffer, batch_size)), [batch_size, 5])
        s_batch  = np.vstack(batch[:, 0])
        a_batch  = np.vstack(batch[:, 1])
        r_batch  = np.vstack(batch[:, 2])
        s1_batch = np.vstack(batch[:, 3])
        d_batch  = np.vstack(batch[:, 4])
        
        return s_batch, a_batch, r_batch, s1_batch, d_batch
        
    def buffer_dump(self):
        self.exp_buffer = []
        
