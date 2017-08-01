# Deep Determinstic Policy Gradient
# DQN Class
# v0.0 @ 7/26/2017 
# Yuanda Wang

# GOD! IT WORKS

import numpy as np
import tensorflow as tf
import random

class DQN_Agent():
    def __init__(self, settings, session):
        # parameters 
        self.buffer_size = settings['DQN']['replay_buffer_size']
        self.batch_size = settings['DQN']['batch_size']
        self.gamma = settings['DQN']['gamma']
        self.dim_s = settings['num_state']
        self.dim_a = settings['num_action']
        self.rl_a = settings['DQN']['learning_rate_A']
        self.rl_c = settings['DQN']['learning_rate_C']
        self.tau = settings['DQN']['soft_update_rate']
        
        self.USECONV = settings['DQN']['USECONV']
        self.structure = settings['DQN']['net_structure']
        if self.USECONV:
            self.conv1num = settings['DQN']['num_conv1']
            self.conv2num = settings['DQN']['num_conv2']
            self.fc1num = settings['DQN']['num_fc1']
            self.fc2num = settings['DQN']['num_fc2']
        else:
            self.fc1num = settings['DQN']['num_fc1']
            self.fc2num = settings['DQN']['num_fc2']
        
        # other parameters
        # ....
        
        # special settings for two stream
        if self.structure == 'TWOSTREAM':
            self.dim_s *= 2
        
        # TensorFlow Network
        self.sess = session
        # input place holders

            
         
        self.S  =  tf.placeholder(tf.float32, [None, self.dim_s], 'State')
        self.S1 =  tf.placeholder(tf.float32, [None, self.dim_s], 'newState')
        self.A  =  tf.placeholder(tf.int32,   [None, 1], 'action')
        self.R  =  tf.placeholder(tf.float32, [None, 1], 'reward')
        
        if self.structure == 'CONV':
            self.Sconv = tf.reshape(self.S, shape=[-1, self.dim_s, 1, 1])
            self.S1conv = tf.reshape(self.S1, shape=[-1, self.dim_s, 1, 1])
            self.Q   = self.new_critic_convnet(self.Sconv, 'criticMain')
            self.Qt  = self.new_critic_convnet(self.S1conv, 'criticTarget')
            
        elif self.structure == 'TWOSTREAM':
            self.Q   = self.new_critic_twostream(self.S, 'criticMain')
            self.Qt  = self.new_critic_twostream(self.S1, 'criticTarget')
            
        elif self.structure == 'NORMAL':
            self.Q   = self.new_critic_net(self.S, 'criticMain')
            self.Qt  = self.new_critic_net(self.S1, 'criticTarget')
            
        else:
            raise ValueError('Network structure NOT recognized')
        
        # param group
        # should be tf.GraphKeys.GLOBAL_VARIABLES in future version of TF
        self.Q_param  = tf.get_collection(tf.GraphKeys.VARIABLES, scope='criticMain')
        self.Qt_param = tf.get_collection(tf.GraphKeys.VARIABLES, scope='criticTarget')
        
        # loss functions of Q
        self.A_onehot = tf.one_hot(self.A, self.dim_a, dtype=tf.float32)
        target_Q = self.R + self.gamma * self.Qt
        masked_target_Q = target_Q * self.A_onehot
        masked_Q = self.Q * self.A_onehot
        td_error = masked_Q - masked_target_Q
        
        self.loss = tf.reduce_mean(tf.square(td_error))
        self.trainer_critic = tf.train.AdamOptimizer(self.rl_c).minimize(self.loss, var_list=self.Q_param)
        
        # soft update to target networks operations
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
    # training critic
    def train_it(self):
        s, a, r, s1, d = self.buffer_sample()
        # train critic network
        # here the A should be assign with stored batch a
        _, loss=self.sess.run([self.trainer_critic, self.loss], feed_dict={self.S:s, self.A:a, self.R:r, self.S1:s1})
        self.sess.run(self.update_criticTarget)
        
        return loss
        
        
    # critic networks
    def new_critic_net(self, s, scope):
        with tf.variable_scope(scope):
            W1 = tf.Variable(tf.random_normal([self.dim_s, self.fc1num] ,stddev=0.01))
            b1 = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            W2 = tf.Variable(tf.random_normal([self.fc1num, self.fc2num] ,stddev=0.01))
            b2 = tf.Variable(tf.random_normal([self.fc2num] ,stddev=0.01))
            W3 = tf.Variable(tf.random_normal([self.fc2num, self.dim_a] ,stddev=0.01))
            b3 = tf.Variable(tf.random_normal([self.dim_a] ,stddev=0.01))
            
            fc1 = tf.nn.relu(tf.matmul(s, W1) + b1)
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
            out = tf.matmul(fc2, W3) + b3
            return out
    
    def new_critic_twostream(self, state, scope):
        with tf.variable_scope(scope):
            # split into two parts
            s, v = tf.split(split_dim=1, num_split=2, value=state)
            
            fc3num = 32
            # opt flow stream params
            W1_v = tf.Variable(tf.random_normal([self.dim_s, self.fc1num] ,stddev=0.01))
            b1_v = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            W2_v = tf.Variable(tf.random_normal([self.fc1num, self.fc2num] ,stddev=0.01))
            b2_v = tf.Variable(tf.random_normal([self.fc2num] ,stddev=0.01))
            W3_v = tf.Variable(tf.random_normal([self.fc2num, fc3num] ,stddev=0.01))
            b3_v = tf.Variable(tf.random_normal([fc3num] ,stddev=0.01))
            
            # still ob stream params
            W1 = tf.Variable(tf.random_normal([self.dim_s, self.fc1num] ,stddev=0.01))
            b1 = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            W2 = tf.Variable(tf.random_normal([self.fc1num, self.fc2num] ,stddev=0.01))
            b2 = tf.Variable(tf.random_normal([self.fc2num] ,stddev=0.01))
            W3 = tf.Variable(tf.random_normal([self.fc2num, fc3num] ,stddev=0.01))
            b3 = tf.Variable(tf.random_normal([fc3num] ,stddev=0.01))
            
            W4 = tf.Variable(tf.random_normal([fc3num, self.dim_a] ,stddev=0.01))
            b4 = tf.Variable(tf.random_normal([self.dim_a], stddev=0.01))
            
            W4_v = tf.Variable(tf.random_normal([fc3num, self.dim_a] ,stddev=0.01))
            b4_v = tf.Variable(tf.random_normal([self.dim_a], stddev=0.01))            
            
            fc1 = tf.nn.relu(tf.matmul(s, W1) + b1)
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
            fc3 = tf.nn.relu(tf.matmul(fc2, W3) + b3)
            fc1_v = tf.nn.relu(tf.matmul(v, W1_v) + b1_v)
            fc2_v = tf.nn.relu(tf.matmul(fc1_v, W2_v) + b2_v)
            fc3_v = tf.nn.relu(tf.matmul(fc2_v, W3_v) + b3_v)
            
            out = tf.matmul(fc3, W4) + b4 + tf.matmul(fc3_v, W4_v) + b4_v

            return out
    
    
    def new_critic_convnet(self, s, scope):
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
            W2 = tf.Variable(tf.random_normal([self.fc1num, self.fc2num] ,stddev=0.01))
            b2 = tf.Variable(tf.random_normal([self.fc2num] ,stddev=0.01))
            W3 = tf.Variable(tf.random_normal([self.fc2num, 1], stddev=0.01))
            b3 = tf.Variable(tf.random_normal([1], stddev=0.01))
            
            fc1 = tf.nn.relu(tf.matmul(convout, W1) + b1)
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
            out = tf.matmul(fc2, W3) + b3
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
        
