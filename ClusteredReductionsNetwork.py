# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 03:49:54 2019

@author: Kenneth
"""

#Clustering Parameter selection->convolutional network->HDBSCAN->Calculate Loss->update convolution parameters->optimize hyperparameters
#parameter selection can be done manually or by RL


import tensorflow as tf
from keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

Data_Y_Train=np.zeros((y_train.shape[0],10))
for count,each in enumerate(y_train):
    Data_Y_Train[count,each]=1
    
Data_Y_Test=np.zeros((y_test.shape[0],10))    
for count,each in enumerate(y_test):
    Data_Y_Test[count,each]=1

#%%
    
class ClusterNetwork():
    def __init__(self,X_train):
        self.imageInput=tf.placeholder(shape=(None,X_train.shape[1],X_train.shape[2],X_train.shape[3]),dtype=tf.float32)
        
        self.Conv1=tf.contrib.layers.conv2d(inputs=self.imageInput,num_outputs=83,kernel_size=7,stride=1,padding='same',scope='Conv1',activation_fn=None)
        self.Conv2=tf.contrib.layers.conv2d(inputs=self.Conv1,num_outputs=83,kernel_size=7,stride=1,padding='same',scope='Conv2',activation_fn=None)
        self.Conv3=tf.contrib.layers.conv2d(inputs=self.Conv2,num_outputs=83,kernel_size=7,stride=1,padding='same',scope='Conv3',activation_fn=None)
#        self.BN1=tf.contrib.layers.batch_norm(inputs=self.Conv3,scale=True)
        self.Activation1=tf.nn.relu(self.Conv3,name='Activation1')
        self.MaxPool1=tf.contrib.layers.max_pool2d(self.Activation1,kernel_size=2)
        
        self.Conv4=tf.contrib.layers.conv2d(inputs=self.MaxPool1,num_outputs=123,kernel_size=5,stride=1,padding='same',scope='Conv4',activation_fn=None)
        self.Conv5=tf.contrib.layers.conv2d(inputs=self.Conv4,num_outputs=123,kernel_size=5,stride=1,padding='same',scope='Conv5',activation_fn=None)
        self.Conv6=tf.contrib.layers.conv2d(inputs=self.Conv5,num_outputs=123,kernel_size=5,stride=1,padding='same',scope='Conv6',activation_fn=None)
#        self.BN2=tf.contrib.layers.batch_norm(inputs=self.Conv6,scale=True)
        self.Activation2=tf.nn.relu(self.Conv6,name='Activation2')
        self.MaxPool2=tf.contrib.layers.max_pool2d(self.Activation2,kernel_size=2)
        
        self.Conv7=tf.contrib.layers.conv2d(inputs=self.MaxPool2,num_outputs=193,kernel_size=3,stride=1,padding='same',scope='Conv7',activation_fn=None)
        self.Conv8=tf.contrib.layers.conv2d(inputs=self.Conv7,num_outputs=193,kernel_size=3,stride=1,padding='same',scope='Conv8',activation_fn=None)
        self.Conv9=tf.contrib.layers.conv2d(inputs=self.Conv8,num_outputs=193,kernel_size=3,stride=1,padding='same',scope='Conv9',activation_fn=None)
        
        
        self.ConvSkip1=tf.contrib.layers.conv2d(inputs=self.imageInput,num_outputs=83,kernel_size=7,stride=1,padding='same',scope='ConvSkip1',activation_fn=None)
        self.MaxPoolSkip1=tf.contrib.layers.max_pool2d(self.ConvSkip1,kernel_size=2)
        self.ConvSkip2=tf.contrib.layers.conv2d(inputs=self.MaxPoolSkip1,num_outputs=123,kernel_size=5,stride=1,padding='same',scope='ConvSkip2',activation_fn=None)
        self.MaxPoolSkip2=tf.contrib.layers.max_pool2d(self.ConvSkip2,kernel_size=2)
        self.ConvSkip3=tf.contrib.layers.conv2d(inputs=self.MaxPoolSkip2,num_outputs=193,kernel_size=3,stride=1,padding='same',scope='ConvSkip3',activation_fn=None)
        
        self.Combine=tf.add(self.Conv9,self.ConvSkip3)
        
#        self.BN3=tf.contrib.layers.batch_norm(inputs=self.Combine,scale=True)
        self.Activation3=tf.nn.relu(self.Combine,name='Activation3')
        self.MaxPool3=tf.contrib.layers.max_pool2d(self.Activation3,kernel_size=2)
        
        self.Flatten=tf.contrib.layers.flatten(self.MaxPool3)
        
        self.Dense1=tf.contrib.layers.fully_connected(self.Flatten,num_outputs=10,activation_fn=None)
        
#        self.ActivationFinal=tf.nn.relu(self.Dense1)
        self.labels=tf.placeholder(shape=[None,10],dtype=tf.float32)
        
        
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.labels,axis=1),logits=self.output))
        
        #may need to train, initialized factors send the numbers everywhere
        
#%%

batch_size=100
batch_list=[]
for i in range(500):
    batch_list.append((x_train[i*batch_size:(i+1)*batch_size,:,:,:],Data_Y_Train[i*batch_size:(i+1)*batch_size,:]))
    
train=batch_list[0:400]
val=batch_list[400:499]

batch_list_test=[]

for i in range(100):
    batch_list_test.append((x_test[i*batch_size:(i+1)*batch_size,:,:,:],Data_Y_Test[i*batch_size:(i+1)*batch_size,:]))
    
test=batch_list_test

import pickle
path='E:\\Other Projects\\ClusteredReductions\\Train.pkl'
with open(path,'wb') as output:
    #brc=Birch(threshold=.56,n_clusters=None)
    pickle.dump(train,output,pickle.HIGHEST_PROTOCOL)

path='E:\\Other Projects\\ClusteredReductions\\Val.pkl'
with open(path,'wb') as output:
    #brc=Birch(threshold=.56,n_clusters=None)
    pickle.dump(val,output,pickle.HIGHEST_PROTOCOL)

path='E:\\Other Projects\\ClusteredReductions\\Test.pkl'
with open(path,'wb') as output:
    #brc=Birch(threshold=.56,n_clusters=None)
    pickle.dump(test,output,pickle.HIGHEST_PROTOCOL)
    
    
x_train_input=np.zeros((100,train[0][0].shape[1],train[0][0].shape[2],train[0][0].shape[3]))
tf.reset_default_graph()
network=ClusterNetwork(x_train_input)
init=tf.global_variables_initializer()
path_model=r'E:\Other Projects\ClusteredReductions\Model'
saver=tf.train.Saver(max_to_keep=5)
np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])     

#%%
import hdbscan
reduced_list=[]
    
with tf.Session() as sess:
    sess.run(init)
    
    for each in train:
#        print(each[0].shape)
        output=sess.run(network.Dense1,feed_dict={network.imageInput:each[0]})
        reduced_list.append(output)
    reduced=np.concatenate(reduced_list)
    
    instance=hdbscan.HDBSCAN(min_cluster_size=5,prediction_data=True)
    instance.fit(np.float64(reduced[0:10000]))
    labels=instance.labels_    
    
    prob=hdbscan.all_points_membership_vectors(instance)
    
    loss=sess.run(network.loss,feed_dict={network.imageInput:train[0][0],network.labels:train[0][1]})


    
    