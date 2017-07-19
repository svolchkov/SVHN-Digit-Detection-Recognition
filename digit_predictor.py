# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:37:13 2016

@author: Sergey
"""
import os
import numpy
# import keras.layers.containers as containers
# from keras.layers.core import containers
from keras.models import Graph
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import CSVLogger, modelCheckpoint
#import matplotlib.pyplot as plt
#from scipy.misc import toimage
from keras.utils import np_utils
import cv2
import cPickle as pickle
from svhn_check import full_predict

from keras import backend as K


class DigitPredictor:
    
    def __init__(self):
#        train_folder = r"F:\SVHN\test\cropped32"
#        
#        train_list = []
#        files = len([f for f in os.listdir(train_folder) if f.endswith(".png")]) + 1
#        for f in range(1,files):
#            filename = str(f) + ".png"
#            fullname = os.path.join(train_folder,filename)
#            img = cv2.imread(fullname)
#            # img = img.transpose(2,0,1)
#            train_list.append(img)
#        train_array = numpy.stack(train_list,axis = 3)
#        
#        print train_array.shape
#        
#        # load test data 
#        
#        X_train = train_array.transpose(3,2,0,1)
#
#        
#        X_train = X_train.astype('float32')
#        X_train = X_train / 255.0
#        self.train_mean = numpy.mean(X_train, axis=0)
#        self.train_std = numpy.std(X_train, axis=0)
#        numpy.save("mean_dp.npy",self.train_mean)
#        numpy.save("std_dp.npy",self.train_std)
#        X_train -= self.train_mean.astype('float32')
#        X_train /= (self.train_std + 1e-7).astype('float32')
        self.train_mean = numpy.load("mean_dp.npy")
        self.train_std = numpy.load("std_dp.npy")        

        self.model = Graph()
        self.model.add_input("sequence_input", input_shape = (3,32,32))
        
        num_classes = 11
        
        #shared_self.model = containers.Sequential()
        self.model.add_node(Convolution2D(32,3,3, border_mode = 'same', 
                                activation='relu'), name = "conv1", input="sequence_input",)  #1
        self.model.add_node(Dropout(0.2),name = "drop1", input="conv1") #2
        self.model.add_node(Convolution2D(32,3,3, border_mode = 'same', 
                                activation='relu'),name = "conv2", input="drop1") #3
        self.model.add_node(MaxPooling2D(pool_size=(2,2)),name = "pool1", input="conv2") #4
        self.model.add_node(Convolution2D(64,3,3, border_mode = 'same', 
                                activation='relu'), name = "conv3", input="pool1") #5
        self.model.add_node(Dropout(0.2),name = "drop2", input="conv3") #6
        self.model.add_node(Convolution2D(64,3,3, border_mode = 'same', 
                                activation='relu'), name = "conv4", input="drop2") #7
        self.model.add_node(MaxPooling2D(pool_size=(2,2)), name = "pool2", input="conv4") #8
        self.model.add_node(Convolution2D(128,3,3, border_mode = 'same', 
                                activation='relu'), name = "conv5", input="pool2") #9
        self.model.add_node(Dropout(0.2), name = "drop3", input="conv5") #10
        self.model.add_node(Convolution2D(128,3,3, border_mode = 'same', 
                                activation='relu'), name = "conv6", input="drop3") #11
        self.model.add_node(MaxPooling2D(pool_size=(2,2)), name = "pool3", input="conv6") #12
        self.model.add_node(Convolution2D(160,3,3, border_mode = 'same', 
                                activation='relu'), name = "conv7", input="pool3") #9
        self.model.add_node(Dropout(0.2), name = "drop4", input="conv7") #10
        self.model.add_node(Convolution2D(160,3,3, border_mode = 'same', 
                                activation='relu'), name = "conv8", input="drop4") #11
        self.model.add_node(MaxPooling2D(pool_size=(2,2)), name = "pool4", input="conv8") #12
        self.model.add_node(Flatten(), name = "flat1", input="pool4") #13
        self.model.add_node(Dropout(0.2), name = "drop5", input="flat1") #14
        self.model.add_node(Dense(1024,activation='relu', W_constraint=maxnorm(3)),name = "dense1", input="drop5" ) #15
        self.model.add_node(Dropout(0.2), name = "drop6", input="dense1") #16
        self.model.add_node(Dense(512,activation='relu', W_constraint=maxnorm(3)),name = "dense2", input="drop6" ) #17
        self.model.add_node(Dropout(0.2),name = "shared_layers", input="dense2") #18

        
        #self.model.add_node(shared_self.model, name="shared_layers", input="sequence_input")
        
        self.model.add_node(Dense(num_classes, activation="softmax"), name="output1", input="shared_layers", create_output=True)
        self.model.add_node(Dense(num_classes, activation="softmax"), name="output2", input="shared_layers", create_output=True)
        self.model.add_node(Dense(num_classes, activation="softmax"), name="output3", input="shared_layers", create_output=True)
        self.model.add_node(Dense(num_classes, activation="softmax"), name="output4", input="shared_layers", create_output=True)
        self.model.add_node(Dense(num_classes, activation="softmax"), name="output5", input="shared_layers", create_output=True)
        
        epochs = 50
        lrate = 0.01
        decay = lrate / (epochs * 5)
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        
        with open('svhn_small_8.pkl', 'rb') as infile:
            weights = pickle.load(infile)
            
        self.model.compile(loss={"output1": "categorical_crossentropy", 
                                  "output2": "categorical_crossentropy",
                                  "output3": "categorical_crossentropy",
                                  "output4": "categorical_crossentropy",
                                  "output5": "categorical_crossentropy"},
                                  optimizer=sgd, metrics=['accuracy'])
        # print(self.model.summary())
        self.model.set_weights(weights)        
    
    def predictDigits(self, img, pre_cropped = False):
        target_size = 32.0
        height, width, channels = img.shape
        BLACK = [0,0,0]
        if pre_cropped:
            constant = img
            #constant = constant / 255.0
        else:
            if width > height:
                new_height = int(height * (target_size / width))
                if (target_size - new_height) % 2 == 1:
                    top = int((target_size - new_height) / 2)
                    bottom = int((target_size - new_height) / 2) + 1
                else:
                    top = bottom = int((target_size - new_height) / 2)
                resized_image = cv2.resize(img, (int(target_size), new_height)) 
                constant= cv2.copyMakeBorder(resized_image,top,bottom,0,0,cv2.BORDER_CONSTANT,value=BLACK)
            elif width < height:
                new_width = int(width * (target_size / height))
                if (target_size - new_width) % 2 == 1:
                    left = int((target_size - new_width) / 2)
                    right = int((target_size - new_width) / 2) + 1
                else:
                    right = left = int((target_size - new_width) / 2)
                resized_image = cv2.resize(img, (new_width,int(target_size))) 
                constant= cv2.copyMakeBorder(resized_image,0,0,left,right,cv2.BORDER_CONSTANT,value=BLACK)
            else:
                constant= cv2.resize(img, (int(target_size),int(target_size)))  
        width, height, channels = constant.shape
        #print "resized:",width, height 
        #cv2.imshow("constant", constant)
        #cv2.waitKey(0)
        constant = constant.astype('float32')
        constant = constant / 255.0
        constant = constant.transpose(2,0,1)
        constant -= self.train_mean.astype('float32')
        constant /= (self.train_std + 1e-7).astype('float32')
        constant = constant[numpy.newaxis]
        pre_dict = self.model.predict({"sequence_input": constant}, batch_size=1, verbose=0)
        p1= numpy.argmax(pre_dict["output1"],axis=1)
        p2= numpy.argmax(pre_dict["output2"],axis=1)
        p3= numpy.argmax(pre_dict["output3"],axis=1)
        p4= numpy.argmax(pre_dict["output4"],axis=1)
        p5= numpy.argmax(pre_dict["output5"],axis=1)
        #print "Prediction type: ", type(p1)
        #pred = [p1,p2,p3,p4,p5]
        pred = numpy.array([p1,p2,p3,p4,p5])
        #pred = [str(x) for x in pred if x != 0]
        #pred = [x if x != '10' else '0' for x in pred]
        #return ''.join(pred)
        return pred.T[0]

if __name__ == '__main__':
    dp = DigitPredictor()