# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 09:03:43 2016

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
#from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
#import matplotlib.pyplot as plt
#from scipy.misc import toimage
#import cv2
import cPickle as pickle
#from svhn_check import full_predict
#from keras.models import model_from_json

#from keras import backend as K

# load train data

class BBoxFinder:

    def __init__(self):
#        train_folder = r"F:\SVHN\train\regression6496"
#        
#        train_list = []
#        total_train = len([q for q in os.listdir(train_folder) if q.endswith(".png")]) + 1
#        total_test = 14368 + 1
#        
#        for f in range(1,total_train):
#            filename = str(f) + ".png"
#            fullname = os.path.join(train_folder,filename)
#            img = cv2.imread(fullname)
#            # img = img.transpose(2,0,1)
#            train_list.append(img)
#        train_array = numpy.stack(train_list,axis = 3)
#        
#        print train_array.shape
#        
#        
#        # load test data 
#        
#        test_folder = r"F:\SVHN\test\regression_shift"
#        
#        test_list = []
#        
#    #    for f in range(1,total_test):
#    #        filename = str(f) + ".png"
#    #        fullname = os.path.join(test_folder,filename)
#    #        img = cv2.imread(fullname)
#    #        # img = img.transpose(2,0,1)
#    #        test_list.append(img)
#    #    test_array = numpy.stack(test_list,axis = 3)
#    #    
#    #    print test_array.shape
#            
#        X_train = train_array.transpose(3,2,0,1)
#    #    X_test = test_array.transpose(3,2,0,1)
#        
#        X_train = X_train.astype('float32')
#    #    X_test = X_test.astype('float32')
#        X_train = X_train / 255.0
    #    X_test = X_test / 255.0
        
        #datagen = ImageDataGenerator(
        #    featurewise_center=True,
        #    featurewise_std_normalization=True)
        
        #datagen.fit(X_train)
        
#        self.train_mean = numpy.mean(X_train, axis=0)
#        self.train_std = numpy.std(X_train, axis=0)
#        numpy.save("mean_bbox.npy",self.train_mean)
#        numpy.save("std_bbox.npy",self.train_std)
        self.train_mean = numpy.load("mean_bbox.npy")
        self.train_std = numpy.load("std_bbox.npy")
        # print "MEAN: ",self.train_mean.shape
        # print "STD: ",self.train_std.shape
#        X_train -= self.train_mean.astype('float32')
#        X_train /= (self.train_std + 1e-7).astype('float32')
    #    test_mean = numpy.mean(X_test, axis=0)
    #    test_std = numpy.std(X_test, axis=0)
    #    X_test -= test_mean.astype('float32')
    #    X_test /= (test_std + 1e-7).astype('float32')
#        try:
#            json_file = open('bb_model.json','r')
#            loaded_model_json = json_file.read()
#            json_file.close()
#            self.model = model_from_json(loaded_model_json)   
#        except:
        # print "No json file found for the model"
        self.model = Graph()
        self.model.add_input("sequence_input", input_shape = (3,64,96))
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

        self.model.add_node(Dense(1), name="x", input="shared_layers", create_output=True)
        self.model.add_node(Dense(1), name="y", input="shared_layers", create_output=True)
        self.model.add_node(Dense(1), name="width", input="shared_layers", create_output=True)
        self.model.add_node(Dense(1), name="height", input="shared_layers", create_output=True)       
    #    epochs = 50
    #    lrate = 0.01
    #    decay = lrate / (epochs * 5)
    #    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
                
        self.model.compile(loss={"x": "mean_squared_error", 
                                  "y": "mean_squared_error",
                                  "width": "mean_squared_error",
                                  "height": "mean_squared_error"},
                                  optimizer='rmsprop')
    #    print(model.summary())
        print "Model loaded"
    # logfile = "log_regression.txt"
    # lf = open(logfile,"w")
    
        pickle_filename = 'svhn_regression_6496_8layers.pkl'
    #pickle_filename_shift = 'svhn_regression_small_shift.pkl'
        with open(pickle_filename, 'rb') as infile:
            weights = pickle.load(infile)
        self.model.set_weights(weights)
        model_json = self.model.to_json()
        with open("bb_model.json","w") as json_file:
            json_file.write(model_json)
        
        
    def predictBox(self, img):
        img = img.astype('float32')
        img = img / 255.0
        img = img.transpose(2,0,1)
        ### DEBUG
        #self.train_mean = numpy.load("mobile/mean.npy")
        #self.train_std = numpy.load("mobile/std.npy")
        ### DEBUG
        img -= self.train_mean.astype('float32')
        img /= (self.train_std + 1e-7).astype('float32')
        img_pred = img[numpy.newaxis]
        pre_dict = self.model.predict({"sequence_input": img_pred}, batch_size=1, verbose=0)
        img = img.transpose(1,2,0)
        height, width, _ = img.shape
        #orig_img = cv2.cvtColor(orig_img,cv2.COLOR_RGB2BGR)
        #resized_img = cv2.resize(orig_img,(100,60))
        x1 = max(int(pre_dict["x"][0] - pre_dict["width"][0] / 2),0)
        x2 = min(int(pre_dict["x"][0] + pre_dict["width"][0] / 2),width)
        y1 = max(int(pre_dict["y"][0] - pre_dict["height"][0] / 2),0)
        y2 = min(int(pre_dict["y"][0] + pre_dict["height"][0] / 2),height)
        # expand bounding box
#        x1 = x1 - 5 if x1 - 5 > 0 else 0
#        x2 = x2 + 5 if x2 + 5 < width else width
#        y1 = y1 - 5 if y1 - 5 > 0 else 0
#        y2 = y2 + 5 if y2 + 5 < height else height
        #cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        #cv2.imshow("predicted_boundary",img)
        #cv2.waitKey(0)
        return x1,x2,y1,y2
#    print "Pos1 Predicted : ", pre_dict["x"], "Expected: ", y_test1[i:i+10]
#    print "Pos2 Predicted : ", pre_dict["y"], "Expected: ", y_test2[i:i+10]
#    print "Pos3 Predicted : ", pre_dict["width"], "Expected: ", y_test3[i:i+10]
#    print "Pos4 Predicted : ", pre_dict["height"], "Expected: ", y_test4[i:i+10]
if __name__ == "__main__":
    bbox = BBoxFinder()