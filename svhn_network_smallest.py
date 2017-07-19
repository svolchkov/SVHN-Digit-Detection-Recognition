# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 19:12:33 2016

@author: Sergey
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 13:51:15 2016

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
from keras.callbacks import CSVLogger, ModelCheckpoint
#import matplotlib.pyplot as plt
#from scipy.misc import toimage
from keras.utils import np_utils
import cv2
import cPickle as pickle
from svhn_check import full_predict, full_predict_all

from keras import backend as K

# load train data

train_folder = r"F:\SVHN\train\cropped32"

train_list = []

for f in range(1,33402):
    filename = str(f) + ".png"
    fullname = os.path.join(train_folder,filename)
    img = cv2.imread(fullname)
    # img = img.transpose(2,0,1)
    train_list.append(img)
train_array = numpy.stack(train_list,axis = 3)

print train_array.shape

label_file = os.path.join(train_folder,"labels.csv")
train_lbls = numpy.genfromtxt(label_file,delimiter=",",dtype=numpy.int32)

y_train1 = train_lbls[:,1:2]
y_train2 = train_lbls[:,2:3]
y_train3 = train_lbls[:,3:4]
y_train4 = train_lbls[:,4:5]
y_train5 = train_lbls[:,5:]

print "y_train1 ", y_train1.shape
print "y_train2 ", y_train2.shape
print "y_train3 ", y_train3.shape
print "y_train4 ", y_train4.shape
print "y_train5 ", y_train5.shape

# load test data 

test_folder = r"F:\SVHN\test\cropped32"

test_list = []

for f in range(1,13068):
    filename = str(f) + ".png"
    fullname = os.path.join(test_folder,filename)
    img = cv2.imread(fullname)
    # img = img.transpose(2,0,1)
    test_list.append(img)
test_array = numpy.stack(test_list,axis = 3)

print test_array.shape

label_file = os.path.join(test_folder,"labels.csv")
test_lbls = numpy.genfromtxt(label_file,delimiter=",",dtype=numpy.int32)

y_test1 = test_lbls[:,1:2]
y_test2 = test_lbls[:,2:3]
y_test3 = test_lbls[:,3:4]
y_test4 = test_lbls[:,4:5]
y_test5 = test_lbls[:,5:]

print "y_test1 ", y_test1.shape
print "y_test2 ", y_test2.shape
print "y_test3 ", y_test3.shape
print "y_test4 ", y_test4.shape
print "y_test5 ", y_test5.shape


X_train = train_array.transpose(3,2,0,1)
X_test = test_array.transpose(3,2,0,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train1 = np_utils.to_categorical(y_train1)
y_train2 = np_utils.to_categorical(y_train2)
y_train3 = np_utils.to_categorical(y_train3)
y_train4 = np_utils.to_categorical(y_train4)
y_train5 = np_utils.to_categorical(y_train5)
y_test1 = np_utils.to_categorical(y_test1)
y_test2 = np_utils.to_categorical(y_test2)
y_test3 = np_utils.to_categorical(y_test3)
y_test4 = np_utils.to_categorical(y_test4)
y_test5 = np_utils.to_categorical(y_test5)

#datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True)

#datagen.fit(X_train)

train_mean = numpy.mean(X_train, axis=0)
train_std = numpy.std(X_train, axis=0)
X_train -= train_mean.astype('float32')
X_train /= (train_std + 1e-7).astype('float32')
test_mean = numpy.mean(X_test, axis=0)
test_std = numpy.std(X_test, axis=0)
X_test -= test_mean.astype('float32')
X_test /= (test_std + 1e-7).astype('float32')

num_classes = y_test5.shape[1]

model = Graph()
model.add_input("sequence_input", input_shape = (3,32,32))

#shared_model = containers.Sequential()
model.add_node(Convolution2D(32,3,3, border_mode = 'same', 
                        activation='relu'), name = "conv1", input="sequence_input",)  #1
model.add_node(Dropout(0.2),name = "drop1", input="conv1") #2
model.add_node(Convolution2D(32,3,3, border_mode = 'same', 
                        activation='relu'),name = "conv2", input="drop1") #3
model.add_node(MaxPooling2D(pool_size=(2,2)),name = "pool1", input="conv2") #4
model.add_node(Convolution2D(64,3,3, border_mode = 'same', 
                        activation='relu'), name = "conv3", input="pool1") #5
model.add_node(Dropout(0.2),name = "drop2", input="conv3") #6
model.add_node(Convolution2D(64,3,3, border_mode = 'same', 
                        activation='relu'), name = "conv4", input="drop2") #7
model.add_node(MaxPooling2D(pool_size=(2,2)), name = "pool2", input="conv4") #8
model.add_node(Convolution2D(128,3,3, border_mode = 'same', 
                        activation='relu'), name = "conv5", input="pool2") #9
model.add_node(Dropout(0.2), name = "drop3", input="conv5") #10
model.add_node(Convolution2D(128,3,3, border_mode = 'same', 
                        activation='relu'), name = "conv6", input="drop3") #11
model.add_node(MaxPooling2D(pool_size=(2,2)), name = "pool3", input="conv6") #12
model.add_node(Convolution2D(160,3,3, border_mode = 'same', 
                        activation='relu'), name = "conv7", input="pool3") #9
model.add_node(Dropout(0.2), name = "drop4", input="conv7") #10
model.add_node(Convolution2D(160,3,3, border_mode = 'same', 
                        activation='relu'), name = "conv8", input="drop4") #11
model.add_node(MaxPooling2D(pool_size=(2,2)), name = "pool4", input="conv8") #12
model.add_node(Flatten(), name = "flat1", input="pool4") #13
model.add_node(Dropout(0.2), name = "drop5", input="flat1") #14
model.add_node(Dense(1024,activation='relu', W_constraint=maxnorm(3)),name = "dense1", input="drop5" ) #15
model.add_node(Dropout(0.2), name = "drop6", input="dense1") #16
model.add_node(Dense(512,activation='relu', W_constraint=maxnorm(3)),name = "dense2", input="drop6" ) #17
model.add_node(Dropout(0.2),name = "shared_layers", input="dense2") #18

#model.add_node(shared_model, name="shared_layers", input="sequence_input")

model.add_node(Dense(num_classes, activation="softmax"), name="output1", input="shared_layers", create_output=True)
model.add_node(Dense(num_classes, activation="softmax"), name="output2", input="shared_layers", create_output=True)
model.add_node(Dense(num_classes, activation="softmax"), name="output3", input="shared_layers", create_output=True)
model.add_node(Dense(num_classes, activation="softmax"), name="output4", input="shared_layers", create_output=True)
model.add_node(Dense(num_classes, activation="softmax"), name="output5", input="shared_layers", create_output=True)

total_rounds = 5
epochs = 50
lrate = 0.001
decay = lrate / (total_rounds * epochs)
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

pickle_filename = 'svhn_small_8.pkl'

with open(pickle_filename, 'rb') as infile:
    weights = pickle.load(infile)



#params = model.trainable_weights + model.non_trainable_weights
#if len(params) != len(weights):
#    raise Exception('You called `set_weights(weights)` on layer "' + model.name +
#                    '" with a  weight list of length ' + str(len(weights)) +
#                    ', but the layer was expecting ' + str(len(params)) +
#                    ' weights. Provided weights: ' + str(weights))
#weight_value_tuples = []
#param_values = K.batch_get_value(params)
#for pv, p, w in zip(param_values, params, weights):
#    if pv.shape != w.shape:
#        raise Exception('Layer weight shape ' +
#                        str(pv.shape) +
#                        ' not compatible with '
#                        'provided weight shape ' + str(w.shape))
#    weight_value_tuples.append((p, w))
#K.batch_set_value(weight_value_tuples)
    
model.compile(loss={"output1": "categorical_crossentropy", 
                          "output2": "categorical_crossentropy",
                          "output3": "categorical_crossentropy",
                          "output4": "categorical_crossentropy",
                          "output5": "categorical_crossentropy"},
                          optimizer=sgd, metrics=['accuracy'])
print(model.summary())
model.set_weights(weights)
#validation_generator = datagen.flow(X_test, {"output1": y_test1, 
#                          "output2": y_test2,
#                          "output3": y_test3,
#                          "output4": y_test4,
#                          "output5": y_test5},batch_size=64)
#model.fit_generator(datagen.flow(X_train, {"output1": y_train1, 
#                          "output2": y_train2,
#                          "output3": y_train3,
#                          "output4": y_train4,
#                          "output5": y_train5}, batch_size=64), validation_data = validation_generator,
#                    samples_per_epoch=len(X_train), nb_epoch=epochs, nb_val_samples=1800, verbose=2)
 
#filepath="svhn-weight-improvement-{epoch:02d}-{loss:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath,monitor='loss', verbose=1,save_best_only=True,
#                             mode='max')
#csv_logger = CSVLogger('training.log', append=True) 
#callbacks_list=[checkpoint]
#callbacks_list=[csv_logger]
logfile = "log_smallest.txt"
lf = open(logfile,"w")

for x in range(total_rounds):
    print "Round ",x
#    model.set_weights(weights)
#    model.fit({"sequence_input": X_train, "output1": y_train1, "output2": y_train2,
#                   "output3": y_train3,"output4": y_train4,"output5": y_train5}, 
#                   validation_data=({"sequence_input": X_test, "output1": y_test1, "output2": y_test2,
#                   "output3": y_test3,"output4": y_test4,"output5": y_test5}), nb_epoch=epochs, 
#                  batch_size=32, verbose=2)
    #    #scores=model.evaluate(X_test,y_test,verbose=0)
#    scores = model.evaluate({"sequence_input": X_test, "output1": y_test1, "output2": y_test2,
#               "output3": y_test3,"output4": y_test4,"output5": y_test5},verbose=0)                    
    #    
    #    #model.save_weights("svhn_standard_small.h5",overwrite=True)
#    weights = model.get_weights()
#    with open(pickle_filename, 'wb') as outfile:
#            pickle.dump(weights, outfile, pickle.HIGHEST_PROTOCOL)
#    print >>lf,"Got to epoch %d" % ((x+1) * 10) 
#    print >>lf,("Accuracy: %.2f%% %.2f%% %.2f%% %.2f%% %.2f%%" % (scores[6]*100, scores[7]*100, 
#                                                            scores[8]*100, scores[9]*100, 
#                                                            scores[10]*100))
    #full_predict(X_test, model, y_test1, y_test2, y_test3, y_test4, y_test5)
    final_acc = full_predict(X_test, model, y_test1, y_test2, y_test3, y_test4, y_test5)
#    print >>lf,"Final accuracy: %f" % final_acc
    print "Final accuracy: %f" % final_acc
#    print >>lf,""
#    lf.flush()
lf.close()