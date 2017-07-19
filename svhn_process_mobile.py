# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:36:42 2016

@author: Sergey
"""


import os, cv2, numpy as np
from bboxfinder import BBoxFinder
from digit_predictor import DigitPredictor
from keras.datasets import cifar10
from random import randint
train_folder = "mobile"
#test_folder = r"F:\SVHN\test\cropped32"
#height, width, channels = img.shape
 #cv2.imwrite("test.jpg",img_np)

#(_c1, _c2),(cifar_test, _) = cifar10.load_data()
label_file = os.path.join(train_folder,"labels.csv")
test_lbls = np.genfromtxt(label_file,delimiter=",",dtype=np.int32)

y_test1 = test_lbls[:,1:2]
y_test2 = test_lbls[:,2:3]
y_test3 = test_lbls[:,3:4]
y_test4 = test_lbls[:,4:5]
y_test5 = test_lbls[:,5:] 

#print "Actual type: ",type(y_test1[0]) 
 
#target_width = 72.0
#target_height = 48.0
#show_width = 720
#show_height = 480
#resized_img = cv2.resize(img,(int(target_width),int(target_height)))
bf = BBoxFinder()

dp = DigitPredictor()
#resize_ratio_h = height / target_height
#resize_ratio_w = width / target_width
 #cv2.imshow("Received image", img_np)
 #cv2.waitKey(0)
files = [f for f in os.listdir(train_folder) if f.endswith(".jpg")]
total_files = len(files)
correct_predictions = 0 
precropped_correct_predictions = 0 
precropped_incorrect_predictions = 0         
for x in range(1,len(files)+1):
    #filename = str(x) + ".png"
    filename = str(x) + ".jpg"
    fullname = os.path.join(train_folder,filename)
    img = cv2.imread(fullname)
    img = cv2.resize(img,(96,64))
    x1,x2,y1,y2 = bf.predictBox(img)
#    try:
    #idx = int(x.split(".")[0])
    correct = test_lbls[x - 1,1:]
#        # testing on pre_cropped_img
#        pre_cropped = os.path.join(test_folder,x)
#        pre_cropped_img = cv2.imread(pre_cropped)
#        pre_cropped_pred = dp.predictDigits(pre_cropped_img,True)
#    except:
#        #cifar_idx = randint(1, cifar_test.shape[0])
#        #pre_cropped_img = cifar_test[cifar_idx].transpose(1,2,0)
#        #pre_cropped_pred = dp.predictDigits(pre_cropped_img)
#        pre_cropped_pred = np.zeros((5),dtype=np.int32)
#        correct = np.zeros((5),dtype=np.int32)
    if x2 - x1 < 3 or y2 - y1 < 3:
         #result = "No digits"
         #print "No digits"
         img = cv2.resize(img,(192,128))
         result = np.zeros((5),dtype=np.int32)
    else:
         #crop_img = resized_img[y1:y2, x1:x2] 
         height, width, channels = img.shape
         y1 = max(0, y1 - 2)
         y2 = min(y2 + 2, height)
         x1 = max(0, x1 - 2)
         x2 = min(x2 + 2, width)
         
         crop_img = img[y1:y2, x1:x2]
         img = cv2.resize(img,(192,128))
         #print "%d %d %d %d" % (x1,x2,y1,y2)
         x1 *= 2
         x2 *= 2
         y1 *= 2
         y2 *= 2
         #print "After resize  %d %d %d %d" % (x1,x2,y1,y2)
         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
         result = dp.predictDigits(crop_img)
         #print result
         #print result.shape
         #correct = test_lbls[x-1,1:]
         #print test_lbls.shape
    #print x," Prediction: ",result," Correct: ",correct," Precropped: ", pre_cropped_pred
    if np.all(result == correct):  
        correct_predictions += 1
#    if np.all(pre_cropped_pred == correct):  
        precropped_correct_predictions += 1
#        if correct_predictions <= 12:
#            #show_img = cv2.resize(pre_cropped_img,(96,96))
        print x," CORRECT"
        result_print = "".join(['0' if y == 10 else str(y) for y in [x for x in list(result) if x != 0]])
        if result_print:
            print int(result_print)
        else:
            print "No digits"
        cv2.imshow("Img",img)
        cv2.waitKey(0)
    else:
        precropped_incorrect_predictions += 1
#        if precropped_incorrect_predictions <= 12:
            #show_img = cv2.resize(pre_cropped_img,(96,96))
        print x," INCORRECT"
#            precrop_pred = "CORRECT" if np.all(pre_cropped_pred == correct) else "INCORRECT"
#            print "PRE_CROP %s %d" % (precrop_pred, int("".join(['0' if y == 10 else str(y) for y in [x for x in list(pre_cropped_pred)]])))
        result_print = "".join(['0' if y == 10 else str(y) for y in [x for x in list(result) if x != 0]])
        if result_print:
            print int(result_print)
        else:
            print "No digits"
        cv2.imshow("Img",img)
        cv2.waitKey(0)
    #cv2.imshow("predicted_boundary",show_img)
    #cv2.waitKey(0)         
#     result = dp.predictDigits(crop_img)
#ratio = float(correct_predictions)/ total_files
#pre_cropped_ratio = float(precropped_correct_predictions)/ total_files
#print ratio, " Pre-crop: ", pre_cropped_ratio