import numpy, cv2,os
train_folder = "mobile"
train_list = []
total_train = len([q for q in os.listdir(train_folder) if q.endswith(".jpg")]) + 1
for f in range(1,total_train):
    filename = str(f) + ".jpg"
    fullname = os.path.join(train_folder,filename)
    img = cv2.imread(fullname)
    if img.shape != (2340,4160,3):
        img = numpy.transpose(img,(1,0,2))
    if img.shape != (2340,4160,3):
        img = cv2.resize(img,(4160,2340))
    img = cv2.resize(img,(96,64))
    # img = img.transpose(2,0,1)
    train_list.append(img)
    print img.shape
train_array = numpy.stack(train_list,axis = 3)

print train_array.shape
X_train = train_array.transpose(3,2,0,1)
#    X_test = test_array.transpose(3,2,0,1)

X_train = X_train.astype('float32')
#    X_test = X_test.astype('float32')
X_train = X_train / 255.0
#    X_test = X_test / 255.0

#datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True)

#datagen.fit(X_train)

train_mean = numpy.mean(X_train, axis=0)
train_std = numpy.std(X_train, axis=0)
print "MEAN: ",train_mean.shape
print "STD: ",train_std.shape
mean_file = os.path.join("mobile","mean.npy")
std_file = os.path.join("mobile","std.npy")
numpy.save(mean_file,train_mean)
numpy.save(std_file,train_std)
