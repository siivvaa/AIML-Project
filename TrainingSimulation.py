
#The following three lines of code are for removing the GPU errors which arise due to incompatibility.
print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
from utlis import *
from sklearn.model_selection import train_test_split

#Collecting data from csv file.
path = 'simData'
data = importDataInfo(path)

#Balancing the data and removing redundant values.
data = balanceData(data,display=True)

#Processes the path of each image and steering angles into a numpy array.
imagesPath, steerings = loadData(path,data)
#print(imagesPath[0], steerings[0]) - Just to show how the images look like

#Splitting the data for training and validation
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#Image Augmentation (can be commented out to show an example of what an augmented image looks like).
img, steerAngle = augmentImage('trial.jpg',0)
plt.imshow(img)
plt.show()

#Preprocessing
prepImg = preProcess(mpimg.imread('trial.jpg'))
plt.imshow(prepImg)
plt.show()

#Creating the model
model = createModel()
model.summary()

#Training the model (Epoch = 2 and steps = 20 just to show how it works)...Original Values = 300,10.
history = model.fit(batchGen(xTrain, yTrain, 100, 1),
                                  steps_per_epoch=300,
                                  epochs=10,
                                  validation_data=batchGen(xVal, yVal, 100, 0),
                                  validation_steps=200)

#Plotting the validation losses
model.save('oldModel.h5')
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
