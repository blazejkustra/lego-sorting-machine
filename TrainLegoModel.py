import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator

################# Parameters #####################

path = "learning-data"  # folder with all the class folders
labels_file = 'labels.csv'  # file with all names of classes
batch_size_val = 50  # how many to process together
steps_per_epoch_val = 2000
epochs_val = 5
img_size = 32
image_dimensions = (img_size, img_size, 3)
test_ratio = 0.2  # if 1000 images split will 200 for testing
validation_ratio = 0.2  # if 1000 images 20% of remaining 800 will be 160 for validation

################# Import images ##################


def DeleteAllHiddenFiles(array):
    for i in range(0, len(array)-1):
        if array[i].startswith('.'):
            array.pop(i)
    return array


images = []
class_numbers = []
class_ids = DeleteAllHiddenFiles(os.listdir(path))

number_of_classes = len(class_ids)
print("Total Classes Detected: ", class_ids)
print("Importing Classes...")

for class_id in class_ids:
    pictures = DeleteAllHiddenFiles(os.listdir(path + "/" + str(class_id)))

    for picture in pictures:
        img = cv2.imread(path + "/" + str(class_id) + "/" + picture)
        img = cv2.resize(img, (32, 32))
        images.append(img)
        class_numbers.append(class_id)

    print(class_id, end=" ")

print(" ")

images = np.array(images)
class_numbers = np.array(class_numbers)

################# Split data #####################

X_train, X_test, y_train, y_test = train_test_split(images, class_numbers, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio)

# X_train = ARRAY OF IMAGES TO TRAIN
# y_train = CORRESPONDING CLASS ID

################# Data validation ################

print("Data Shapes:")
print("Train", end=" ")
print(X_train.shape, y_train.shape)
print("Validation", end=" ")
print(X_validation.shape, y_validation.shape)
print("Test", end=" ")
print(X_test.shape, y_test.shape)
assert (X_train.shape[0] == y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert (X_validation.shape[0] == y_validation.shape[0]), "The number of images in not equal to the number of lables in validation set"
assert (X_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert (X_train.shape[1:] == image_dimensions), " The dimesions of the Training images are wrong "
assert (X_validation.shape[1:] == image_dimensions), " The dimesionas of the Validation images are wrong "
assert (X_test.shape[1:] == image_dimensions), " The dimesionas of the Test images are wrong"

################# Reads scv file #################

labels = pd.read_csv(labels_file)
print("label file shape: ", labels.shape)

############ Displays sample images ##############

num_of_samples = []
cols = 5

fig, axs = plt.subplots(nrows=number_of_classes, ncols=cols, figsize=(5, number_of_classes))
fig.tight_layout()
#
#
# for j, row in labels.iterrows():
#     x_selected1 = X_train[y_train == int(row["ClassId"])]
#     x_selected2 = X_train[y_train == row["ClassId"]]
#     x_selected3 = X_train[y_train == j]
#
#
#     print(x_selected1)
#     print(x_selected2)
#     print(x_selected3)
#     print(y_train)
#     print()




for i in range(cols):
    for j, row in labels.iterrows():
        x_selected = X_train[y_train == str(row["ClassId"])]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(row["ClassId"]) + "-" + str(row["Name"]))
            num_of_samples.append(len(x_selected))

########## Number of images plot ###############

print("number of images of each class: ", num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, number_of_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

############## Pre-processing image ###############


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)  # CONVERT TO GRAYSCALE
    img = equalize(img)  # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img / 255  # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img


X_train = np.array(list(map(preprocessing, X_train)))  # ITERATE AND PREPROCESS ALL IMAGES
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
cv2.imshow("GrayScale Image", X_train[random.randint(0, len(X_train) - 1)])  # TO CHECK IF THE TRAINING IS DONE PROPERLY

############## Add depth of 1 #####################

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

########### Augmentation of images ################

data_generator = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.1,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                                    shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                                    rotation_range=10)  # DEGREES
data_generator.fit(X_train)
batches = data_generator.flow(X_train, y_train, batch_size=20)  # REQUESTING DATA GENRATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAED EACH TIME ITS CALLED
X_batch, y_batch = next(batches)

# TO SHOW AGMENTED IMAGE SAMPLES
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(image_dimensions[0], image_dimensions[1]))
    axs[i].axis('off')
plt.show()


def classIDtoVector(y_set):
    y_train_temp = []
    for y in y_set:
        y_train_temp.append(class_ids.index(y))
    return y_train_temp


y_train = to_categorical(y_train, number_of_classes)
y_validation = to_categorical(classIDtoVector(y_validation), number_of_classes)
y_test = to_categorical(classIDtoVector(y_test), number_of_classes)

######## Convulution neural network model #########


def Model():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)  # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
    # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    number_of_nodes = 500  # NO. OF NODES IN HIDDEN LAYERS
    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(image_dimensions[0], image_dimensions[1], 1), activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))  # DOES NOT EFFECT THE DEPTH/NO OF FILTERS

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(number_of_nodes, activation='relu'))
    model.add(Dropout(0.5))  # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dense(number_of_classes, activation='softmax'))  # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#################### Training #####################

model = Model()
print(model.summary())
history = model.fit_generator(data_generator.flow(X_train, y_train, batch_size=batch_size_val),
                              steps_per_epoch=steps_per_epoch_val,
                              epochs=epochs_val,
                              validation_data=(X_validation, y_validation), shuffle=1)

################ Plot results #####################

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

pickle_out = open("model_trained.p", "wb")  # STORE THE MODEL AS A PICKLE OBJECT
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)
