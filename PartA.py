import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

NUM_EPOCHS = 2

def loadDataH5():
    with h5py.File('data1.h5','r') as hf:
        trainX = np.array(hf.get('trainX'))
        trainY = np.array(hf.get('trainY'))
        valX = np.array(hf.get('valX'))
        valY = np.array(hf.get('valY'))
        print (trainX.shape,trainY.shape)
        print (valX.shape,valY.shape)
    return trainX, trainY, valX, valY

trainX, trainY, testX, testY = loadDataH5()

def buildBaseCNN(width, height, depth, classes):
    model = tf.keras.Sequential()
    inputShape= (height, width, depth)

    # define the first layer
    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same",input_shape=inputShape, activation='relu'))

    #pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    #flatten the results
    model.add(tf.keras.layers.Flatten())

    #feed the flattened results into full connected layers
    model.add(tf.keras.layers.Dense(500, activation='relu'))

    #softmax classifier
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))

    return model

def cnn_model_1(width, height, depth, classes):
    print("into the cnn model 1----")
    inputShape = (width, height, depth)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=inputShape, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))
        
    # return the constructed network architecture
    return model

def data_aug_explore():
    print("data aug function begins---")
    # resize all images to thie width and height
    width, height= 128, 128
    trainDataDir = '/content/dogs_cats/data/train'
    # print(trainDataDir.shape)
    validationDataDir= '/content/dogs_cats/data/validation'
    numTrainingSamples = 1020
    numValidationSamples = 340
    NUM_EPOCHS = 50
    batchSize = 64

    opt = keras.optimizers.SGD(lr=0.01)
    model = cnn_model_1(width, height, 3, 17)
    print(model.summary())

    model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])


    # In this example we create an ImageDataGenerator for both training and set images
    # This ImageDataSetGenerator will  normalize the images for us. 
    #trainDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1.0/255)
    # The ImageDataSetGenerator for testing
    #testDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
    
    trainDataGenerator= tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=20, width_shift_range=0.1, shear_range=0.2, zoom_range=0.4, horizontal_flip=True)
    testDataGenerator= tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=20, width_shift_range=0.1, shear_range=0.2, zoom_range=0.4, horizontal_flip=True)

    train_generator = trainDataGenerator.flow(trainX, trainY,batchSize)
    validation_generator = testDataGenerator.flow(testX, testY,batchSize)
  
    H = model.fit(
        train_generator,
        steps_per_epoch= numTrainingSamples / batchSize,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=numValidationSamples / batchSize)
  
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], loc='upper right')
    plt.show()

data_aug_explore()

def initial_run():
    # initialize the optimizer and model
    print("Compiling model...")
    opt = keras.optimizers.SGD(lr=0.01)

    model = buildBaseCNN(width=128, height=128, depth=3, classes=17)

    print(model.summary())

    model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

    #trainthenetwork
    print("Trainingnetwork...")
    history=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=NUM_EPOCHS)

    print ("Test Data Loss and Accuracy: ", model.evaluate(testX, testY))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, NUM_EPOCHS), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, NUM_EPOCHS), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], loc='upper right')
    plt.show()