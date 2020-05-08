import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
from contextlib import redirect_stdout

#NUM_EPOCHS = 20

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

def plotAccLoss(H, NUM_EPOCHS):

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

def save_model_summary(modelname, model):
    filename = modelname+'_model_summary.txt'
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            model.summary()

def baselineCNN(width, height, depth, classes):
    """Implementation of baseline CNN with single convolutional layer, 
    single pooling layer, fully connected layer and softmax layer.
    """

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

def CNN_Model_1(width, height, depth, classes):
    """Implementation of baseline CNN with single convolutional layer, 
    single pooling layer, fully connected layer and softmax layer."""

    inputShape = (width, height, depth)
    model = tf.keras.models.Sequential()
    
    # define the first convolutional layer
    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same",input_shape=inputShape, activation='relu'))

    #pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #second convolutional layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    
    #pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    #flatten the results
    model.add(tf.keras.layers.Flatten())

    #feed the flattened results into full connected layers
    model.add(tf.keras.layers.Dense(500, activation='relu'))

    #softmax classifier
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))
    
    return model

def CNN_Model_2(width, height, depth, classes):
    """CNN Model 2"""

    inputShape = (width, height, depth)
    model = tf.keras.models.Sequential()
    
    #first convolutional layer
    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same",input_shape=inputShape, activation='relu'))

    #first pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #second convolutional layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    
    #second pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    #third convolutional layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    
    #third pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    #flatten the results
    model.add(tf.keras.layers.Flatten())

    #feed the flattened results into full connected layers
    model.add(tf.keras.layers.Dense(500, activation='relu'))

    #apply dropout
    model.add(tf.keras.layers.Dropout(0.5))

    #softmax classifier
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))
    
    return model

def CNN_Model_3(width, height, depth, classes):
    """CNN Model 3"""

    inputShape = (width, height, depth)
    model = tf.keras.models.Sequential()
    
    #first convolutional layer
    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same",input_shape=inputShape, activation='relu'))

    #second convolutional layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))

    #first pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #third convolutional layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))

    #fourth convolutional layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    
    #second pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    #flatten the results
    model.add(tf.keras.layers.Flatten())

    #feed the flattened results into full connected layers
    model.add(tf.keras.layers.Dense(1000, activation='relu'))

    #apply dropout
    model.add(tf.keras.layers.Dropout(0.5))

    #softmax classifier
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))
    
    return model

def CNN_Model_4(width, height, depth, classes):
    """CNN Model 4"""

    inputShape = (width, height, depth)
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same",input_shape=inputShape, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu')) 
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(classes, activation='softmax'))
    
    return model

def CNN_Model_5(width, height, depth, classes):
    """CNN Model 5"""

    inputShape = (width, height, depth)
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same",input_shape=inputShape, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(classes, activation='softmax'))
    
    return model

def CNN_Model_5(width, height, depth, classes):
    """CNN Model 5"""

    inputShape = (width, height, depth)
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same",input_shape=inputShape, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(classes, activation='softmax'))
    
    return model

def CNN_Model_6(width, height, depth, classes):
    """CNN Model 6"""

    inputShape = (width, height, depth)
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same",input_shape=inputShape, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(classes, activation='softmax'))
    
    return model

def data_augmentation_compile_and_train(modelName, batch_size, learning_rate, NUM_EPOCHS):
    print("data aug function begins---")
    # resize all images to thie width and height
    #width, height= 128, 128
    #trainDataDir = '/content/dogs_cats/data/train'
    # print(trainDataDir.shape)
    #validationDataDir= '/content/dogs_cats/data/validation'
    numTrainingSamples = 1020
    numValidationSamples = 340

    batchSize = 64

    opt = keras.optimizers.SGD(lr=learning_rate)
    model = globals()[modelName](width=128, height=128, depth=3, classes=17)

    print(model.summary())
    save_model_summary(modelName+'_data_augmentation_', model)

    model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

    #initialize data generator with various parameters
    trainDataGenerator= tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=90, height_shift_range=0.5, width_shift_range=0.3, shear_range=0.2, zoom_range=[0.4,0.9], horizontal_flip=True)
    testDataGenerator= tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=90, height_shift_range=0.5, width_shift_range=0.3, shear_range=0.2, zoom_range=[0.4,0.9], horizontal_flip=True)

    #assign train, test data to data generator along with the labels
    train_generator = trainDataGenerator.flow(trainX, trainY, batch_size)
    validation_generator = testDataGenerator.flow(testX, testY, batch_size)
  
    #fit the model on real time data-augmentation
    history = model.fit(
        train_generator,
        steps_per_epoch= numTrainingSamples / batch_size,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=numValidationSamples / batch_size)

    print ("Test Data Loss and Accuracy: ", model.evaluate(testX, testY))
  
    plotAccLoss(history, NUM_EPOCHS)


def compile_and_train(modelName, batch_size, learning_rate, NUM_EPOCHS):

    # initialize the optimizer and model
    print("Compiling model...")
    opt = keras.optimizers.SGD(lr=learning_rate)

    model = globals()[modelName](width=128, height=128, depth=3, classes=17)

    print(model.summary())
    save_model_summary(modelName, model)

    model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

    #trainthenetwork
    print("Trainingnetwork...")
    history=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=batch_size,epochs=NUM_EPOCHS)

    print ("Test Data Loss and Accuracy: ", model.evaluate(testX, testY))

    plotAccLoss(history, NUM_EPOCHS)


def compile_train_and_checkpoint():
    
    # initialize the optimizer and model
    print("Compiling model...")
    opt = keras.optimizers.SGD(lr=0.01)

    fields = {'baselineCNN':baselineCNN,'CNN_Model_1':CNN_Model_1}

    for key in fields:
        print("key is ", key)
        model = fields[key](width=128, height=128, depth=3, classes=17)
        #model = baselineCNN(width=128, height=128, depth=3, classes=17)

        print(model.summary())

        model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

        fname = "Testing/"+key+".hdf5"
        print("file name-->","weights.{epoch:02d}-{val_loss:.2f}.hdf5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)

        #trainthenetwork
        print("Trainingnetwork...")
        history=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=NUM_EPOCHS, callbacks=[checkpoint])

        print ("Test Data Loss and Accuracy: ", model.evaluate(testX, testY))

        plotAccLoss(history, NUM_EPOCHS)


def fetch_and_test():
    models = []
    ensemble_models = {'baselineCNN':baselineCNN,'CNN_Model_1':CNN_Model_1}
    for key in ensemble_models:
        print("key is ", key)
        model = baselineCNN(width=128, height=128, depth=3, classes=17)
        model = ensemble_models[key](width=128, height=128, depth=3, classes=17)
        
        fname = "Testing/"+key+".hdf5"
        model.load_weights(fname)
        opt = keras.optimizers.SGD(lr=0.01)
        model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
        models.append(model)
    
    print("models-->", models)
    #outputs = [model.outputs[0] for model in models]
    pred = keras.layers.Average()([model.predict_proba(testX) for model in models])
    print("pred avg---",pred)
    pred = tf.math.argmax(pred, axis = -1)
    print("384 predictions--->", pred)

    print("labels-->", testY.shape)
    
    equality = tf.equal(pred, testY)
    reduce_t = tf.reduce_all(equality)
    print(equality)

    count = (tf.reduce_sum(tf.cast(equality,tf.float32)))
    print(count)
    print(tf.math.divide(count, testY.shape)[0])

    #print(keras.layers.Average()pred)
    #print(pred.shape)
    #y = Average(outputs)

    #scores = model_a.evaluate(testX, testY, verbose=0)
    #print(scores)
    #print (scores[0])

#compile_and_train()
#fetch_and_test()

"""execution for compiling and training CNN models"""
#compile_and_train("baselineCNN", batch_size=32, learning_rate=0.01, NUM_EPOCHS=100)
#compile_and_train("CNN_Model_1", batch_size=32, learning_rate=0.01, NUM_EPOCHS=50)
#compile_and_train("CNN_Model_2", batch_size=32, learning_rate=0.01, NUM_EPOCHS=100)
#compile_and_train("CNN_Model_3", batch_size=32, learning_rate=0.01, NUM_EPOCHS=100)
#compile_and_train("CNN_Model_4", batch_size=32, learning_rate=0.01, NUM_EPOCHS=50)

#compile_and_train("CNN_Model_4", batch_size=32, learning_rate=0.01, NUM_EPOCHS=50)

#data_augmentation_compile_and_train(modelName="CNN_Model_2", batch_size=32, learning_rate=0.01, NUM_EPOCHS=100)

compile_and_train("CNN_Model_5", batch_size=32, learning_rate=0.01, NUM_EPOCHS=50)




