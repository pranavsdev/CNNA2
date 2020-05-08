import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

NUM_EPOCHS = 30

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

def featureExtractionTransferLearning_RF():
    vggModel = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    print (vggModel.summary())
    featuresTrain = vggModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = vggModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics

    #model = RandomForestClassifier(200)
    model = LogisticRegression()
    model.fit(featuresTrain, trainY)
    results = model.predict(featuresVal)
    print("results are --->")
    print (metrics.accuracy_score(results, testY))


def baselineCNN(width, height, depth, classes):
    model = tf.keras.Sequential()

    """ 
    inputShape= (height, width, depth)

    # # define the first layer
    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same",input_shape=inputShape, activation='relu'))

    # #pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    """
    # flatten the results
    #model.add(tf.keras.layers.Flatten())

    #feed the flattened results into full connected layers
    model.add(tf.keras.layers.Dense(500, activation='relu'))

    #softmax classifier
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))

    return model

def featureExtractionTransferLearning_NN():
    vggModel = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    print (vggModel.summary())
    featuresTrain = vggModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    print("shape of features train is --->", featuresTrain.shape)

    featuresVal = vggModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)

    # initialize the optimizer and model
    print("Compiling model...")
    opt = keras.optimizers.SGD(lr=0.01)

    model = baselineCNN(width=128, height=128, depth=3, classes=17)

    model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

    #trainthenetwork
    print("Trainingnetwork...")
    history=model.fit(featuresTrain,trainY,validation_data=(featuresVal,testY),batch_size=32,epochs=NUM_EPOCHS)
    print(model.summary())

    print ("Test Data Loss and Accuracy: ", model.evaluate(featuresVal, testY))

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


#featureExtractionTransferLearning_NN()

def fineTuning():
    vggModel = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    vggModel.trainable = False
    model = tf.keras.models.Sequential()
    model.add(vggModel)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))  
    model.add(tf.keras.layers.Dense(17, activation='softmax'))

    print ("\n Phase A - Training Fully Connected Layers\n")
    
    print("Compiling model...")
    opt = keras.optimizers.SGD(lr=0.01)
    model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
    
    #usualCallback = EarlyStopping()
    overfitCallback = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    history=model.fit(trainX, trainY, epochs=NUM_EPOCHS, callbacks=[overfitCallback], batch_size=32, validation_data=(testX, testY))

    plotAccLoss(history, len(history.history['val_loss']))

    print ("\n Phase B  - Fine Tune Fully Connected Layer and Selected Convolutional Layers \n")
    vggModel.trainable = True
    trainableFlag = False
    
    for layer in vggModel.layers:
        if layer.name == 'block4_conv1':
            trainableFlag = True
        layer.trainable = trainableFlag
    vggModel.summary()

    for layer in vggModel.layers[:-3]:
        layer.trainable=False

    for layer in vggModel.layers:
        sp= '  '[len(layer.name):]
        print("sp--->",layer.name,sp,layer.trainable)
    #print("model summary--->", model.summary())

    model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.SGD(lr=1e-5),metrics=["accuracy"])
    history =model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))

    plotAccLoss(history, NUM_EPOCHS)


#featureExtractionTransferLearning_NN()
fineTuning()