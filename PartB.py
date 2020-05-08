import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from contextlib import redirect_stdout

NUM_EPOCHS = 50

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

def save_accuracy_score(filename, r, labels):
    path = "accuracy_results/"
    filename = path+filename+".txt"

    accuracy_score = metrics.accuracy_score(r, labels)

    with open(filename, 'w') as f:
        f.write(str(accuracy_score))
        f.close()

def save_model_summary(filename, model):
    path = "model_summary/"
    filename = path+filename+".txt"

    with open(filename, 'w') as f:
        with redirect_stdout(f):
            model.summary()
            f.close

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

def featureExtractionTransferLearning_variant1():
    vggModel = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    print (vggModel.summary())
    save_model_summary("FE_TL_VGG16_LogisticRegression", vggModel)

    featuresTrain = vggModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = vggModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)


    model = LogisticRegression()
    model.fit(featuresTrain, trainY)

    results = model.predict(featuresVal)
    print("results are --->")
    print (metrics.accuracy_score(results, testY))
    save_accuracy_score("FE_TL_VGG16_LogisticRegression", results, testY)


def featureExtractionTransferLearning_variant2():
    vggModel = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    print (vggModel.summary())
    save_model_summary("FE_TL_VGG16_RandomForestClassifier_500", vggModel)

    featuresTrain = vggModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = vggModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)

    # 500 estimators (trees)
    model = RandomForestClassifier(n_estimators=500)

    model.fit(featuresTrain, trainY)
    results = model.predict(featuresVal)
    print("results are --->")
    print (metrics.accuracy_score(results, testY))
    save_accuracy_score("FE_TL_VGG16_RandomForestClassifier_200", results, testY)

def featureExtractionTransferLearning_variant3():
    initialModel = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    print (initialModel.summary())
    save_model_summary("FE_TL_Inception_v3_LogisticRegression", initialModel)

    featuresTrain = initialModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = initialModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)

    model = LogisticRegression()

    model.fit(featuresTrain, trainY)
    results = model.predict(featuresVal)
    print("accuracy: ")
    print (metrics.accuracy_score(results, testY))
    save_accuracy_score("FE_TL_Inception_v3_LogisticRegression", results, testY)


def featureExtractionTransferLearning_variant4():
    initialModel = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    print (initialModel.summary())
    save_model_summary("FE_TL_Inception_v3_RandomForestClassifier_200", initialModel)

    featuresTrain = initialModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = initialModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)

    # 500 estimators (trees)
    model = RandomForestClassifier(n_estimators=200)

    model.fit(featuresTrain, trainY)
    results = model.predict(featuresVal)
    print("accuracy: ")
    print (metrics.accuracy_score(results, testY))
    save_accuracy_score("FE_TL_Inception_v3_RandomForestClassifier_200", results, testY)


def featureExtractionTransferLearning_variant5():
    initialModel = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    print (initialModel.summary())
    save_model_summary("FE_TL_ResNet152V2_LogisticRegression", initialModel)

    featuresTrain = initialModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = initialModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)

    model = LogisticRegression()

    model.fit(featuresTrain, trainY)
    results = model.predict(featuresVal)
    print("accuracy: ")
    print (metrics.accuracy_score(results, testY))
    save_accuracy_score("FE_TL_ResNet152V2_LogisticRegression", results, testY)


def featureExtractionTransferLearning_variant6():
    initialModel = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    print (initialModel.summary())
    save_model_summary("FE_TL_ResNet152V2_RandomForestClassifier_200", initialModel)

    featuresTrain = initialModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = initialModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)

    # 500 estimators (trees)
    model = RandomForestClassifier(n_estimators=200)

    model.fit(featuresTrain, trainY)
    results = model.predict(featuresVal)
    print("accuracy: ")
    print (metrics.accuracy_score(results, testY))
    save_accuracy_score("FE_TL_ResNet152V2_RandomForestClassifier_200", results, testY)


def featureExtractionTransferLearning_variant7():
    """Create a New Model Using a Portion of an Original Model"""

    vggModel = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    portionOfVGG16= tf.keras.Model(inputs=vggModel.input, outputs=vggModel.get_layer('block4_conv2').output)

    print (portionOfVGG16.summary())
    save_model_summary("FE_TL_portionOfVGG16_LogisticRegression", portionOfVGG16)

    featuresTrain = portionOfVGG16.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = portionOfVGG16.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)


    model = LogisticRegression()
    model.fit(featuresTrain, trainY)

    results = model.predict(featuresVal)
    print("results are --->")
    print (metrics.accuracy_score(results, testY))
    save_accuracy_score("FE_TL_portionOfVGG16_LogisticRegression", results, testY)


def featureExtractionTransferLearning_variant8():
    """Create a New Model Using a Portion of an Original Model"""

    initialModel = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    #cut the original network at conv2d_50
    portionOfInitialModel= tf.keras.Model(inputs=initialModel.input, outputs=initialModel.get_layer('conv2d_50').output)

    print (portionOfInitialModel.summary())
    save_model_summary("FE_TL_portionOfInceptionV3_LogisticRegression", portionOfInitialModel)

    featuresTrain = portionOfInitialModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = portionOfInitialModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)


    model = LogisticRegression()
    model.fit(featuresTrain, trainY)

    results = model.predict(featuresVal)
    print("results are --->")
    print (metrics.accuracy_score(results, testY))
    save_accuracy_score("FE_TL_portionOfInceptionV3_LogisticRegression", results, testY)

def featureExtractionTransferLearning_variant9():
    """Create a New Model Using a Portion of an Original Model"""

    initialModel = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    #cut the original network at conv4_block23_2_conv
    portionOfInitialModel= tf.keras.Model(inputs=initialModel.input, outputs=initialModel.get_layer('conv4_block23_2_conv').output)

    print (portionOfInitialModel.summary())
    save_model_summary("FE_TL_portionOfResNet152V2_LogisticRegression", portionOfInitialModel)

    featuresTrain = portionOfInitialModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = portionOfInitialModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)


    model = LogisticRegression()
    model.fit(featuresTrain, trainY)

    results = model.predict(featuresVal)
    print("results are --->")
    print (metrics.accuracy_score(results, testY))
    save_accuracy_score("FE_TL_portionOfResNet152V2_LogisticRegression", results, testY)


def featureExtractionTransferLearning_variant10():
    """Create a New Model Using a Portion of an Original Model"""

    initialModel = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    #cut the original network at conv2d_50
    portionOfInitialModel= tf.keras.Model(inputs=initialModel.input, outputs=initialModel.get_layer('conv2d_50').output)

    print (portionOfInitialModel.summary())
    save_model_summary("FE_TL_portionOfInceptionV3_RandomForestClassifier_500", portionOfInitialModel)

    featuresTrain = portionOfInitialModel.predict(trainX)
    featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)

    featuresVal = portionOfInitialModel.predict(testX)
    featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)


    model = RandomForestClassifier(n_estimators=500)
    model.fit(featuresTrain, trainY)

    results = model.predict(featuresVal)
    print("accuracy: ")
    print (metrics.accuracy_score(results, testY))
    

    save_accuracy_score("FE_TL_portionOfInceptionV3_RandomForestClassifier_500", results, testY)





def fineTuning_Variant1():
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
    
    #stop training when val_loss will not improve
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

def fineTuning_Variant2():
    """
    Details of this variant  
    Phase A: portion of InceptionV3 (till conv2d_39)
    Phase B: unfreeze the convolutional layers block with a very low learning rate
    """
    initialModel = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    #cut the original network at conv2d_39
    portionOfInitialModel= tf.keras.Model(inputs=initialModel.input, outputs=initialModel.get_layer('conv2d_39').output)


    portionOfInitialModel.trainable = False
    model = tf.keras.models.Sequential()
    model.add(portionOfInitialModel)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))  
    model.add(tf.keras.layers.Dense(17, activation='softmax'))

    print ("\n Phase A - Training Fully Connected Layers\n")
    
    print("Compiling model...")
    opt = keras.optimizers.SGD(lr=0.01)
    model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
    
    #stop training when val_loss will not improve
    overfitCallback = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    history=model.fit(trainX, trainY, epochs=NUM_EPOCHS, callbacks=[overfitCallback], batch_size=32, validation_data=(testX, testY))

    plotAccLoss(history, len(history.history['val_loss']))

    print ("\n Phase B  - Fine Tune Fully Connected Layer and Selected Convolutional Layers \n")
    portionOfInitialModel.trainable = True
    trainableFlag = False
    
    for layer in portionOfInitialModel.layers:

        #unfreeze all the layers from block4_conv1 onwards
        if layer.name == 'conv2d_30':
            trainableFlag = True
        layer.trainable = trainableFlag
    portionOfInitialModel.summary()

    for layer in portionOfInitialModel.layers[:-3]:
        layer.trainable=False

    for layer in portionOfInitialModel.layers:
        sp= '  '[len(layer.name):]
        print("sp--->",layer.name,sp,layer.trainable)

    model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.SGD(lr=1e-5),metrics=["accuracy"])
    history =model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))

    plotAccLoss(history, NUM_EPOCHS)



def fineTuning_Variant3():
    vggModel = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    vggModel.trainable = False
    model = tf.keras.models.Sequential()
    model.add(vggModel)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(500, activation='relu')) 
    model.add(tf.keras.layers.Dense(17, activation='softmax'))

    print ("\n Phase A - Training Fully Connected Layers\n")
    
    print("Compiling model...")
    opt = keras.optimizers.SGD(lr=0.01)
    model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
    
    #stop training when val_loss does not improve
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


def fineTuning_Variant4():
    vggModel = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    portionOfVGG16= tf.keras.Model(inputs=vggModel.input, outputs=vggModel.get_layer('block5_conv2').output)

    portionOfVGG16.trainable = False
    model = tf.keras.models.Sequential()
    model.add(portionOfVGG16)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(500, activation='relu')) 
    model.add(tf.keras.layers.Dense(17, activation='softmax'))

    print ("\n Phase A - Training Fully Connected Layers\n")
    
    print("Compiling model...")
    opt = keras.optimizers.SGD(lr=0.001)
    model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
    
    #stop training when val_loss does not improve
    overfitCallback = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    history=model.fit(trainX, trainY, epochs=NUM_EPOCHS, callbacks=[overfitCallback], batch_size=32, validation_data=(testX, testY))

    plotAccLoss(history, len(history.history['val_loss']))

    print ("\n Phase B  - Fine Tune Fully Connected Layer and Selected Convolutional Layers \n")
    portionOfVGG16.trainable = True
    trainableFlag = False
    
    for layer in portionOfVGG16.layers:
        if layer.name == 'block4_conv1':
            trainableFlag = True
        layer.trainable = trainableFlag
    portionOfVGG16.summary()

    for layer in portionOfVGG16.layers[:-3]:
        layer.trainable=False

    for layer in portionOfVGG16.layers:
        sp= '  '[len(layer.name):]
        print("sp--->",layer.name,sp,layer.trainable)
    #print("model summary--->", model.summary())

    model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.SGD(lr=1e-5),metrics=["accuracy"])
    history =model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))

    plotAccLoss(history, NUM_EPOCHS)

#featureExtractionTransferLearning_NN()



#featureExtractionTransferLearning_variant1()
#featureExtractionTransferLearning_variant2()
#featureExtractionTransferLearning_variant3()
#featureExtractionTransferLearning_variant4()
#featureExtractionTransferLearning_variant5()
#featureExtractionTransferLearning_variant6()
#featureExtractionTransferLearning_variant7()
#featureExtractionTransferLearning_variant8()
#featureExtractionTransferLearning_variant9()
#featureExtractionTransferLearning_variant10()

#fineTuning_Variant1()
#fineTuning_Variant2()
#fineTuning_Variant3()
fineTuning_Variant4()