#imports
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten, Dense, Dropout
from sklearn import metrics
from keras import optimizers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def model_maker():
    """
    The function is used to create the Sequential model and add the respective layers of convolution and MaxPooling2D with the activate function of relu. Input shape is (64,3,3) and with SGD optimizer. It returns the model.
    """

    # Initialing the CNN
    classifier = Sequential()

    # Step 1 - Convolution Layer
    classifier.add(Convolution2D(32, 3,  3, input_shape = (64, 64, 3), activation = 'relu'))
    #classifier.add(Activation = 'relu')
    #step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size =(2,2)))

    # Adding second convolution layer
    classifier.add(Convolution2D(32, 3,  3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size =(2,2)))

    #Adding 3rd Concolution Layer
    classifier.add(Convolution2D(64, 3,  3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size =(2,2)))


    #Step 3 - Flattening
    classifier.add(Flatten())

    #Step 4 - Full Connection
    classifier.add(Dense(256, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(26, activation = 'softmax'))

    #Compiling The CNN
    classifier.compile(
                  optimizer = optimizers.SGD(lr = 0.01),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    return classifier

classifier = model_maker()

#Generator of training images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#Generator of test images
test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(
        'mydata/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'mydata/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


#Model being fit to the training data and tested on the test dataset.
model = classifier.fit_generator(
        training_set,
        steps_per_epoch=800,
        epochs=1,
        validation_data = test_set,
        validation_steps = 6500
      )

'''#Saving the model
import h5py
classifier.save('Trained_model.h5')'''

test_steps_per_epoch = np.math.ceil(test_set.samples / test_set.batch_size)

#predictions for all the images of test dataset
predictions = classifier.predict_generator(test_set, steps=test_steps_per_epoch)

#actual class labels
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())

#predicted class labels from the one hot encoding
predicted_classes = np.argmax(predictions, axis=1)

#  metrics include accuracy,precision,recall,f-score
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

#Create confusion matrix from the actual and predicted classes
conf_matrix = metrics.confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
conf_matrix_display = metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
#plot for the matrix
figure, axes = plt.subplots(figsize=(10,10))
conf_axes = conf_matrix_display.plot(ax=axes)
plt.show()


# summarize history for accuracy
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()