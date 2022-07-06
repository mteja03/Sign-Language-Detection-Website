from keras.models import load_model
from keras import backend as keras_backend
import numpy as np
from keras.preprocessing import image


def predictor(image_path):
        keras_backend.clear_session()

        classifier = load_model('Trained_model.h5')

        # Prediction of  image
        loaded_image = image.load_img(
            image_path, target_size=(64, 64))
        img_array = image.img_to_array(loaded_image)
        img_dims = np.expand_dims(img_array, axis=0)
        classifier_result = classifier.predict(img_dims)
        predicted_char = ''


        #map to the character in the alphabet from one hot encoding.
        for i in range(26):
	        if classifier_result[0][i] == 1:
                        predicted_char = chr(i + 65)

        keras_backend.clear_session()
        return predicted_char
