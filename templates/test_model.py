from keras.preprocessing import image
import numpy as np
from keras.models import load_model

img_size_width, img_size_height = 64, 64

classifier = load_model('Trained_model.h5')

# Prediction of single image

test_image = image.load_img('./predicting_data/2.png', target_size=(64, 64))
# test_image = image.load_img('./predicting_data/example_0.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
classifier_result = classifier.predict(test_image)
# training_set.class_indices
print('Predicted Sign is:')
print('')
if classifier_result[0][0] == 1:
    print('A')
elif classifier_result[0][1] == 1:
    print('B')
elif classifier_result[0][2] == 1:
    print('C')
elif classifier_result[0][3] == 1:
    print('D')
elif classifier_result[0][4] == 1:
    print('E')
elif classifier_result[0][5] == 1:
    print('F')
elif classifier_result[0][6] == 1:
    print('G')
elif classifier_result[0][7] == 1:
    print('H')
elif classifier_result[0][8] == 1:
    print('I')
elif classifier_result[0][9] == 1:
    print('J')
elif classifier_result[0][10] == 1:
    print('K')
elif classifier_result[0][11] == 1:
    print('L')
elif classifier_result[0][12] == 1:
    print('M')
elif classifier_result[0][13] == 1:
    print('N')
elif classifier_result[0][14] == 1:
    print('O')
elif classifier_result[0][15] == 1:
    print('P')
elif classifier_result[0][16] == 1:
    print('Q')
elif classifier_result[0][17] == 1:
    print('R')
elif classifier_result[0][18] == 1:
    print('S')
elif classifier_result[0][19] == 1:
    print('T')
elif classifier_result[0][20] == 1:
    print('U')
elif classifier_result[0][21] == 1:
    print('V')
elif classifier_result[0][22] == 1:
    print('W')
elif classifier_result[0][23] == 1:
    print('X')
elif classifier_result[0][24] == 1:
    print('Y')
elif classifier_result[0][25] == 1:
    print('Z')
