import cv2
import time
import numpy as np
import os

img_size_x, img_size_y = 64, 64

def create_folder(folder_name):
    """Checks if folders in the current path exists or not and
    creates the directory for training and test datasets"""
    
    if not os.path.exists('./mydata/training_set/' + folder_name):
        os.makedirs('./mydata/training_set/' + folder_name)
    if not os.path.exists('./mydata/test_set/' + folder_name):
        os.makedirs('./mydata/test_set/' + folder_name)
    return 


def capture_images(gesture_name):
    """Used to create the windows and adjustment to capture the image and
       place them into training and test dataset folders."""

    #create or use the provided folder.
    create_folder(str(gesture_name))
    
    # start streaming the from webcam of index 0.
    video = cv2.VideoCapture(0)
    cv2.namedWindow("camera feed")

    #initialize the counters for training and test datasets.
    img_count = 1
    train_set_img_number = 1
    test_set_img_number = 1

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 179,lambda x: None)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255,lambda x: None)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255,lambda x: None)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179,lambda x: None)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255,lambda x: None)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, lambda x: None)
    while True:

       #capture the fraame from the video.
       success, frame = video.read()
       frame = cv2.flip(frame, 1)

       # get the trackbars positions of the parameters.
       l_h = cv2.getTrackbarPos("L - H", "Trackbars")
       l_s = cv2.getTrackbarPos("L - S", "Trackbars")
       l_v = cv2.getTrackbarPos("L - V", "Trackbars")
       u_h = cv2.getTrackbarPos("U - H", "Trackbars")
       u_s = cv2.getTrackbarPos("U - S", "Trackbars")
       u_v = cv2.getTrackbarPos("U - V", "Trackbars")

       # create a box at given coordinates  of green color and thickness of 2.
       img_with_box = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)
       img_box_cropped = img_with_box[102:298, 427:623]

       #create array of bound parameters.
       lower_bound = np.array([l_h, l_s, l_v])
       upper_bound = np.array([u_h, u_s, u_v])

       # create the hsv filter and place it within the bound arrays.
       img_hsv = cv2.cvtColor(img_box_cropped , cv2.COLOR_BGR2HSV)
       img_mask = cv2.inRange(img_hsv , lower_bound, upper_bound)
       
       # create the and operation of box cropped with the mask.
       img_bit_and = cv2.bitwise_and(img_box_cropped , img_box_cropped , mask=img_mask)


       # place the image number on the camera feed.
       cv2.putText(frame, str(img_count - 1), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))

       # show the images of frame,mask and AND operation images.
       cv2.imshow("camera feed", frame)
       cv2.imshow("mask feed", img_mask)
       cv2.imshow("final output", img_bit_and)

       if cv2.waitKey(1) == ord('c'):
       # to be executed when user hits the 'c' key on the keyboard.

           if img_count <= 350:
               img_name = "./mydata/training_set/" + str(gesture_name) + "/{}.png".format(train_set_img_number)
               save_img = cv2.resize(img_mask, (img_size_x, img_size_y))
               # save the image after resizing in the training dataset folder.
               cv2.imwrite(img_name, save_img)
               print("{} written!".format(img_name))
               train_set_img_number += 1


           elif img_count > 350 and img_count <= 400:
               img_name = "./mydata/test_set/" + str(gesture_name) + "/{}.png".format(test_set_img_number)
               save_img = cv2.resize(img_mask, (img_size_x, img_size_y))
               # save the image after resizing in the test dataset folder.
               cv2.imwrite(img_name, save_img)
               print("{} written!".format(img_name))
               test_set_img_number += 1
           else:
               # if the count exceeds 400 end the gesture.
               return

           img_count += 1


       elif cv2.waitKey(1) == 27:
           # close the windows when user hits ESC on their keyboard.
           video.release()
           cv2.destroyAllWindows()
           return


    


# call the functions as per user input.
gesture_name = input("Enter gesture name: ")
capture_images(gesture_name)
