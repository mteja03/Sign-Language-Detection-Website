# Sign-Language-Detection-Website
Sign language detection website can recognition gesture through webcam in real-time and by uploading images using CNN, image-classification and image-recognition.

## DataSet
This dataset has a total of 52,000 images of signs(alphabets); the dataset is separated into two sections: training and testing. The testing dataset has a capacity of 6,500 images, and the training dataset has a total of 45,000 images. In both the training and testing dataset, the images are being classified into 26 categories, each category images labelled with alphabets from A to Z. They are 26 labelled images folders in both the training and testing dataset. In each alphabet labelled folder, they are 250 images of that alphabet in the testing dataset and 1,750 images of that alphabet in the training dataset.

## Installation
Install the environment
```bash
conda create -n tf python==3.7.9
conda activate tf
```
Installing dependencies
```bash
pip install -r requirements.txt
```
Running the django server
```bash
python manage.py runserver
```
Link to access the website
[127.0.0.1:8000/](127.0.0.1:8000/)
