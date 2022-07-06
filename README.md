# Sign-Language-Detection-Website
Sign language detection website can recognition gesture through webcam in real-time and by uploading images using CNN, image-classification and image-recognition.

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
```sh
127.0.0.1:8000/
```

## DataSet
This dataset has a total of 52,000 images of signs(alphabets); the dataset is separated into two sections: training and testing. The testing dataset has a capacity of 6,500 images, and the training dataset has a total of 45,000 images. In both the training and testing dataset, the images are being classified into 26 categories, each category images labelled with alphabets from A to Z. They are 26 labelled images folders in both the training and testing dataset. In each alphabet labelled folder, they are 250 images of that alphabet in the testing dataset and 1,750 images of that alphabet in the training dataset.

## Images
Home Page
<img width="1440" alt="home" src="https://user-images.githubusercontent.com/62012634/177539075-6bf13a87-09ce-47ce-9590-01a72bcbcdbe.png">

Upload Page
<img width="1440" alt="upload" src="https://user-images.githubusercontent.com/62012634/177539562-32967063-7d24-4c76-919b-2967e1bde438.png">

Detection Page
<img width="1440" alt="webcam" src="https://user-images.githubusercontent.com/62012634/177539873-a9ec2d80-78b2-4f25-bced-7904d9161fd4.png">

Results
<img width="1440" alt="P" src="https://user-images.githubusercontent.com/62012634/177540004-8267d910-ce6a-42da-a48e-12943c6e5db0.png">

<img width="600" alt="2" src="https://user-images.githubusercontent.com/62012634/177539928-7f20ae40-e440-47d9-8f26-66b59399b7d4.png">

