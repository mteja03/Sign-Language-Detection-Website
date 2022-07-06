import os
# ignore lack of gpu for keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.preprocessing import image
import numpy as np
import cv2
from .forms import SignForm
from .predictor import predictor
from django.views.decorators import gzip
from django.shortcuts import render
from django.http import StreamingHttpResponse


# default parameters for image preprocessing
img_size_width, img_size_height = 96, 96
l_h = 0
l_s = 55
l_v = 0
u_h = 179
u_s = 255
u_v = 237
lower_bound = np.array([l_h, l_s, l_v])
upper_bound = np.array([u_h, u_s, u_v])
predicted_char = ''
weight_param = 0.1


def home(request):
    """ Returns the homepage html."""
    return render(request, 'index.html')


def video_feed(request):
    """Renders the web cam feed and receives the
    POST request updating the l_s value and re-renders."""
    if request.method == "POST":
        cv2.destroyAllWindows
        postdata = request.POST['slider1']
        l_s_update = postdata
        return render(request, 'video_feed.html', context={'l_s': l_s_update})
    else:
        return render(request, 'video_feed.html', context={'l_s': 55})


def upload_view(request):
    """Renders the file uploaded page and
    returns the prediction if image is valid."""
    if request.method == 'POST':
        form = SignForm(request.POST, request.FILES)
        if form.is_valid():
            global predicted_char
            predicted_char = form.sav()
            return render(request, 'upload.html', {'predicted_char': predicted_char})
    else:
        form = SignForm()
    return render(request, 'upload.html', {'form': form, 'predicted_char': 0})


def frame_generator(camera):
    """Gets the frame from the webcam and
    creates a generator for streaming the updated frames."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def preprocess(frame, l_s):
    """Inputs the frame and l_s value to
    create the bounding box, filters
    the cropped image into mask and
    saves it as img_sign.png."""

    img_with_box = cv2.rectangle(frame, (425, 100), (625, 300),
                                 (0, 255, 0), thickness=2, lineType=8, shift=0)
    if not img_with_box.any():
        # if frame is not obtained, load the previous image as feed.
        img_loaded = image.load_img('img_sign.png')
        img_loaded_array = np.array(img_loaded)
        return img_loaded_array
    else:
        # crop the sign image inside bounding box and apply hsv filter.
        img_box_cropped = img_with_box[102:298, 427:623]
        img_hsv = cv2.cvtColor(img_box_cropped, cv2.COLOR_BGR2HSV)
        # only updated l_s value is used.
        lower_bound = np.array([l_h, l_s, l_v])
        img_mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
        img_name = "img_sign.png"
        img_save = cv2.resize(img_mask, (img_size_width, img_size_height))
        # save image in local storage.
        cv2.imwrite(img_name, img_save)
        return img_save


@gzip.gzip_page
def video_loader(request, l_s):
    """Creates the Video Camera object and
    obtains the generator and returns the webcam feed."""

    try:
        camera = VideoCamera(l_s)
        return StreamingHttpResponse(frame_generator(camera), content_type='multipart/x-mixed-replace; boundary=frame')
    except Warning as e:
        print("video error " + e)


@gzip.gzip_page
def mask_loader(request, l_s):
    """Creates the Mask Camera object and
    obtains the generator and returns the masking feed."""
    try:
        camera = MaskCamera(l_s)
        return StreamingHttpResponse(frame_generator(camera), content_type='multipart/x-mixed-replace; boundary=frame')
    except Warning as e:
        print("Mask error " + e)


class MaskCamera(object):
    """Access the web cam and show the mask images as feed."""

    def __init__(self, l_s):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.id = l_s

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, image = self.video.read()
        frame = cv2.flip(image, 1)
        # obtain the masked frame
        mask = preprocess(frame, self.l_s)
        _, jpeg = cv2.imencode('.jpg', mask)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


class VideoCamera(object):
    """Access the web cam and show the frame images as feed
    concatinated with prediction and masking feed"""

    def __init__(self, l_s):
        self.video = cv2.VideoCapture(0)
	# fix the width and height to work on all legacy webcams.
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.l_s = l_s

    def __del__(self):
        self.video.release()

    def get_frame(self):

        frame_captured, video_frame = self.video.read()
        while not frame_captured:
	    #until frame is captured keep reading.
            frame_captured, video_frame = self.video.read()
        frame = cv2.flip(video_frame, 1)
        mask_frame = preprocess(frame, self.l_s)

	#fill in other dimensions with mask
        mask_dim = np.stack((mask_frame, mask_frame, mask_frame), axis=2)
	# mask image to be added on right top of the frame.
        frame_with_mask = cv2.addWeighted(
            frame[0:img_size_width, -(img_size_height + 1):-1, :],
	    weight_param,
            mask_dim[0:img_size_width, 0:img_size_height],
            1-weight_param, 0,
	     dtype=cv2.CV_64F)
        frame[0:img_size_width, -(img_size_height + 1):-1] = frame_with_mask

        predicted_char = predictor('img_sign.png')

	#white background on left hand top to place predicted letter on.
        white_bg = cv2.rectangle(
            frame, (0, 0), (70, 70), (255, 255, 255),
	    thickness=-1, lineType=8, shift=0)
        frame = white_bg

	#place the character on the frame.
        cv2.putText(frame, predicted_char, (10, 60),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255))
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

