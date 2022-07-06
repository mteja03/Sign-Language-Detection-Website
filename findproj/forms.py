import os
from django import forms
from .models import *
from .predictor import predictor


class SignForm(forms.ModelForm):
    """Form with one Image upload field,
    acquiring the image and predicting the letter."""

    class Meta:
        # assign model to the form
        model = Sign
        fields = ['gesture']

    def __init__(self, *args, **kwargs):
        super(SignForm, self).__init__(*args, **kwargs)
        # disable labels and validation errors for input fields.
        self.fields['gesture'].label = ""
        self.fields['gesture'].error_messages = {
            'blank': 'INVALID!!11', 'null': 'NULL11!', 'required': ''}

    def sav(self):
        # get the image uploaded from the form.
        uploaded_image = super(SignForm, self).save()
        image_name = uploaded_image.gesture.name
        image_path =  f'./media/{image_name}'
        predicted_char = predictor(image_path)

        # remove image after prediction
        try:
            os.remove(f'./media/{image_name}')
        except:
            print("unable to remove image")
        return predicted_char
