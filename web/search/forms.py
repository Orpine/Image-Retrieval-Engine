from django import forms
import PIL


class ImageForm(forms.Form):
    image = forms.ImageField()