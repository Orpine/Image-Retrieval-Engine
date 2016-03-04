from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
import cv2
import json

# Create your views here.
from search.forms import ImageForm


def search(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        response = {}
        if form.is_valid():
            image = form.cleaned_data['image']
            save_name = 'upload/' + image.name
            uploaded_image = open(save_name, 'wb')
            for chunk in image.chunks():
                uploaded_image.write(chunk)
            uploaded_image.close()

            image_data = cv2.imread(save_name)

            # compute

            result = [save_name] * 50

            # print(np.asarray(image))
            # cvImage = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            # print(cvImage)

            response['valid'] = True
            response['result'] = result
            print('valid')
        else:
            print('invalid')
            response['valid'] = True
        return HttpResponse(json.dumps(response), content_type='application/json')
    else:
        form = ImageForm()

    return render(request, 'search/index.html', {'form': form})