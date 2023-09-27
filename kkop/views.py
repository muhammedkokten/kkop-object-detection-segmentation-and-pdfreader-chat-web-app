from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.http.response import HttpResponse, JsonResponse
from kkop.models import UploadPic, DetectionResult
import os
from kokten import settings
from objection import run_detection
from segmentation import run_segmentation
from chat import chat_ai

# Create your views here.



def upload_image(request):
    if request.method == 'POST':
        image = request.FILES['image']
        title = request.POST['title']
        value = request.POST.get('value')

        if image:
            UploadPic.objects.create(title=title, image=image, is_active=True, is_home=False)
            if value == '0':
                run_detection()
            elif value == '1':
                run_segmentation()

            return redirect('sonuc')

    return render(request, 'index.html')



def index(request):
    images = UploadPic.objects.all()
    return render(request, "index.html", {'images': images})


def detection(request):
    return render(request, "detection.html")


def segmentation(request):
    return render(request, "segmentation.html")


def chat(request):
    return render(request, "chat.html")

def upload_chat(request):
    return render(request, "upload_chat.html")


def upload_pdf(request):
    if request.method == 'POST' and request.FILES['pdf_file']:
        pdf_file = request.FILES['pdf_file']
        fs = FileSystemStorage()
        fs.save('pdf/' + pdf_file.name, pdf_file)
        return redirect('chat_page')

    return render(request, 'upload_chat.html')


def sonuc(request):
    son_resim = DetectionResult.objects.order_by('-id').first()
    history_path = DetectionResult.objects.order_by('-id').all()
    return render(request, 'sonuc.html', {'son_resim': son_resim, 'history_path': history_path})


def chat_view(request):
    if request.method == "POST":
        user_message = request.POST.get("user_message")
        response = chat_ai(user_message)
        return HttpResponse(response)

    return render(request, "chat.html")