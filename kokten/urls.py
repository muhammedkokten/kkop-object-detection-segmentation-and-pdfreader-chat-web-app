from django.urls import path
from kkop import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path("", views.index),
    path("index", views.index, name='index'),
    path("detection", views.detection, name='detection_page'),
    path("segmentation", views.segmentation, name='segmentation_page'),
    path("segmentation", views.segmentation),
    path("chat", views.chat, name='chat_page'),
    path("upload_chat", views.upload_chat, name='upload_chat'),
    path('pdf/', views.upload_pdf, name='upload_pdf'),
    path("upload_image", views.upload_image, name='upload_image'),
    path('sonuc/', views.sonuc, name='sonuc'),
    path('chat/', views.chat_view, name='chat_view'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
