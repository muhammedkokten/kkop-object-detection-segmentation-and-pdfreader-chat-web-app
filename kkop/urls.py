from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from kkop import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('kokten.urls')),
    path('uploads/', views.upload_image, name='upload_image'),
    path('uploads/', views.upload_pdf, name='upload_pdf'),
    path('chat/', views.chat_view, name='chat_view'),
]
