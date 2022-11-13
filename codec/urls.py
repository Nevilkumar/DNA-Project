from django.urls import path
from codec import views as codec_views

urlpatterns = [
    path('', codec_views.home, name='home'),
    path('download/', codec_views.download, name='download'),
]
