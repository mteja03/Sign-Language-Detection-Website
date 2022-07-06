from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
        path('',views.home,name='home'),
        path('upload',views.upload_view,name='upload'),
	path("video/<int:l_s>",views.video_loader,name="video"),
	path("mask/<int:l_s>",views.mask_loader,name="mask"),
	path("webcam/",views.video_feed,name="webcam"),
]

# add static media folder to be accessible.
if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)