from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('scripts/', views.scripts, name='scripts'),
    path('run-script/<int:pk>', views.run_script, name='run_script'),
    path('download-output-file/<int:pk>', views.download_output_file, name='download_output_file'),
]
