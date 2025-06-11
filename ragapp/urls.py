from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload_file/', views.upload_file, name='upload_file'),
    path('ask_question/', views.ask_question, name='ask_question'),
    path('reset_session/', views.reset_session, name='reset_session'),
    path('delete_file/', views.delete_file, name='delete_file'),
]