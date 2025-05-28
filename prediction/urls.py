from django.urls import path
from . import views

app_name = 'prediction'

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('register/', views.register, name='register'),
    path('submit/', views.submit_sequence, name='submit'),
    path('history/', views.prediction_history, name='history'),
    path('visualize/<int:prediction_id>/', views.visualize_structure, name='visualize'),
    path('packs/', views.download_packs, name='packs'),
    path('admin-manage/', views.admin_manage, name='admin_manage'),
]