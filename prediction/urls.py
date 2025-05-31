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
    path('admin/packs/upload/', views.upload_model_pack, name='upload_model_pack'),
    path('admin/packs/', views.admin_packs, name='admin_packs'),
    path('admin/packs/delete/<int:pack_id>/', views.admin_delete_pack, name='admin_delete_pack'),
]