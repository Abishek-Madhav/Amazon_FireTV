from django.urls import path
from . import views

urlpatterns = [
    path('recommendations/', views.generate_recommendations, name='generate_recommendations'),
    path('content/', views.get_all_content, name='get_all_content'),
    path('track/', views.track_interaction, name='track_interaction'),
    path('train/', views.train_model, name='train_model'),
    path('health/', views.health_check, name='health_check'),
    path('profile/<str:user_id>/', views.get_user_profile, name='get_user_profile'),
    path('feedback/', views.update_user_feedback, name='update_user_feedback'),
]