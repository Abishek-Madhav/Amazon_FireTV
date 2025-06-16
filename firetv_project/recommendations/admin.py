from django.contrib import admin
from .models import Content, UserProfile, ViewingSession, Recommendation

@admin.register(Content)
class ContentAdmin(admin.ModelAdmin):
    list_display = ['title', 'genre', 'rating', 'duration', 'release_year', 'popularity_score']
    list_filter = ['genre', 'release_year', 'group_friendly']
    search_fields = ['title', 'description']
    ordering = ['-popularity_score', '-rating']

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user_id', 'age_group', 'social_activity_level', 'avg_rating_given', 'created_at']
    list_filter = ['age_group', 'social_activity_level']
    search_fields = ['user_id']

@admin.register(ViewingSession)
class ViewingSessionAdmin(admin.ModelAdmin):
    list_display = ['user_profile', 'content', 'mood', 'hour_of_day', 'user_rating', 'created_at']
    list_filter = ['mood', 'is_weekend', 'is_group_watch']
    search_fields = ['user_profile__user_id', 'content__title']

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ['user_profile', 'content', 'confidence_score', 'recommendation_type', 'was_clicked', 'created_at']
    list_filter = ['recommendation_type', 'was_clicked', 'was_watched']
    search_fields = ['user_profile__user_id', 'content__title']