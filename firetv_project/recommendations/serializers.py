from rest_framework import serializers
from .models import Content, UserProfile, ViewingSession, Recommendation
import json

class ContentSerializer(serializers.ModelSerializer):
    mood_tags = serializers.SerializerMethodField()
    time_slots = serializers.SerializerMethodField()
    age_groups = serializers.SerializerMethodField()
    
    class Meta:
        model = Content
        fields = '__all__'
    
    def get_mood_tags(self, obj):
        return obj.mood_list
    
    def get_time_slots(self, obj):
        return obj.time_slots_list
    
    def get_age_groups(self, obj):
        return obj.age_groups_list

class UserProfileSerializer(serializers.ModelSerializer):
    preferred_genres = serializers.SerializerMethodField()
    
    class Meta:
        model = UserProfile
        fields = '__all__'
    
    def get_preferred_genres(self, obj):
        return json.loads(obj.preferred_genres)

class RecommendationRequestSerializer(serializers.Serializer):
    user_id = serializers.CharField(max_length=100)
    age_group = serializers.ChoiceField(choices=UserProfile.AGE_GROUP_CHOICES)
    mood = serializers.ChoiceField(choices=ViewingSession.MOOD_CHOICES)
    hour_of_day = serializers.IntegerField(min_value=0, max_value=23)
    day_of_week = serializers.IntegerField(min_value=1, max_value=7)
    is_weekend = serializers.BooleanField()
    avg_session_duration = serializers.IntegerField(min_value=5, max_value=300)
    social_activity_level = serializers.ChoiceField(choices=UserProfile.SOCIAL_ACTIVITY_CHOICES)
    is_group_watch = serializers.BooleanField()
    group_size = serializers.IntegerField(min_value=1, max_value=10)
    recent_genre_preference = serializers.ChoiceField(choices=Content.GENRE_CHOICES)
    avg_rating_given = serializers.FloatField(min_value=1.0, max_value=10.0)

class RecommendationResponseSerializer(serializers.ModelSerializer):
    content = ContentSerializer()
    reasoning_list = serializers.SerializerMethodField()
    
    class Meta:
        model = Recommendation
        fields = ['content', 'confidence_score', 'reasoning_list', 'recommendation_type']
    
    def get_reasoning_list(self, obj):
        try:
            return json.loads(obj.reasoning)
        except:
            return []