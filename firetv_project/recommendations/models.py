from django.db import models
from django.contrib.auth.models import User
import json

class Content(models.Model):
    GENRE_CHOICES = [
        ('action', 'Action'),
        ('comedy', 'Comedy'),
        ('drama', 'Drama'),
        ('sci-fi', 'Sci-Fi'),
        ('romance', 'Romance'),
        ('thriller', 'Thriller'),
        ('horror', 'Horror'),
        ('documentary', 'Documentary'),
        ('animation', 'Animation'),
        ('fantasy', 'Fantasy'),
    ]
    
    title = models.CharField(max_length=200)
    genre = models.CharField(max_length=20, choices=GENRE_CHOICES)
    rating = models.FloatField()
    duration = models.IntegerField()  # in minutes
    release_year = models.IntegerField(default=2023)
    description = models.TextField(blank=True)
    
    # Store as JSON strings for flexibility
    mood_tags = models.TextField(default='[]')  # JSON array of moods
    time_slots = models.TextField(default='[]')  # JSON array of time slots
    age_groups = models.TextField(default='[]')  # JSON array of age groups
    
    group_friendly = models.BooleanField(default=True)
    popularity_score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-popularity_score', '-rating']
    
    def __str__(self):
        return self.title
    
    @property
    def mood_list(self):
        return json.loads(self.mood_tags)
    
    @property
    def time_slots_list(self):
        return json.loads(self.time_slots)
    
    @property
    def age_groups_list(self):
        return json.loads(self.age_groups)

class UserProfile(models.Model):
    AGE_GROUP_CHOICES = [
        ('teen', 'Teen (13-17)'),
        ('young_adult', 'Young Adult (18-25)'),
        ('adult', 'Adult (26-35)'),
        ('middle_age', 'Middle Age (36-50)'),
        ('senior', 'Senior (51+)'),
    ]
    
    SOCIAL_ACTIVITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('very_high', 'Very High'),
    ]
    
    user_id = models.CharField(max_length=100, unique=True)
    age_group = models.CharField(max_length=20, choices=AGE_GROUP_CHOICES)
    preferred_genres = models.TextField(default='[]')  # JSON array
    avg_session_duration = models.IntegerField(default=60)
    social_activity_level = models.CharField(max_length=20, choices=SOCIAL_ACTIVITY_CHOICES, default='medium')
    avg_rating_given = models.FloatField(default=7.0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Profile: {self.user_id}"

class ViewingSession(models.Model):
    MOOD_CHOICES = [
        ('happy', 'Happy'),
        ('excited', 'Excited'),
        ('relaxed', 'Relaxed'),
        ('romantic', 'Romantic'),
        ('adventurous', 'Adventurous'),
        ('nostalgic', 'Nostalgic'),
        ('stressed', 'Stressed'),
        ('sad', 'Sad'),
        ('energetic', 'Energetic'),
    ]
    
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    content = models.ForeignKey(Content, on_delete=models.CASCADE, null=True, blank=True)
    
    # Session context
    mood = models.CharField(max_length=20, choices=MOOD_CHOICES)
    hour_of_day = models.IntegerField()
    day_of_week = models.IntegerField()
    is_weekend = models.BooleanField()
    is_group_watch = models.BooleanField(default=False)
    group_size = models.IntegerField(default=1)
    
    # Interaction data
    watch_duration = models.IntegerField(default=0)  # minutes watched
    completion_rate = models.FloatField(default=0.0)  # percentage completed
    user_rating = models.FloatField(null=True, blank=True)
    liked = models.BooleanField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class Recommendation(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    content = models.ForeignKey(Content, on_delete=models.CASCADE)
    
    # ML model outputs
    confidence_score = models.FloatField()
    reasoning = models.TextField()  # JSON string of reasons
    recommendation_type = models.CharField(max_length=50, default='ai_hybrid')
    
    # Context when recommendation was made
    context_mood = models.CharField(max_length=20)
    context_time = models.IntegerField()
    context_group_size = models.IntegerField(default=1)
    
    # Interaction tracking
    was_clicked = models.BooleanField(default=False)
    was_watched = models.BooleanField(default=False)
    click_timestamp = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-confidence_score', '-created_at']
        unique_together = ['user_profile', 'content', 'created_at']
