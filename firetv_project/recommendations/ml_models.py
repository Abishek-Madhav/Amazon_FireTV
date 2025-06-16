import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from django.conf import settings
from .models import Content, UserProfile, ViewingSession
import json

class FireTVRecommendationEngine:
    def __init__(self):
        self.content_encoder = LabelEncoder()
        self.mood_encoder = LabelEncoder()
        self.age_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # Content-based similarity matrix
        self.content_similarity_matrix = None
        self.content_features_df = None
        
    def prepare_features(self, user_data, content_data=None):
        """Prepare features for ML model"""
        features = []
        
        # User context features
        features.extend([
            user_data.get('hour_of_day', 20),
            user_data.get('day_of_week', 1),
            user_data.get('is_weekend', 0),
            user_data.get('avg_session_duration', 60),
            user_data.get('group_size', 1),
            user_data.get('avg_rating_given', 7.0),
            1 if user_data.get('is_group_watch', False) else 0,
        ])
        
        # Mood encoding
        mood_map = {
            'happy': 1, 'excited': 2, 'relaxed': 3, 'romantic': 4,
            'adventurous': 5, 'nostalgic': 6, 'stressed': 7, 'sad': 8, 'energetic': 9
        }
        features.append(mood_map.get(user_data.get('mood', 'happy'), 1))
        
        # Age group encoding
        age_map = {
            'teen': 1, 'young_adult': 2, 'adult': 3, 'middle_age': 4, 'senior': 5
        }
        features.append(age_map.get(user_data.get('age_group', 'adult'), 3))
        
        # Social activity encoding
        social_map = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
        features.append(social_map.get(user_data.get('social_activity_level', 'medium'), 2))
        
        # Genre preference encoding
        genre_map = {
            'action': 1, 'comedy': 2, 'drama': 3, 'sci-fi': 4, 'romance': 5,
            'thriller': 6, 'horror': 7, 'documentary': 8, 'animation': 9, 'fantasy': 10
        }
        features.append(genre_map.get(user_data.get('recent_genre_preference', 'action'), 1))
        
        if content_data:
            # Content features
            features.extend([
                content_data.rating,
                content_data.duration,
                content_data.popularity_score,
                1 if content_data.group_friendly else 0,
            ])
            
            # Genre matching
            features.append(genre_map.get(content_data.genre, 1))
        
        return np.array(features).reshape(1, -1)
    
    def calculate_mood_compatibility(self, user_mood, content_moods):
        """Calculate mood compatibility score"""
        mood_compatibility = {
            'happy': ['comedy', 'animation', 'romance'],
            'excited': ['action', 'adventure', 'sci-fi'],
            'relaxed': ['drama', 'documentary', 'romance'],
            'romantic': ['romance', 'drama'],
            'adventurous': ['action', 'fantasy', 'sci-fi'],
            'nostalgic': ['drama', 'documentary'],
            'stressed': ['comedy', 'animation'],
            'sad': ['drama', 'documentary'],
            'energetic': ['action', 'animation', 'adventure']
        }
        
        try:
            mood_list = json.loads(content_moods) if isinstance(content_moods, str) else content_moods
            compatible_moods = mood_compatibility.get(user_mood, [])
            
            # Check if any content moods match user's mood preferences
            compatibility_score = 0
            for mood in mood_list:
                if mood in compatible_moods:
                    compatibility_score += 30
                elif mood == user_mood:
                    compatibility_score += 50
            
            return min(compatibility_score, 100)
        except:
            return 0
    
    def calculate_time_compatibility(self, user_hour, content_time_slots):
        """Calculate time compatibility score"""
        try:
            time_slots = json.loads(content_time_slots) if isinstance(content_time_slots, str) else content_time_slots
            
            if user_hour in time_slots:
                return 30
            
            # Check nearby time slots (±2 hours)
            for slot in time_slots:
                if abs(user_hour - slot) <= 2:
                    return 15
            
            return 0
        except:
            return 0
    
    def calculate_age_compatibility(self, user_age_group, content_age_groups):
        """Calculate age group compatibility score"""
        try:
            age_groups = json.loads(content_age_groups) if isinstance(content_age_groups, str) else content_age_groups
            
            if user_age_group in age_groups:
                return 25
            
            # Age group similarity scoring
            age_similarity = {
                'teen': ['young_adult'],
                'young_adult': ['teen', 'adult'],
                'adult': ['young_adult', 'middle_age'],
                'middle_age': ['adult', 'senior'],
                'senior': ['middle_age']
            }
            
            similar_ages = age_similarity.get(user_age_group, [])
            for age in age_groups:
                if age in similar_ages:
                    return 15
            
            return 0
        except:
            return 0
    
    def generate_recommendations(self, user_data, limit=5):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Get all content
        all_content = Content.objects.all()
        
        for content in all_content:
            score = 0
            reasons = []
            
            # Mood compatibility
            mood_score = self.calculate_mood_compatibility(
                user_data.get('mood', 'happy'), 
                content.mood_tags
            )
            if mood_score > 0:
                score += mood_score
                reasons.append(f"Perfect for {user_data.get('mood', 'happy')} mood")
            
            # Time compatibility
            time_score = self.calculate_time_compatibility(
                user_data.get('hour_of_day', 20),
                content.time_slots
            )
            if time_score > 0:
                score += time_score
                reasons.append("Ideal for current time")
            
            # Age compatibility
            age_score = self.calculate_age_compatibility(
                user_data.get('age_group', 'adult'),
                content.age_groups
            )
            if age_score > 0:
                score += age_score
                reasons.append(f"Great for {user_data.get('age_group', 'adult')} viewers")
            
            # Genre preference matching
            if content.genre == user_data.get('recent_genre_preference'):
                score += 35
                reasons.append(f"Matches your {content.genre} preference")
            
            # Group watching compatibility
            if user_data.get('is_group_watch', False) and content.group_friendly:
                score += 15
                reasons.append("Perfect for group watching")
            
            # Session duration matching
            duration_diff = abs(content.duration - user_data.get('avg_session_duration', 60))
            if duration_diff <= 30:
                score += 10
                reasons.append("Matches your viewing time")
            
            # Weekend boost for longer content
            if user_data.get('is_weekend', False) and content.duration > 90:
                score += 10
                reasons.append("Weekend binge-worthy")
            
            # High rating boost
            if content.rating >= 8.5:
                score += 15
                reasons.append("Highly rated content")
            
            # Social activity influence
            if (user_data.get('social_activity_level') == 'high' and 
                content.rating > 8.5):
                score += 10
                reasons.append("Highly shareable content")
            
            # Rating preference alignment
            rating_diff = abs(content.rating - user_data.get('avg_rating_given', 7.0))
            if rating_diff <= 1:
                score += 15
                reasons.append("Matches your rating preference")
            
            # Add some diversity factor
            score += np.random.random() * 10
            
            # Cap the score at 100
            final_score = min(score, 100)
            
            if final_score > 20:  # Only include if score is reasonable
                recommendations.append({
                    'content': content,
                    'score': final_score,
                    'reasons': reasons[:3]  # Top 3 reasons
                })
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:limit]
    
    def train_model(self):
        """Train the ML model with available data"""
        # This would be called periodically to retrain with new data
        sessions = ViewingSession.objects.filter(user_rating__isnull=False)
        
        if sessions.count() < 10:
            # Not enough data to train, use rule-based system
            self.is_trained = False
            return
        
        # Prepare training data
        X = []
        y = []
        
        for session in sessions:
            user_data = {
                'hour_of_day': session.hour_of_day,
                'day_of_week': session.day_of_week,
                'is_weekend': session.is_weekend,
                'avg_session_duration': session.user_profile.avg_session_duration,
                'group_size': session.group_size,
                'avg_rating_given': session.user_profile.avg_rating_given,
                'is_group_watch': session.is_group_watch,
                'mood': session.mood,
                'age_group': session.user_profile.age_group,
                'social_activity_level': session.user_profile.social_activity_level,
                'recent_genre_preference': session.content.genre if session.content else 'action'
            }
            
            if session.content:
                features = self.prepare_features(user_data, session.content)
                X.append(features.flatten())
                y.append(session.user_rating)
        
        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
        
        return self.is_trained

# Initialize the recommendation engine
recommendation_engine = FireTVRecommendationEngine()
