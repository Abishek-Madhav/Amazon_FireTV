import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import random
from datetime import datetime, timedelta
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class FireTVRecommendationEngine:
    """Enhanced Fire TV Recommendation Engine based on user inputs"""

    def __init__(self):
        self.moods = ['happy', 'sad', 'excited', 'relaxed', 'stressed', 'romantic', 'adventurous', 'energetic', 'nostalgic']
        self.genres = ['action', 'comedy', 'drama', 'horror', 'romance', 'sci-fi', 'documentary', 'thriller', 'animation', 'musical', 'mystery', 'fantasy']
        self.platforms = ['Netflix', 'Prime Video', 'Disney+', 'Hulu', 'HBO Max', 'Apple TV+', 'Paramount+', 'YouTube', 'Peacock']
        self.content_types = ['movie', 'series', 'documentary', 'special', 'short']
        self.model = None
        self.user_tower = None
        self.item_tower = None
        self.label_encoders = {}
        self.content_database = None
        self.user_profiles = {}

    def initialize_content_database(self, n_content=5000):
        """Initialize content database with metadata"""
        content = []
        for i in range(n_content):
            item = {
                'content_id': f'content_{i}',
                'title': f'Title_{i}',
                'genre': np.random.choice(self.genres),
                'secondary_genre': np.random.choice(self.genres),
                'platform': np.random.choice(self.platforms),
                'content_type': np.random.choice(self.content_types),
                'duration': np.random.normal(90, 30) if np.random.choice(self.content_types) == 'movie' else np.random.normal(45, 15),
                'release_year': np.random.randint(2010, 2024),
                'rating': np.random.uniform(3.0, 9.5),
                'mood_tags': np.random.choice(self.moods, size=np.random.randint(1, 4), replace=False).tolist(),
                'popularity_score': np.random.exponential(2),
                'is_trending': np.random.choice([True, False], p=[0.1, 0.9]),
                'director': f'Director_{i % 100}',
                'cast': [f'Actor_{j}' for j in range(i % 5 + 1)],
                'language': np.random.choice(['English', 'Spanish', 'French', 'German', 'Japanese', 'Korean']),
                'description': f'Description for {i}'
            }
            content.append(item)

        self.content_database = pd.DataFrame(content)
        return self.content_database

    def create_user_profile(self, user_id, age_group, favorite_genres, avg_session_length=60):
        """Create or update user profile"""
        profile = {
            'user_id': user_id,
            'age_group': age_group,  # '18-25', '26-35', '36-45', '46-55', '55+'
            'favorite_genres': favorite_genres,
            'avg_session_length': avg_session_length,
            'watch_history': [],
            'ratings_history': {},
            'mood_preferences': defaultdict(list),
            'time_preferences': defaultdict(list),
            'social_activity_level': 'medium',
            'preferred_languages': ['English'],
            'created_at': datetime.now()
        }
        self.user_profiles[user_id] = profile
        return profile

    def update_watch_history(self, user_id, content_id, rating=None, completion_rate=1.0, mood=None, timestamp=None):
        """Update user's watch history"""
        if user_id not in self.user_profiles:
            return False

        watch_entry = {
            'content_id': content_id,
            'timestamp': timestamp or datetime.now(),
            'completion_rate': completion_rate,
            'mood': mood,
            'rating': rating
        }

        self.user_profiles[user_id]['watch_history'].append(watch_entry)

        if rating:
            self.user_profiles[user_id]['ratings_history'][content_id] = rating

        if mood:
            self.user_profiles[user_id]['mood_preferences'][mood].append(content_id)

        return True

class TwoTowerRecommendationModel:
    """Enhanced Two-Tower Architecture for Fire TV Recommendations"""

    def __init__(self, embedding_dim=64, hidden_dims=[128, 64]):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.model = None
        self.user_tower = None
        self.item_tower = None
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def create_user_tower(self, user_vocab_sizes):
        """Create user tower for encoding user features"""
        # User inputs
        user_id = layers.Input(shape=(), name='user_id', dtype='int32')
        age_group = layers.Input(shape=(), name='age_group', dtype='int32')
        mood = layers.Input(shape=(), name='mood', dtype='int32')
        hour_of_day = layers.Input(shape=(), name='hour_of_day', dtype='float32')
        day_of_week = layers.Input(shape=(), name='day_of_week', dtype='float32')
        is_weekend = layers.Input(shape=(), name='is_weekend', dtype='float32')
        avg_session_duration = layers.Input(shape=(), name='avg_session_duration', dtype='float32')
        social_activity_level = layers.Input(shape=(), name='social_activity_level', dtype='int32')
        is_group_watch = layers.Input(shape=(), name='is_group_watch', dtype='float32')
        group_size = layers.Input(shape=(), name='group_size', dtype='float32')
        recent_genre_preference = layers.Input(shape=(), name='recent_genre_preference', dtype='int32')
        avg_rating_given = layers.Input(shape=(), name='avg_rating_given', dtype='float32')

        # Embeddings
        user_emb = layers.Embedding(user_vocab_sizes['user_id'], self.embedding_dim)(user_id)
        user_emb = layers.Flatten()(user_emb)

        age_emb = layers.Embedding(user_vocab_sizes['age_group'], self.embedding_dim//2)(age_group)
        age_emb = layers.Flatten()(age_emb)

        mood_emb = layers.Embedding(user_vocab_sizes['mood'], self.embedding_dim//2)(mood)
        mood_emb = layers.Flatten()(mood_emb)

        genre_pref_emb = layers.Embedding(user_vocab_sizes['recent_genre_preference'], self.embedding_dim//2)(recent_genre_preference)
        genre_pref_emb = layers.Flatten()(genre_pref_emb)

        social_emb = layers.Embedding(user_vocab_sizes['social_activity_level'], self.embedding_dim//4)(social_activity_level)
        social_emb = layers.Flatten()(social_emb)

        # Normalize numerical features
        hour_norm = layers.Lambda(lambda x: x / 24.0)(hour_of_day)
        day_norm = layers.Lambda(lambda x: x / 7.0)(day_of_week)
        duration_norm = layers.Lambda(lambda x: x / 120.0)(avg_session_duration)
        group_norm = layers.Lambda(lambda x: x / 6.0)(group_size)
        rating_norm = layers.Lambda(lambda x: x / 10.0)(avg_rating_given)

        # Reshape for concatenation
        hour_norm = layers.Reshape((1,))(hour_norm)
        day_norm = layers.Reshape((1,))(day_norm)
        duration_norm = layers.Reshape((1,))(duration_norm)
        is_weekend_reshaped = layers.Reshape((1,))(is_weekend)
        is_group_watch_reshaped = layers.Reshape((1,))(is_group_watch)
        group_norm = layers.Reshape((1,))(group_norm)
        rating_norm = layers.Reshape((1,))(rating_norm)

        # Concatenate all features
        user_features = layers.Concatenate()([
            user_emb, age_emb, mood_emb, genre_pref_emb, social_emb,
            hour_norm, day_norm, duration_norm, is_weekend_reshaped,
            is_group_watch_reshaped, group_norm, rating_norm
        ])

        # Dense layers with attention mechanism
        x = user_features
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)

        user_embedding = layers.Dense(self.embedding_dim, activation='relu', name='user_embedding')(x)
        user_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(user_embedding)

        user_inputs = [user_id, age_group, mood, hour_of_day, day_of_week,
                      is_weekend, avg_session_duration, social_activity_level,
                      is_group_watch, group_size, recent_genre_preference, avg_rating_given]

        return keras.Model(inputs=user_inputs, outputs=user_embedding, name='user_tower')

    def create_item_tower(self, item_vocab_sizes):
        """Create item tower for encoding content features"""
        # Item inputs
        content_id = layers.Input(shape=(), name='content_id', dtype='int32')
        genre = layers.Input(shape=(), name='genre', dtype='int32')
        secondary_genre = layers.Input(shape=(), name='secondary_genre', dtype='int32')
        platform = layers.Input(shape=(), name='platform', dtype='int32')
        content_type = layers.Input(shape=(), name='content_type', dtype='int32')
        duration = layers.Input(shape=(), name='duration', dtype='float32')
        release_year = layers.Input(shape=(), name='release_year', dtype='float32')
        rating = layers.Input(shape=(), name='rating', dtype='float32')
        popularity_score = layers.Input(shape=(), name='popularity_score', dtype='float32')
        is_trending = layers.Input(shape=(), name='is_trending', dtype='float32')

        # Embeddings
        content_emb = layers.Embedding(item_vocab_sizes['content_id'], self.embedding_dim)(content_id)
        content_emb = layers.Flatten()(content_emb)

        genre_emb = layers.Embedding(item_vocab_sizes['genre'], self.embedding_dim//2)(genre)
        genre_emb = layers.Flatten()(genre_emb)

        secondary_genre_emb = layers.Embedding(item_vocab_sizes['secondary_genre'], self.embedding_dim//2)(secondary_genre)
        secondary_genre_emb = layers.Flatten()(secondary_genre_emb)

        platform_emb = layers.Embedding(item_vocab_sizes['platform'], self.embedding_dim//4)(platform)
        platform_emb = layers.Flatten()(platform_emb)

        type_emb = layers.Embedding(item_vocab_sizes['content_type'], self.embedding_dim//4)(content_type)
        type_emb = layers.Flatten()(type_emb)

        # Normalize numerical features
        duration_norm = layers.Lambda(lambda x: x / 180.0)(duration)
        year_norm = layers.Lambda(lambda x: (x - 2010) / 14.0)(release_year)
        rating_norm = layers.Lambda(lambda x: x / 10.0)(rating)
        popularity_norm = layers.Lambda(lambda x: tf.keras.utils.normalize(tf.expand_dims(x, -1), axis=-1)[:, 0])(popularity_score)

        # Reshape for concatenation
        duration_norm = layers.Reshape((1,))(duration_norm)
        year_norm = layers.Reshape((1,))(year_norm)
        rating_norm = layers.Reshape((1,))(rating_norm)
        popularity_norm = layers.Reshape((1,))(popularity_norm)
        is_trending_reshaped = layers.Reshape((1,))(is_trending)

        # Concatenate all features
        item_features = layers.Concatenate()([
            content_emb, genre_emb, secondary_genre_emb, platform_emb, type_emb,
            duration_norm, year_norm, rating_norm, popularity_norm, is_trending_reshaped
        ])

        # Dense layers
        x = item_features
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)

        item_embedding = layers.Dense(self.embedding_dim, activation='relu', name='item_embedding')(x)
        item_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(item_embedding)

        item_inputs = [content_id, genre, secondary_genre, platform, content_type,
                      duration, release_year, rating, popularity_score, is_trending]

        return keras.Model(inputs=item_inputs, outputs=item_embedding, name='item_tower')

class PersonalizedRecommendationSystem:
    """Main recommendation system that processes user inputs"""

    def __init__(self):
        self.fire_tv_engine = FireTVRecommendationEngine()
        self.two_tower_model = TwoTowerRecommendationModel(embedding_dim=64, hidden_dims=[128, 64])
        self.social_engine = SocialWatchingEngine()
        self.is_trained = False

    def initialize_system(self, n_content=2000):
        """Initialize the recommendation system"""
        print("Fire TV Recommendation System by team Innos")
        self.fire_tv_engine.initialize_content_database(n_content)
        print(f" Initialized with {n_content} content items")

    def get_user_recommendations(self, user_input):

        print(f"\n Getting recommendations for user: {user_input['user_id']}")

        # Extract user information
        user_id = user_input['user_id']
        mood = user_input['current_mood']
        hour = user_input['current_time']
        day = user_input['current_day']
        is_weekend = day >= 5
        watch_history = user_input.get('watch_history', [])
        profile = user_input.get('user_profile', {})
        group_info = user_input.get('group_activity', {})

        # Create or update user profile
        if user_id not in self.fire_tv_engine.user_profiles:
            self.fire_tv_engine.create_user_profile(
                user_id=user_id,
                age_group=profile.get('age_group', '26-35'),
                favorite_genres=profile.get('favorite_genres', ['comedy']),
                avg_session_length=profile.get('avg_session_length', 60)
            )

        # Update watch history
        for item in watch_history:
            self.fire_tv_engine.update_watch_history(
                user_id=user_id,
                content_id=item['content_id'],
                rating=item.get('rating'),
                completion_rate=item.get('completion_rate', 1.0),
                mood=mood
            )

        # Get content recommendations using various strategies
        recommendations = self._generate_recommendations(user_input)

        # Format response
        response = {
            'user_id': user_id,
            'recommendations': recommendations,
            'context': {
                'mood': mood,
                'time_context': self._get_time_context(hour, is_weekend),
                'is_group_session': group_info.get('is_group_watch', False)
            },
            'explanation': self._generate_explanation(user_input, recommendations)
        }

        return response

    def _generate_recommendations(self, user_input):
        """Generate recommendations using multiple strategies"""

        # Strategy 1: Mood-based filtering
        mood_recs = self._get_mood_based_recommendations(user_input)

        # Strategy 2: Time-based filtering
        time_recs = self._get_time_based_recommendations(user_input)

        # Strategy 3: History-based collaborative filtering
        history_recs = self._get_history_based_recommendations(user_input)

        # Strategy 4: Social recommendations for group watching
        social_recs = self._get_social_recommendations(user_input)

        # Combine and rank recommendations
        all_recommendations = self._combine_recommendations([
            mood_recs, time_recs, history_recs, social_recs
        ])

        return all_recommendations[:10]  # Top 10 recommendations

    def _get_mood_based_recommendations(self, user_input):
        """Get recommendations based on current mood"""
        mood = user_input['current_mood']

        # Mood to genre mapping
        mood_genre_map = {
            'happy': ['comedy', 'musical', 'animation'],
            'sad': ['drama', 'documentary'],
            'excited': ['action', 'thriller', 'adventure'],
            'relaxed': ['comedy', 'romance', 'documentary'],
            'stressed': ['comedy', 'animation', 'romance'],
            'romantic': ['romance', 'drama', 'musical'],
            'adventurous': ['action', 'sci-fi', 'adventure', 'fantasy'],
            'energetic': ['action', 'thriller', 'musical'],
            'nostalgic': ['drama', 'musical', 'classic']
        }

        preferred_genres = mood_genre_map.get(mood, ['comedy'])

        # Filter content by mood-appropriate genres
        mood_content = self.fire_tv_engine.content_database[
            self.fire_tv_engine.content_database['genre'].isin(preferred_genres) |
            self.fire_tv_engine.content_database['secondary_genre'].isin(preferred_genres)
        ].copy()

        # Add mood compatibility score
        mood_content['mood_score'] = np.random.uniform(0.7, 1.0, len(mood_content))

        return mood_content.nlargest(5, 'mood_score')[['content_id', 'title', 'genre', 'platform', 'rating', 'mood_score']].to_dict('records')

    def _get_time_based_recommendations(self, user_input):
        """Get recommendations based on time of day"""
        hour = user_input['current_time']
        is_weekend = user_input['current_day'] >= 5

        time_content = self.fire_tv_engine.content_database.copy()

        # Time-based content preferences
        if hour < 8:  # Early morning
            time_content = time_content[time_content['content_type'].isin(['documentary', 'short'])]
        elif hour < 12:  # Morning
            time_content = time_content[time_content['genre'].isin(['comedy', 'animation', 'documentary'])]
        elif hour < 18:  # Afternoon
            time_content = time_content[time_content['duration'] < 90]  # Shorter content
        elif hour < 22:  # Evening
            time_content = time_content[time_content['genre'].isin(['drama', 'action', 'thriller'])]
        else:  # Late night
            time_content = time_content[time_content['genre'].isin(['horror', 'thriller', 'mystery'])]

        # Weekend vs weekday preferences
        if is_weekend:
            time_content = time_content[time_content['duration'] >= 60]  # Longer content on weekends

        time_content['time_score'] = np.random.uniform(0.6, 0.9, len(time_content))

        return time_content.nlargest(5, 'time_score')[['content_id', 'title', 'genre', 'platform', 'rating', 'time_score']].to_dict('records')

    def _get_history_based_recommendations(self, user_input):
        """Get recommendations based on watch history"""
        watch_history = user_input.get('watch_history', [])
        user_profile = user_input.get('user_profile', {})

        if not watch_history:
            # Use profile preferences
            favorite_genres = user_profile.get('favorite_genres', ['comedy'])
            history_content = self.fire_tv_engine.content_database[
                self.fire_tv_engine.content_database['genre'].isin(favorite_genres)
            ].copy()
        else:
            # Find similar content based on watch history
            watched_content_ids = [item['content_id'] for item in watch_history]
            watched_content = self.fire_tv_engine.content_database[
                self.fire_tv_engine.content_database['content_id'].isin(watched_content_ids)
            ]

            # Get genres from watched content
            watched_genres = watched_content['genre'].unique().tolist()

            # Find similar content
            history_content = self.fire_tv_engine.content_database[
                (self.fire_tv_engine.content_database['genre'].isin(watched_genres)) &
                (~self.fire_tv_engine.content_database['content_id'].isin(watched_content_ids))
            ].copy()

        history_content['history_score'] = np.random.uniform(0.8, 1.0, len(history_content))

        return history_content.nlargest(5, 'history_score')[['content_id', 'title', 'genre', 'platform', 'rating', 'history_score']].to_dict('records')

    def _get_social_recommendations(self, user_input):
        """Get recommendations for group watching"""
        group_info = user_input.get('group_activity', {})

        if not group_info.get('is_group_watch', False):
            return []

        group_preferences = group_info.get('group_preferences', ['comedy', 'action'])
        group_size = group_info.get('group_size', 2)

        # Filter content suitable for group watching
        social_content = self.fire_tv_engine.content_database[
            self.fire_tv_engine.content_database['genre'].isin(group_preferences)
        ].copy()

        # Prefer popular and highly rated content for groups
        social_content['social_score'] = (
            social_content['rating'] * 0.4 +
            social_content['popularity_score'] * 0.6
        )

        return social_content.nlargest(3, 'social_score')[['content_id', 'title', 'genre', 'platform', 'rating', 'social_score']].to_dict('records')

    def _combine_recommendations(self, recommendation_lists):
        """Combine multiple recommendation strategies"""
        all_recs = []
        seen_content = set()

        # Add recommendations from each strategy, avoiding duplicates
        for rec_list in recommendation_lists:
            for rec in rec_list:
                if rec['content_id'] not in seen_content:
                    seen_content.add(rec['content_id'])
                    all_recs.append(rec)

        # Sort by various scores (simplified ranking)
        for rec in all_recs:
            scores = [rec.get(key, 0) for key in ['mood_score', 'time_score', 'history_score', 'social_score']]
            rec['final_score'] = np.mean([s for s in scores if s > 0]) * rec.get('rating', 5) / 10

        return sorted(all_recs, key=lambda x: x['final_score'], reverse=True)

    def _get_time_context(self, hour, is_weekend):
        """Get time context description"""
        if hour < 8:
            return "Early Morning" + (" Weekend" if is_weekend else " Weekday")
        elif hour < 12:
            return "Morning" + (" Weekend" if is_weekend else " Weekday")
        elif hour < 18:
            return "Afternoon" + (" Weekend" if is_weekend else " Weekday")
        elif hour < 22:
            return "Evening" + (" Weekend" if is_weekend else " Weekday")
        else:
            return "Late Night" + (" Weekend" if is_weekend else " Weekday")

    def _generate_explanation(self, user_input, recommendations):
        """Generate explanation for recommendations"""
        mood = user_input['current_mood']
        time_context = self._get_time_context(user_input['current_time'], user_input['current_day'] >= 5)
        is_group = user_input.get('group_activity', {}).get('is_group_watch', False)

        explanation = f"Based on your {mood} mood and {time_context.lower()} timing"

        if is_group:
            explanation += " and your group watching session"

        explanation += f", we've selected {len(recommendations)//2} personalized recommendations."

        return explanation

class SocialWatchingEngine:
    """Enhanced Social features for group watching"""

    def __init__(self):
        self.active_sessions = {}
        self.user_preferences = defaultdict(dict)

    def create_group_session(self, group_members, session_preferences=None):
        """Create a group watching session"""
        session_id = f"session_{len(self.active_sessions)}_{datetime.now().strftime('%H%M%S')}"

        session = {
            'session_id': session_id,
            'members': group_members,
            'preferences': session_preferences or {},
            'current_content': None,
            'start_time': datetime.now(),
            'sync_status': {member: 'ready' for member in group_members},
            'chat_history': [],
            'voting_active': False,
            'content_votes': defaultdict(int)
        }

        self.active_sessions[session_id] = session
        return session_id

    def start_content_voting(self, session_id, content_options):
        """Start voting for content selection"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['voting_active'] = True
            self.active_sessions[session_id]['content_options'] = content_options
            return True
        return False

    def cast_vote(self, session_id, user_id, content_id):
        """Cast vote for content"""
        if session_id in self.active_sessions and self.active_sessions[session_id]['voting_active']:
            self.active_sessions[session_id]['content_votes'][content_id] += 1
            return True
        return False

def demo_recommendation_system():
    """Demo the personalized recommendation system"""

    print("Fire TV Personalized Recommendation Engine Demo")
    print("=" * 70)

    # Initialize system
    rec_system = PersonalizedRecommendationSystem()
    rec_system.initialize_system(n_content=1000)

    # Demo user inputs
    demo_users = [
        {
            'user_id': 'alice_123',
            'current_mood': 'relaxed',
            'current_time': 20,  # 8 PM
            'current_day': 5,    # Saturday
            'watch_history': [
                {'content_id': 'content_1', 'rating': 8.5, 'completion_rate': 0.9},
                {'content_id': 'content_15', 'rating': 7.2, 'completion_rate': 0.8}
            ],
            'user_profile': {
                'age_group': '26-35',
                'favorite_genres': ['comedy', 'drama'],
                'avg_session_length': 90
            },
            'group_activity': {
                'is_group_watch': False,
                'group_size': 1
            }
        },
        {
            'user_id': 'bob_456',
            'current_mood': 'excited',
            'current_time': 21,  # 9 PM
            'current_day': 6,    # Sunday
            'watch_history': [
                {'content_id': 'content_5', 'rating': 9.0, 'completion_rate': 1.0},
                {'content_id': 'content_12', 'rating': 8.8, 'completion_rate': 0.95}
            ],
            'user_profile': {
                'age_group': '18-25',
                'favorite_genres': ['action', 'sci-fi'],
                'avg_session_length': 120
            },
            'group_activity': {
                'is_group_watch': True,
                'group_size': 3,
                'group_preferences': ['action', 'thriller']
            }
        },
        {
            'user_id': 'carol_789',
            'current_mood': 'romantic',
            'current_time': 19,  # 7 PM
            'current_day': 0,    # Monday
            'watch_history': [
                {'content_id': 'content_8', 'rating': 7.5, 'completion_rate': 0.85},
                {'content_id': 'content_22', 'rating': 8.0, 'completion_rate': 0.92}
            ],
            'user_profile': {
                'age_group': '36-45',
                'favorite_genres': ['romance', 'drama', 'comedy'],
                'avg_session_length': 75
            },
            'group_activity': {
                'is_group_watch': True,
                'group_size': 2,
                'group_preferences': ['romance', 'comedy']
            }
        }
    ]

    # Process each user
    for i, user_input in enumerate(demo_users, 1):
        print(f"\n{'='*20} USER {i} DEMO {'='*20}")
        print(f" User: {user_input['user_id']}")
        print(f" Mood: {user_input['current_mood']}")
        print(f" Time: {user_input['current_time']}:00 on {'Weekend' if user_input['current_day'] >= 5 else 'Weekday'}")
        print(f" Watch History: {len(user_input['watch_history'])} items")
        print(f" Group Watch: {'Yes' if user_input['group_activity']['is_group_watch'] else 'No'}")

        # Get recommendations
        recommendations = rec_system.get_user_recommendations(user_input)

        print(f"\n🎯 {recommendations['explanation']}")
        print(f"📋 Top Recommendations:")

        for j, rec in enumerate(recommendations['recommendations'][:5], 1):
            print(f"   {j}. {rec['title']} ({rec['genre']}) - {rec['platform']}")
            print(f"       Rating: {rec['rating']:.1f} | Score: {rec.get('final_score', 0):.2f}")

        # Social features demo for group sessions
        if user_input['group_activity']['is_group_watch']:
            print(f"\n👥 Group Session Features:")
            session_id = rec_system.social_engine.create_group_session(
                [user_input['user_id'], 'friend_1', 'friend_2']
            )
            print(f"   • Session ID: {session_id}")
            print(f"   • Group Size: {user_input['group_activity']['group_size']}")
            print(f"   • Preferred Genres: {user_input['group_activity']['group_preferences']}")

            # Start voting demo
            content_options = [rec['content_id'] for rec in recommendations['recommendations'][:3]]
            rec_system.social_engine.start_content_voting(session_id, content_options)
            print(f"   • Voting started for {len(content_options)} content options")



class AdvancedRecommendationFeatures:
    """Additional advanced features for the recommendation system"""

    def __init__(self, recommendation_system):
        self.rec_system = recommendation_system
        self.mood_transition_patterns = self._initialize_mood_patterns()
        self.time_preference_cache = {}

    def _initialize_mood_patterns(self):
        """Initialize mood transition patterns for better predictions"""
        return {
            'happy': ['excited', 'relaxed', 'romantic'],
            'sad': ['relaxed', 'nostalgic', 'happy'],
            'excited': ['happy', 'energetic', 'adventurous'],
            'relaxed': ['happy', 'romantic', 'nostalgic'],
            'stressed': ['relaxed', 'happy', 'energetic'],
            'romantic': ['happy', 'relaxed', 'nostalgic'],
            'adventurous': ['excited', 'energetic', 'happy'],
            'energetic': ['excited', 'adventurous', 'happy'],
            'nostalgic': ['sad', 'relaxed', 'romantic']
        }

    def predict_next_mood(self, current_mood, time_of_day, user_history):
        """Predict user's next mood based on patterns"""
        possible_moods = self.mood_transition_patterns.get(current_mood, ['happy'])

        # Factor in time of day
        if time_of_day > 22:  # Late night
            if 'relaxed' in possible_moods:
                return 'relaxed'
        elif time_of_day < 8:  # Early morning
            if 'energetic' in possible_moods:
                return 'energetic'

        # Default to most common transition
        return possible_moods[0] if possible_moods else 'happy'

    def generate_mood_based_playlist(self, user_input, playlist_length=10):
        """Generate a mood-based content playlist"""
        current_mood = user_input['current_mood']
        playlist = []

        for i in range(playlist_length):
            # Get recommendations for current mood
            mood_input = user_input.copy()
            mood_input['current_mood'] = current_mood

            recommendations = self.rec_system.get_user_recommendations(mood_input)

            if recommendations['recommendations']:
                playlist.append({
                    'position': i + 1,
                    'mood': current_mood,
                    'content': recommendations['recommendations'][0]
                })

                # Predict next mood for variety
                current_mood = self.predict_next_mood(
                    current_mood,
                    user_input['current_time'],
                    user_input.get('watch_history', [])
                )

        return playlist

    def analyze_user_behavior(self, user_id):
        """Analyze user behavior patterns"""
        if user_id not in self.rec_system.fire_tv_engine.user_profiles:
            return None

        profile = self.rec_system.fire_tv_engine.user_profiles[user_id]
        watch_history = profile['watch_history']

        if not watch_history:
            return None

        analysis = {
            'total_content_watched': len(watch_history),
            'average_completion_rate': np.mean([item.get('completion_rate', 1.0) for item in watch_history]),
            'most_active_hours': self._get_most_active_hours(watch_history),
            'mood_patterns': self._analyze_mood_patterns(watch_history),
            'genre_preferences': self._analyze_genre_preferences(user_id),
            'binge_watching_tendency': self._calculate_binge_tendency(watch_history)
        }

        return analysis

    def _get_most_active_hours(self, watch_history):
        """Get user's most active viewing hours"""
        hours = [item['timestamp'].hour for item in watch_history if 'timestamp' in item]
        if not hours:
            return []

        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        return sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    def _analyze_mood_patterns(self, watch_history):
        """Analyze user's mood patterns"""
        moods = [item.get('mood') for item in watch_history if item.get('mood')]
        if not moods:
            return {}

        mood_counts = {}
        for mood in moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1

        return mood_counts

    def _analyze_genre_preferences(self, user_id):
        """Analyze user's genre preferences from watch history"""
        profile = self.rec_system.fire_tv_engine.user_profiles[user_id]
        watch_history = profile['watch_history']

        genre_scores = {}
        for item in watch_history:
            content_id = item['content_id']
            rating = item.get('rating', 5.0)
            completion_rate = item.get('completion_rate', 1.0)

            # Find content details
            content_info = self.rec_system.fire_tv_engine.content_database[
                self.rec_system.fire_tv_engine.content_database['content_id'] == content_id
            ]

            if not content_info.empty:
                genre = content_info.iloc[0]['genre']
                score = rating * completion_rate
                genre_scores[genre] = genre_scores.get(genre, 0) + score

        return genre_scores

    def _calculate_binge_tendency(self, watch_history):
        """Calculate user's binge watching tendency"""
        if len(watch_history) < 2:
            return 0.0

        # Sort by timestamp
        sorted_history = sorted(watch_history, key=lambda x: x.get('timestamp', datetime.now()))

        binge_sessions = 0
        current_session_length = 1

        for i in range(1, len(sorted_history)):
            prev_time = sorted_history[i-1].get('timestamp', datetime.now())
            curr_time = sorted_history[i].get('timestamp', datetime.now())

            # If within 2 hours, consider it same session
            if (curr_time - prev_time).total_seconds() < 7200:  # 2 hours
                current_session_length += 1
            else:
                if current_session_length >= 3:  # 3+ items = binge
                    binge_sessions += 1
                current_session_length = 1

        return binge_sessions / len(watch_history) if watch_history else 0.0

def comprehensive_demo():
    """Comprehensive demo with advanced features"""

    print(" COMPREHENSIVE FIRE TV RECOMMENDATION ENGINE DEMO ")
    print("=" * 80)

    # Initialize system
    rec_system = PersonalizedRecommendationSystem()
    rec_system.initialize_system(n_content=1500)

    # Initialize advanced features
    advanced_features = AdvancedRecommendationFeatures(rec_system)

    # Demo user with comprehensive input
    user_input = {
        'user_id': 'power_user_001',
        'current_mood': 'excited',
        'current_time': 20,  # 8 PM
        'current_day': 5,    # Saturday
        'watch_history': [
            {'content_id': 'content_1', 'rating': 9.0, 'completion_rate': 1.0},
            {'content_id': 'content_15', 'rating': 8.5, 'completion_rate': 0.95},
            {'content_id': 'content_23', 'rating': 7.8, 'completion_rate': 0.88},
            {'content_id': 'content_45', 'rating': 8.2, 'completion_rate': 0.92},
            {'content_id': 'content_67', 'rating': 8.8, 'completion_rate': 0.98}
        ],
        'user_profile': {
            'age_group': '26-35',
            'favorite_genres': ['action', 'sci-fi', 'thriller'],
            'avg_session_length': 105
        },
        'group_activity': {
            'is_group_watch': True,
            'group_size': 4,
            'group_preferences': ['action', 'comedy', 'thriller']
        }
    }

    print(" GETTING PERSONALIZED RECOMMENDATIONS")
    print("-" * 50)

    # Get basic recommendations
    recommendations = rec_system.get_user_recommendations(user_input)

    print(f"👤 User: {user_input['user_id']}")
    print(f"Context: {recommendations['context']['time_context']}")
    print(f"Mood: {recommendations['context']['mood']}")
    print(f"Group Session: {'Yes' if recommendations['context']['is_group_session'] else 'No'}")

    print(f"\n📋 Top 5 Recommendations:")
    for i, rec in enumerate(recommendations['recommendations'][:5], 1):
        print(f"   {i}. {rec['title']} ({rec['genre']}) - {rec['platform']}")
        print(f"       Rating: {rec['rating']:.1f} | Score: {rec.get('final_score', 0):.2f}")

    print(f"\n ADVANCED FEATURES DEMO")
    print("-" * 50)

    # Generate mood-based playlist
    print("\n🎵 Mood-Based Content Playlist:")
    playlist = advanced_features.generate_mood_based_playlist(user_input, 5)
    for item in playlist:
        print(f"   {item['position']}. {item['content']['title']} (Mood: {item['mood']})")

    # User behavior analysis
    print(f"\n User Behavior Analysis:")
    behavior_analysis = advanced_features.analyze_user_behavior(user_input['user_id'])
    if behavior_analysis:
        print(f"   • Total Content Watched: {behavior_analysis['total_content_watched']}")
        print(f"   • Average Completion Rate: {behavior_analysis['average_completion_rate']:.2f}")
        print(f"   • Binge Watching Tendency: {behavior_analysis['binge_watching_tendency']:.2f}")

    # Social features showcase
    print(f"\n SOCIAL FEATURES SHOWCASE")
    print("-" * 50)

    # Create group session
    session_id = rec_system.social_engine.create_group_session(
        ['power_user_001', 'friend_002', 'friend_003', 'friend_004'],
        {'preferred_genres': ['action', 'comedy']}
    )

    print(f" Group Session Created: {session_id}")
    print(f"   • Members: 4 users")
    print(f"   • Preferred Genres: {user_input['group_activity']['group_preferences']}")

    # Start voting
    voting_options = [rec['content_id'] for rec in recommendations['recommendations'][:3]]
    rec_system.social_engine.start_content_voting(session_id, voting_options)
    print(f"   • Voting Started for {len(voting_options)} options")

    # Simulate votes
    for i, option in enumerate(voting_options):
        votes = np.random.randint(1, 5)
        for _ in range(votes):
            rec_system.social_engine.cast_vote(session_id, f'user_{_}', option)

    session = rec_system.social_engine.active_sessions[session_id]
    print(f"   • Vote Results: {dict(session['content_votes'])}")


if __name__ == "__main__":
    # Run the comprehensive demo
    comprehensive_demo()

    # Also run the basic demo
    print("\n" + "="*80)
    demo_recommendation_system()
