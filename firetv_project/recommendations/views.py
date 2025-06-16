from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
from .models import Content, UserProfile, ViewingSession, Recommendation
from .serializers import (
    RecommendationRequestSerializer, 
    RecommendationResponseSerializer,
    ContentSerializer
)
from .ml_models import recommendation_engine

@api_view(['POST'])
def generate_recommendations(request):
    """Generate AI-powered recommendations based on user context"""
    serializer = RecommendationRequestSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'error': 'Invalid input data',
            'details': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    user_data = serializer.validated_data
    
    try:
        # Get or create user profile
        user_profile, created = UserProfile.objects.get_or_create(
            user_id=user_data['user_id'],
            defaults={
                'age_group': user_data['age_group'],
                'avg_session_duration': user_data['avg_session_duration'],
                'social_activity_level': user_data['social_activity_level'],
                'avg_rating_given': user_data['avg_rating_given'],
                'preferred_genres': json.dumps([user_data['recent_genre_preference']])
            }
        )
        
        # Update profile with latest preferences
        if not created:
            user_profile.age_group = user_data['age_group']
            user_profile.avg_session_duration = user_data['avg_session_duration']
            user_profile.social_activity_level = user_data['social_activity_level']
            user_profile.avg_rating_given = user_data['avg_rating_given']
            
            # Update preferred genres
            try:
                current_genres = json.loads(user_profile.preferred_genres)
                if user_data['recent_genre_preference'] not in current_genres:
                    current_genres.append(user_data['recent_genre_preference'])
                    user_profile.preferred_genres = json.dumps(current_genres[-5:])  # Keep last 5
            except:
                user_profile.preferred_genres = json.dumps([user_data['recent_genre_preference']])
            
            user_profile.save()
        
        # Record viewing session for context
        viewing_session = ViewingSession.objects.create(
            user_profile=user_profile,
            mood=user_data['mood'],
            hour_of_day=user_data['hour_of_day'],
            day_of_week=user_data['day_of_week'],
            is_weekend=user_data['is_weekend'],
            is_group_watch=user_data['is_group_watch'],
            group_size=user_data['group_size']
        )
        
        # Generate recommendations using ML engine
        recommendations_data = recommendation_engine.generate_recommendations(user_data)
        
        # Save recommendations to database
        recommendations = []
        for rec_data in recommendations_data:
            recommendation = Recommendation.objects.create(
                user_profile=user_profile,
                content=rec_data['content'],
                confidence_score=rec_data['score'],
                reasoning=json.dumps(rec_data['reasons']),
                recommendation_type='ai_hybrid',
                context_mood=user_data['mood'],
                context_time=user_data['hour_of_day'],
                context_group_size=user_data['group_size']
            )
            recommendations.append(recommendation)
        
        # Serialize response
        serializer = RecommendationResponseSerializer(recommendations, many=True)
        
        return Response({
            'success': True,
            'user_id': user_data['user_id'],
            'recommendations': serializer.data,
            'context': {
                'mood': user_data['mood'],
                'time_slot': f"{user_data['hour_of_day']}:00",
                'group_size': user_data['group_size'],
                'genre_preference': user_data['recent_genre_preference']
            },
            'total_recommendations': len(recommendations)
        })
        
    except Exception as e:
        return Response({
            'error': 'Failed to generate recommendations',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_all_content(request):
    """Get all available content"""
    content = Content.objects.all()
    serializer = ContentSerializer(content, many=True)
    return Response({
        'content': serializer.data,
        'total_count': content.count()
    })

@api_view(['POST'])
def track_interaction(request):
    """Track user interactions with recommendations"""
    try:
        data = request.data
        recommendation_id = data.get('recommendation_id')
        interaction_type = data.get('interaction_type')  # 'click', 'watch', 'like', 'skip'
        
        recommendation = Recommendation.objects.get(id=recommendation_id)
        
        if interaction_type == 'click':
            recommendation.was_clicked = True
            recommendation.click_timestamp = timezone.now()
        elif interaction_type == 'watch':
            recommendation.was_watched = True
        
        recommendation.save()
        
        return Response({'success': True})
        
    except Recommendation.DoesNotExist:
        return Response({'error': 'Recommendation not found'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def train_model(request):
    """Trigger ML model training"""
    try:
        success = recommendation_engine.train_model()
        return Response({
            'success': success,
            'message': 'Model training completed' if success else 'Insufficient data for training'
        })
    except Exception as e:
        return Response({
            'error': 'Training failed',
            'details': str(e)
        }, status=500)

@api_view(['GET'])
def health_check(request):
    """API health check"""
    return Response({
        'status': 'healthy',
        'service': 'Fire TV AI Recommendations',
        'version': '1.0',
        'timestamp': timezone.now().isoformat(),
        'database_connected': True,
        'ml_model_trained': recommendation_engine.is_trained
    })

@api_view(['GET'])
def get_user_profile(request, user_id):
    """Get user profile information"""
    try:
        profile = UserProfile.objects.get(user_id=user_id)
        from .serializers import UserProfileSerializer
        serializer = UserProfileSerializer(profile)
        return Response({
            'success': True,
            'profile': serializer.data
        })
    except UserProfile.DoesNotExist:
        return Response({
            'error': 'User profile not found'
        }, status=404)

@api_view(['POST'])
def update_user_feedback(request):
    """Update user feedback for content"""
    try:
        data = request.data
        user_id = data.get('user_id')
        content_id = data.get('content_id')
        rating = data.get('rating')
        liked = data.get('liked')
        
        user_profile = UserProfile.objects.get(user_id=user_id)
        content = Content.objects.get(id=content_id)
        
        # Find the most recent viewing session
        session = ViewingSession.objects.filter(
            user_profile=user_profile,
            content=content
        ).first()
        
        if session:
            session.user_rating = rating
            session.liked = liked
            session.save()
        else:
            # Create a new session for feedback
            ViewingSession.objects.create(
                user_profile=user_profile,
                content=content,
                mood='happy',  # Default
                hour_of_day=20,  # Default
                day_of_week=1,  # Default
                is_weekend=False,
                user_rating=rating,
                liked=liked
            )
        
        return Response({'success': True})
        
    except (UserProfile.DoesNotExist, Content.DoesNotExist):
        return Response({'error': 'User or content not found'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)
