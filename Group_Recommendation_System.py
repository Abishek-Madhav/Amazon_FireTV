def _analyze_individual_member(self, member_id, profile, content_database):
    """Deep analysis of individual member preferences"""
    watch_history = profile.get('watch_history', [])
    ratings_history = profile.get('ratings_history', {})
    mood_preferences = profile.get('mood_preferences', {})

    member_analysis = {
        'member_id': member_id,
        'favorite_genres': [],
        'disliked_genres': [],
        'average_rating': 7.0,
        'rating_variance': 1.0,
        'preferred_duration': 90,
        'duration_flexibility': 30,
        'mood_preferences': {},
        'platform_usage': {},
        'content_type_preference': {},
        'time_preferences': {},
        'completion_rate_avg': 0.9,
        'genre_diversity_score': 0.5,
        'rating_strictness': 0.5
    }

    if watch_history:
        # Analyze watch history
        watched_content_ids = [item['content_id'] for item in watch_history]
        watched_content = content_database[content_database['content_id'].isin(watched_content_ids)]

        if not watched_content.empty:
            # Genre preferences based on watch frequency
            genre_counts = watched_content['genre'].value_counts()
            secondary_genre_counts = watched_content['secondary_genre'].value_counts()
            all_genre_counts = genre_counts.add(secondary_genre_counts, fill_value=0)

            # Top genres (liked)
            member_analysis['favorite_genres'] = all_genre_counts.head(3).index.tolist()

            # Calculate average duration preference
            member_analysis['preferred_duration'] = int(watched_content['duration'].mean())
            member_analysis['duration_flexibility'] = int(watched_content['duration'].std() or 30)

            # Platform usage analysis
            platform_counts = watched_content['platform'].value_counts(normalize=True)
            member_analysis['platform_usage'] = platform_counts.to_dict()

            # Content type preferences
            content_type_counts = watched_content['content_type'].value_counts(normalize=True)
            member_analysis['content_type_preference'] = content_type_counts.to_dict()

            # Calculate completion rates
            completion_rates = [item.get('completion_rate', 1.0) for item in watch_history]
            member_analysis['completion_rate_avg'] = np.mean(completion_rates)

            # Genre diversity score
            unique_genres = len(all_genre_counts)
            member_analysis['genre_diversity_score'] = min(unique_genres / 8.0, 1.0)  # Normalized to max 8 genres

    if ratings_history:
        # Analyze rating patterns
        ratings = list(ratings_history.values())
        member_analysis['average_rating'] = np.mean(ratings)
        member_analysis['rating_variance'] = np.var(ratings)
        member_analysis['rating_strictness'] = 1.0 - (member_analysis['average_rating'] / 10.0)

        # Find disliked genres (low-rated content)
        low_rated_content = [cid for cid, rating in ratings_history.items() if rating <= 4.0]
        if low_rated_content:
            disliked_content = content_database[content_database['content_id'].isin(low_rated_content)]
            if not disliked_content.empty:
                disliked_genres = disliked_content['genre'].value_counts()
                member_analysis['disliked_genres'] = disliked_genres.head(2).index.tolist()

    # Mood preferences analysis
    if mood_preferences:
        total_mood_entries = sum(len(content_list) for content_list in mood_preferences.values())
        if total_mood_entries > 0:
            member_analysis['mood_preferences'] = {
                mood: len(content_list) / total_mood_entries
                for mood, content_list in mood_preferences.items()
            }

    # Fill defaults if no data available
    if not member_analysis['favorite_genres']:
        member_analysis['favorite_genres'] = profile.get('favorite_genres', ['comedy', 'drama'])

    if not member_analysis['mood_preferences']:
        member_analysis['mood_preferences'] = {'happy': 0.3, 'relaxed': 0.3, 'excited': 0.4}

    if not member_analysis['platform_usage']:
        member_analysis['platform_usage'] = {'Netflix': 0.4, 'Prime Video': 0.3, 'Disney+': 0.3}

    if not member_analysis['content_type_preference']:
        member_analysis['content_type_preference'] = {'movie': 0.6, 'series': 0.4}

    return member_analysis

def _find_common_genres(self, member_data):
    """Find genres that are preferred by majority of group members"""
    genre_scores = defaultdict(float)
    total_members = len(member_data)

    for member in member_data:
        favorite_genres = member.get('favorite_genres', [])
        for i, genre in enumerate(favorite_genres):
            # Weight by preference order (first choice gets higher weight)
            weight = (len(favorite_genres) - i) / len(favorite_genres)
            genre_scores[genre] += weight

    # Normalize scores and filter for majority preference
    common_genres = []
    for genre, score in genre_scores.items():
        normalized_score = score / total_members
        if normalized_score >= 0.4:  # At least 40% weighted preference
            common_genres.append((genre, normalized_score))

    # Sort by score and return top genres
    common_genres.sort(key=lambda x: x[1], reverse=True)
    return [genre for genre, score in common_genres[:5]]

def _find_conflicting_genres(self, member_data):
    """Identify genres that create conflicts in the group"""
    genre_preferences = defaultdict(list)

    for member in member_data:
        favorite_genres = member.get('favorite_genres', [])
        disliked_genres = member.get('disliked_genres', [])

        for genre in favorite_genres:
            genre_preferences[genre].append(1)  # Like
        for genre in disliked_genres:
            genre_preferences[genre].append(-1)  # Dislike

    conflicting_genres = []
    for genre, preferences in genre_preferences.items():
        if len(preferences) > 1:  # Multiple opinions
            likes = sum(1 for p in preferences if p > 0)
            dislikes = sum(1 for p in preferences if p < 0)
            if likes > 0 and dislikes > 0:
                conflict_score = min(likes, dislikes) / len(preferences)
                conflicting_genres.append((genre, conflict_score))

    conflicting_genres.sort(key=lambda x: x[1], reverse=True)
    return [genre for genre, score in conflicting_genres[:3]]

def _calculate_group_rating_preferences(self, member_data):
    """Calculate group's rating preferences and standards"""
    all_avg_ratings = [member.get('average_rating', 7.0) for member in member_data]
    rating_variances = [member.get('rating_variance', 1.0) for member in member_data]
    rating_strictness = [member.get('rating_strictness', 0.5) for member in member_data]

    return {
        'group_avg_rating': np.mean(all_avg_ratings),
        'group_rating_std': np.std(all_avg_ratings),
        'avg_rating_variance': np.mean(rating_variances),
        'avg_strictness': np.mean(rating_strictness),
        'min_acceptable_rating': max(5.0, np.mean(all_avg_ratings) - np.std(all_avg_ratings))
    }

def _find_preferred_content_types(self, member_data):
    """Find group's preferred content types"""
    content_type_scores = defaultdict(float)
    total_members = len(member_data)

    for member in member_data:
        content_prefs = member.get('content_type_preference', {})
        for content_type, preference in content_prefs.items():
            content_type_scores[content_type] += preference

    # Normalize and sort
    normalized_prefs = {
        ctype: score / total_members
        for ctype, score in content_type_scores.items()
    }

    return sorted(normalized_prefs.items(), key=lambda x: x[1], reverse=True)

def _analyze_mood_compatibility(self, member_data):
    """Analyze mood compatibility across group members"""
    mood_compatibility = defaultdict(float)
    total_members = len(member_data)

    for member in member_data:
        mood_prefs = member.get('mood_preferences', {})
        for mood, preference in mood_prefs.items():
            mood_compatibility[mood] += preference

    # Calculate mood harmony scores
    mood_harmony = {}
    for mood, total_pref in mood_compatibility.items():
        avg_preference = total_pref / total_members
        # Calculate variance in mood preferences
        member_mood_prefs = [
            member.get('mood_preferences', {}).get(mood, 0)
            for member in member_data
        ]
        mood_variance = np.var(member_mood_prefs)

        # Higher harmony = high average preference + low variance
        harmony_score = avg_preference * (1 - mood_variance)
        mood_harmony[mood] = harmony_score

    return mood_harmony

def _calculate_optimal_duration(self, member_data):
    """Calculate optimal content duration for the group"""
    preferred_durations = [member.get('preferred_duration', 90) for member in member_data]
    duration_flexibilities = [member.get('duration_flexibility', 30) for member in member_data]

    # Calculate weighted average considering flexibility
    total_weight = 0
    weighted_duration = 0

    for duration, flexibility in zip(preferred_durations, duration_flexibilities):
        weight = 1 / (flexibility + 1)  # Higher flexibility = lower weight
        weighted_duration += duration * weight
        total_weight += weight

    optimal_duration = int(weighted_duration / total_weight) if total_weight > 0 else 90

    return optimal_duration

def _analyze_platform_preferences(self, member_data):
    """Analyze group's platform preferences"""
    platform_scores = defaultdict(float)
    total_members = len(member_data)

    for member in member_data:
        platform_usage = member.get('platform_usage', {})
        for platform, usage in platform_usage.items():
            platform_scores[platform] += usage

    # Normalize scores
    normalized_platform_prefs = {
        platform: score / total_members
        for platform, score in platform_scores.items()
    }

    return dict(sorted(normalized_platform_prefs.items(), key=lambda x: x[1], reverse=True))

def _calculate_group_compatibility(self, member_data):
    """Calculate overall group compatibility score"""
    if len(member_data) < 2:
        return 1.0

    compatibility_factors = []

    # Genre compatibility
    all_favorite_genres = [set(member.get('favorite_genres', [])) for member in member_data]
    if len(all_favorite_genres) > 1:
        genre_overlaps = []
        for i in range(len(all_favorite_genres)):
            for j in range(i + 1, len(all_favorite_genres)):
                overlap = len(all_favorite_genres[i] & all_favorite_genres[j])
                union = len(all_favorite_genres[i] | all_favorite_genres[j])
                jaccard_similarity = overlap / union if union > 0 else 0
                genre_overlaps.append(jaccard_similarity)

        genre_compatibility = np.mean(genre_overlaps)
        compatibility_factors.append(genre_compatibility)

    # Rating compatibility (how similar are their rating standards)
    avg_ratings = [member.get('average_rating', 7.0) for member in member_data]
    rating_compatibility = 1 - (np.std(avg_ratings) / 10.0)  # Lower std = higher compatibility
    compatibility_factors.append(max(0, rating_compatibility))

    # Duration compatibility
    durations = [member.get('preferred_duration', 90) for member in member_data]
    duration_std = np.std(durations)
    duration_compatibility = max(0, 1 - (duration_std / 60.0))  # Normalize by 60 minutes
    compatibility_factors.append(duration_compatibility)

    # Mood compatibility
    mood_agreement = []
    all_moods = set()
    for member in member_data:
        all_moods.update(member.get('mood_preferences', {}).keys())

    for mood in all_moods:
        mood_prefs = [member.get('mood_preferences', {}).get(mood, 0) for member in member_data]
        mood_std = np.std(mood_prefs)
        mood_agreement.append(1 - mood_std)  # Lower std = higher agreement

    if mood_agreement:
        mood_compatibility = np.mean(mood_agreement)
        compatibility_factors.append(max(0, mood_compatibility))

    # Overall compatibility score
    return np.mean(compatibility_factors) if compatibility_factors else 0.5

def _identify_consensus_factors(self, member_data):
    """Identify factors that can help reach group consensus"""
    consensus_factors = {
        'genre_compromise_needed': False,
        'duration_compromise_needed': False,
        'platform_limitation': None,
        'rating_threshold': 7.0,
        'mood_alignment': 'medium',
        'flexibility_score': 0.5,
        'decision_strategy': 'democratic'
    }

    # Check if genre compromise is needed
    common_genres = self._find_common_genres(member_data)
    conflicting_genres = self._find_conflicting_genres(member_data)
    consensus_factors['genre_compromise_needed'] = len(conflicting_genres) > 0

    # Check duration compromise
    durations = [member.get('preferred_duration', 90) for member in member_data]
    duration_range = max(durations) - min(durations)
    consensus_factors['duration_compromise_needed'] = duration_range > 60

    # Determine rating threshold
    rating_prefs = self._calculate_group_rating_preferences(member_data)
    consensus_factors['rating_threshold'] = rating_prefs['min_acceptable_rating']

    # Calculate flexibility score
    flexibility_scores = []
    for member in member_data:
        genre_diversity = member.get('genre_diversity_score', 0.5)
        duration_flex = member.get('duration_flexibility', 30) / 60.0  # Normalize
        rating_variance = member.get('rating_variance', 1.0) / 10.0  # Normalize

        member_flexibility = (genre_diversity + duration_flex + rating_variance) / 3
        flexibility_scores.append(member_flexibility)

    consensus_factors['flexibility_score'] = np.mean(flexibility_scores)

    # Determine decision strategy
    if consensus_factors['flexibility_score'] > 0.7:
        consensus_factors['decision_strategy'] = 'flexible_consensus'
    elif len(member_data) <= 3:
        consensus_factors['decision_strategy'] = 'unanimous'
    else:
        consensus_factors['decision_strategy'] = 'majority_vote'

    return consensus_factors

def generate_group_recommendations(self, group_analysis, content_database):
    """Generate the best recommendations for the group based on comprehensive analysis"""
    print(f" Generating group recommendations with compatibility score: {group_analysis['group_compatibility_score']:.2f}")

    recommendations = []

    # Get group preferences
    common_genres = group_analysis.get('common_genres', [])
    conflicting_genres = group_analysis.get('conflicting_genres', [])
    rating_prefs = group_analysis.get('average_ratings', {})
    min_rating = rating_prefs.get('min_acceptable_rating', 6.0)
    optimal_duration = group_analysis.get('recommended_duration', 90)
    platform_prefs = group_analysis.get('platform_preferences', {})
    content_type_prefs = group_analysis.get('preferred_content_types', [])
    consensus_factors = group_analysis.get('consensus_factors', {})

    # Filter content based on group analysis
    filtered_content = content_database.copy()

    # Apply rating filter
    filtered_content = filtered_content[filtered_content['rating'] >= min_rating]

    # Apply genre filters
    if common_genres:
        genre_filter = (
            filtered_content['genre'].isin(common_genres) |
            filtered_content['secondary_genre'].isin(common_genres)
        )
        filtered_content = filtered_content[genre_filter]

    # Remove conflicting genres if group compatibility is low
    if conflicting_genres and group_analysis['group_compatibility_score'] < 0.6:
        conflict_filter = ~(
            filtered_content['genre'].isin(conflicting_genres) |
            filtered_content['secondary_genre'].isin(conflicting_genres)
        )
        filtered_content = filtered_content[conflict_filter]

    # Apply duration preferences (with tolerance)
    duration_tolerance = 30  # minutes
    duration_filter = (
        (filtered_content['duration'] >= optimal_duration - duration_tolerance) &
        (filtered_content['duration'] <= optimal_duration + duration_tolerance)
    )
    filtered_content = filtered_content[duration_filter]

    # Prefer top platforms
    if platform_prefs:
        top_platforms = list(platform_prefs.keys())[:3]
        platform_filter = filtered_content['platform'].isin(top_platforms)
        filtered_content = filtered_content[platform_filter]

    # Apply content type preferences
    if content_type_prefs:
        preferred_types = [ctype for ctype, score in content_type_prefs if score > 0.3]
        if preferred_types:
            type_filter = filtered_content['content_type'].isin(preferred_types)
            filtered_content = filtered_content[type_filter]

    if filtered_content.empty:
        print(" No content found matching all criteria, relaxing filters...")
        # Fallback with relaxed criteria
        filtered_content = content_database[content_database['rating'] >= min_rating - 1.0]

    # Calculate group compatibility scores for each content
    scored_content = []

    for _, content in filtered_content.iterrows():
        score_components = {}

        # Genre alignment score
        genre_score = 0
        if content['genre'] in common_genres:
            genre_score += 0.4
        if content['secondary_genre'] in common_genres:
            genre_score += 0.2
        if content['genre'] not in conflicting_genres:
            genre_score += 0.2
        score_components['genre_score'] = genre_score

        # Rating score (normalize content rating to group preferences)
        rating_diff = abs(content['rating'] - rating_prefs.get('group_avg_rating', 7.0))
        rating_score = max(0, 1 - (rating_diff / 5.0))
        score_components['rating_score'] = rating_score

        # Duration score
        duration_diff = abs(content['duration'] - optimal_duration)
        duration_score = max(0, 1 - (duration_diff / 60.0))
        score_components['duration_score'] = duration_score

        # Platform preference score
        platform_score = platform_prefs.get(content['platform'], 0.1)
        score_components['platform_score'] = platform_score

        # Popularity and trending bonus
        popularity_score = min(content['popularity_score'] / 10.0, 0.3)
        trending_bonus = 0.1 if content['is_trending'] else 0
        score_components['popularity_score'] = popularity_score + trending_bonus

        # Calculate weighted final score
        weights = {
            'genre_score': 0.35,
            'rating_score': 0.25,
            'duration_score': 0.15,
            'platform_score': 0.15,
            'popularity_score': 0.10
        }

        final_score = sum(score * weights[component] for component, score in score_components.items())

        # Apply group compatibility multiplier
        compatibility_multiplier = 0.8 + (group_analysis['group_compatibility_score'] * 0.4)
        final_score *= compatibility_multiplier

        content_recommendation = {
            'content_id': content['content_id'],
            'title': content['title'],
            'genre': content['genre'],
            'secondary_genre': content['secondary_genre'],
            'platform': content['platform'],
            'content_type': content['content_type'],
            'duration': content['duration'],
            'rating': content['rating'],
            'is_trending': content['is_trending'],
            'group_compatibility_score': final_score,
            'social_score': final_score,  # For compatibility with main system
            'score_breakdown': score_components,
            'recommendation_reason': self._generate_recommendation_reason(content, group_analysis)
        }

        scored_content.append(content_recommendation)

    # Sort by group compatibility score
    scored_content.sort(key=lambda x: x['group_compatibility_score'], reverse=True)

    # Select top recommendations with diversity
    final_recommendations = self._ensure_recommendation_diversity(scored_content[:15], group_analysis)

    print(f" Generated {len(final_recommendations)} group recommendations")

    return final_recommendations[:8]  # Return top 8 diverse recommendations

def _generate_recommendation_reason(self, content, group_analysis):
    """Generate explanation for why this content is recommended for the group"""
    reasons = []

    common_genres = group_analysis.get('common_genres', [])
    if content['genre'] in common_genres or content['secondary_genre'] in common_genres:
        reasons.append(f"matches group's favorite genres")

    if content['rating'] >= group_analysis.get('average_ratings', {}).get('min_acceptable_rating', 6.0):
        reasons.append(f"meets group's quality standards")

    optimal_duration = group_analysis.get('recommended_duration', 90)
    if abs(content['duration'] - optimal_duration) <= 20:
        reasons.append(f"fits group's preferred viewing time")

    if content['is_trending']:
        reasons.append(f"currently trending")

    platform_prefs = group_analysis.get('platform_preferences', {})
    if content['platform'] in list(platform_prefs.keys())[:2]:
        reasons.append(f"available on preferred platform")

    if group_analysis['group_compatibility_score'] > 0.7:
        reasons.append(f"great compatibility with all members")
    elif group_analysis['group_compatibility_score'] > 0.5:
        reasons.append(f"good compromise for the group")

    return "; ".join(reasons) if reasons else "recommended for your group"

def _ensure_recommendation_diversity(self, recommendations, group_analysis):
    """Ensure diversity in final recommendations"""
    if not recommendations:
        return []

    diverse_recs = []
    used_genres = set()
    used_platforms = set()
    used_content_types = set()

    # First pass: add highest scoring items with diversity constraints
    for rec in recommendations:
        add_item = True

        # Genre diversity
        if len(diverse_recs) >= 2 and rec['genre'] in used_genres and len(used_genres) < 4:
            add_item = False

        # Platform diversity
        if len(diverse_recs) >= 3 and rec['platform'] in used_platforms and len(used_platforms) < 3:
            add_item = False

        # Content type diversity
        if len(diverse_recs) >= 4 and rec['content_type'] in used_content_types and len(used_content_types) < 2:
            add_item = False

        if add_item:
            diverse_recs.append(rec)
            used_genres.add(rec['genre'])
            used_platforms.add(rec['platform'])
            used_content_types.add(rec['content_type'])

            if len(diverse_recs) >= 8:
                break

    # Second pass: fill remaining slots if needed
    if len(diverse_recs) < 6:
        for rec in recommendations:
            if rec not in diverse_recs:
                diverse_recs.append(rec)
                if len(diverse_recs) >= 8:
                    break

    return diverse_recs

# Enhanced usage example showing the group watch functionality
def demonstrate_group_watch_system():
    """Demonstrate the enhanced group watching recommendation system"""

    # Initialize the system
    rec_system = PersonalizedRecommendationSystem()
    rec_system.initialize_system(n_content=3000)

    # Create sample user profiles with rich watch histories
    users_data = {
        'alice_123': {
            'age_group': '26-35',
            'favorite_genres': ['comedy', 'romance', 'drama'],
            'avg_session_length': 120,
            'watch_history': [
                {'content_id': 'content_45', 'rating': 8.5, 'completion_rate': 1.0, 'mood': 'happy'},
                {'content_id': 'content_123', 'rating': 7.0, 'completion_rate': 0.9, 'mood': 'relaxed'},
                {'content_id': 'content_234', 'rating': 9.0, 'completion_rate': 1.0, 'mood': 'romantic'}
            ]
        },
        'bob_456': {
            'age_group': '26-35',
            'favorite_genres': ['action', 'sci-fi', 'thriller'],
            'avg_session_length': 90,
            'watch_history': [
                {'content_id': 'content_67', 'rating': 8.0, 'completion_rate': 1.0, 'mood': 'excited'},
                {'content_id': 'content_189', 'rating': 6.5, 'completion_rate': 0.8, 'mood': 'energetic'},
                {'content_id': 'content_298', 'rating': 8.5, 'completion_rate': 1.0, 'mood': 'adventurous'}
            ]
        },
        'carol_789': {
            'age_group': '36-45',
            'favorite_genres': ['documentary', 'drama', 'mystery'],
            'avg_session_length': 75,
            'watch_history': [
                {'content_id': 'content_89', 'rating': 9.0, 'completion_rate': 1.0, 'mood': 'relaxed'},
                {'content_id': 'content_156', 'rating': 7.5, 'completion_rate': 0.95, 'mood': 'nostalgic'},
                {'content_id': 'content_367', 'rating': 8.0, 'completion_rate': 1.0, 'mood': 'stressed'}
            ]
        }
    }

    # Create user profiles in the system
    for user_id, data in users_data.items():
        rec_system.fire_tv_engine.create_user_profile(
            user_id=user_id,
            age_group=data['age_group'],
            favorite_genres=data['favorite_genres'],
            avg_session_length=data['avg_session_length']
        )

    # Simulate group watching session
    group_watch_input = {
        'user_id': 'group_session_001',  # Group session ID
        'current_mood': 'relaxed',
        'current_time': 20,  # 8 PM
        'current_day': 5,    # Saturday
        'user_profile': {
            'age_group': '26-35',
            'favorite_genres': ['drama'],  # This will be overridden by group analysis
            'avg_session_length': 90
        },
        'watch_history': [],  # Group doesn't have individual watch history
        'group_activity': {
            'is_group_watch': True,
            'group_members': ['alice_123', 'bob_456', 'carol_789'],
            'session_type': 'movie_night',
            'group_mood': 'mixed'
        }
    }

    print(" FIRE TV GROUP WATCH RECOMMENDATION DEMO")
    print("=" * 50)

    # Get group recommendations
    recommendations = rec_system.get_user_recommendations(group_watch_input)

    # Display results
    print(f"\n Group Members: {', '.join(group_watch_input['group_activity']['group_members'])}")
    print(f" Context: {recommendations['context']}")
    print(f" Explanation: {recommendations['explanation']}")

    print(f"\n TOP GROUP RECOMMENDATIONS:")
    print("-" * 60)

    for i, rec in enumerate(recommendations['recommendations'][:5], 1):
        print(f"{i}. {rec['title']}")
        print(f"   Genre: {rec['genre']} | Platform: {rec['platform']}")
        print(f"   Rating: {rec['rating']:.1f} | Duration: {rec.get('duration', 90):.0f} min")
        print(f"   Group Score: {rec.get('group_compatibility_score', 0):.2f}")
        print(f"   Why: {rec.get('recommendation_reason', 'Great for your group!')}")
        print()

    return recommendations

# Run the demonstration
if __name__ == "__main__":
    demo_results = demonstrate_group_watch_system()
