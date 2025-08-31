# Recipe Recommendation Inference Guide

This guide shows you how to run recipe recommendations for users using their historical interaction data.

## Quick Start

### 1. Get User Interaction History

```bash
# Get a summary of user activity
python -m recipe_recommender.utils.fetch_user_interactions 123a02b5-cac8-4d1b-973e-3a9fe0f2303d --summary

# Get detailed interactions
python -m recipe_recommender.utils.fetch_user_interactions 123a02b5-cac8-4d1b-973e-3a9fe0f2303d

# Save interactions to a file
python -m recipe_recommender.utils.fetch_user_interactions 123a02b5-cac8-4d1b-973e-3a9fe0f2303d --output user_interactions.json
```

### 2. Run Complete Inference Pipeline

```bash
# Get 10 recommendations (default)
python run_inference_example.py 123a02b5-cac8-4d1b-973e-3a9fe0f2303d

# Get 5 recommendations
python run_inference_example.py 123a02b5-cac8-4d1b-973e-3a9fe0f2303d -n 5

# Save results to file
python run_inference_example.py 123a02b5-cac8-4d1b-973e-3a9fe0f2303d --output recommendations.json

# Just see interactions without running inference
python run_inference_example.py 123a02b5-cac8-4d1b-973e-3a9fe0f2303d --interactions-only
```

## How It Works

### 1. User Interaction Fetching (`UserInteractionFetcher`)

The system fetches user interactions from the cleaned Mixpanel data in `recipe_recommender/output/combined_events.csv`. It looks for these recipe-related events:

- `Recipe Viewed`
- `Recipe Cooked` 
- `Recipe Favourited`
- `Recipe Link Clicked`
- `Recipe Cook Started`
- `Recipe Added To Collections`
- `Recipe Removed From Collections`
- `Recipe Search Queried`

**Important**: You **DO need to provide interaction history** - the model doesn't store user data, it only learns patterns from the training data.

### 2. User Profile Creation

The system creates a user profile from interactions including:
- Total interactions
- Average rating (based on interaction weights)
- Activity patterns (days active, interactions per day)
- Engagement scores

### 3. Recipe Scoring

The trained LightGBM model scores all 1,967+ available recipes based on:
- User profile features
- Recipe content features (ingredients, complexity, time, etc.)
- User-recipe interaction features

### 4. Recommendation Ranking

Recipes are ranked by score and returned with metadata including:
- Recipe name, author, URL
- Cooking time and servings
- Ingredient count and complexity
- Tags and description

## Example Output

```bash
🍳 RECIPE RECOMMENDATIONS FOR USER: 123a02b5-cac8-4d1b-973e-3a9fe0f2303d
================================================================================
📊 Model: hybrid_lightgbm
⚡ Processing time: 0.085 seconds
📈 Total recipes scored: 1,967
🎯 User interactions: 11
🏆 Recommendations returned: 5

🏆 TOP 5 RECOMMENDATIONS:
--------------------------------------------------------------------------------

1. Instant Pot Boiled Eggs
   📊 Score: 0.3775
   👨‍🍳 Author: Well Plated
   ⏱️  20min • 1 servings
   🔧 1 ingredients • complexity: 0.27
   🏷️  Gluten Free
   📝 How to make hard or soft Instant Pot Boiled Eggs...
   🔗 https://www.wellplated.com/instant-pot-boiled-eggs/
```

## Programmatic Usage

```python
from recipe_recommender.utils.fetch_user_interactions import UserInteractionFetcher
from recipe_recommender.inference.production_recipe_scorer import ProductionRecipeScorer

# Fetch user interactions
fetcher = UserInteractionFetcher()
interactions = fetcher.fetch_user_interactions("123a02b5-cac8-4d1b-973e-3a9fe0f2303d")

# Get recommendations
scorer = ProductionRecipeScorer()
result = scorer.get_user_recipe_recommendations(
    user_id="123a02b5-cac8-4d1b-973e-3a9fe0f2303d",
    interaction_history=interactions,
    n_recommendations=10
)

# Access recommendations
for rec in result["recommendations"]:
    print(f"{rec['recipe_name']}: {rec['score']:.4f}")
```

## Files Created

- `recipe_recommender/utils/fetch_user_interactions.py` - Fetches user interactions from cleaned data
- `run_inference_example.py` - Complete inference pipeline example
- `INFERENCE_GUIDE.md` - This guide

## Requirements

- Trained model files in `recipe_recommender/output/hybrid_models/`
- Recipe features in `recipe_recommender/output/enhanced_recipe_features_from_db.csv`
- User events in `recipe_recommender/output/combined_events.csv`
- Virtual environment activated: `source .venv/bin/activate`

## Performance

- **Processing time**: ~85ms to score 1,967 recipes
- **Memory efficient**: Reads large event files in chunks
- **Scalable**: Can handle millions of events efficiently

## Notes

- Recipe IDs are currently extracted as "unknown" from routes - you may need to enhance the `_extract_recipe_id` method if you have actual recipe IDs in your event data
- The system handles users with no interactions by using default profiles
- All timestamps are Unix timestamps and automatically converted to readable dates
- The model uses 22 features combining user behavior and recipe characteristics
