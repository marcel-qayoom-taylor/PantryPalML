# PantryPal Recipe Recommendation System

A gradient boosted model (GBM) that combines user interaction history with rich recipe content features to provide personalized recipe recommendations.

## ☁️ Cloud-Executable End-to-End Notebook

Run the entire pipeline (training + inference) with one click. No local setup required.

### End-to-End Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcel-qayoom-taylor/PantryPalML/blob/main/notebooks/A2_Production_EndToEnd.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marcel-qayoom-taylor/PantryPalML/HEAD?labpath=notebooks%2FA2_Production_EndToEnd.ipynb)

This notebook covers: data loading → feature engineering → model training → evaluation → real-time recommendation demo.

### Task I/O (for A2 criteria)
- **Training Input**: CSV files with user–recipe pairs and engineered features (`recipe_recommender/output/hybrid_*_data.csv`)
- **Training Output**: Trained LightGBM model + metadata (`recipe_recommender/output/hybrid_models/`)
- **Inference Input**: `user_id` and interaction events (recipe_id, event_type, timestamp)
- **Inference Output**: Ranked list of recipes with scores and metadata

### How to Run (Colab)
1. Click the Colab badge above
2. Runtime → Run all (installs dependencies, runs full pipeline)
3. Optional: Modify parameters and re-run cells to see different results

### How to Run (Binder/Jupyter)
1. Click the Binder badge above
2. Wait for environment to load
3. Kernel → Restart & Run All

### Notes
- Notebooks use pre-computed datasets for fast execution
- For full data refresh and DB-backed runs, see production scripts below
- See [`notebooks/README.md`](notebooks/README.md) for detailed notebook documentation

## 🎯 What This System Does

**Input:** User's recipe interaction history (viewed, cooked, favourited, etc.)  
**Output:** Scored and ranked recipe recommendations based on predicted user preferences  
**Model:** Hybrid LightGBM combining collaborative filtering + content-based features  

### Key Features
- ⚡ **Real-time scoring**: Score 1,967 recipes in ~0.08 seconds
- 🧠 **Hybrid approach**: Combines user behavior with rich recipe metadata  
- 📊 **Strong ranking performance**: NDCG@5 0.6545, Recall@10 0.9894, Spearman 0.9958
- 🔧 **Production ready**: Clean API for integration
- 🚀 **Scalable**: Handles new users, recipes, and retraining

## 🏗️ System Architecture

```
📱 User Events → 🧠 Hybrid GBM Model → 🍳 Ranked Recipe Recommendations
     │                    │                           │
     │              ┌─────────────┐                   │
     │              │ User Profile│                   │  
     │              │  Features   │                   │
     └──────────────┤             ├───────────────────┘
                    │ Recipe      │
                    │ Content     │
                    │ Features    │
                    └─────────────┘
                           │
                    📊 Supabase Database
                    (1,967 recipes + metadata)
```

### Data Sources
1. **User Interactions**: Recipe events (viewed, cooked, favourited) from analytics
2. **Recipe Database**: Complete PantryPal recipe catalog from Supabase
3. **Recipe Metadata**: Ingredients, tags, timing, authors, descriptions

### Feature Engineering
- **User Features (11)**: Engagement patterns, cooking frequency, preferences
- **Recipe Features (10+)**: Complexity, ingredients, timing, categories
- **Interaction Features (3)**: Compatibility scores between user and recipe characteristics
- **Optional Text Features (configurable)**: Encoded `author_id`, `tags`, `recipe_name`, `description`, `instruction` (see Text Feature Encodings below)

## 🚀 Quick Start - Using the Model

### 1. Setup Environment
```bash
# Install dependencies
uv add pandas numpy scikit-learn lightgbm supabase python-dotenv

# Set up Supabase credentials (required)
export SUPABASE_URL=https://your-project-id.supabase.co
export SUPABASE_ANON_KEY=your-anon-key
export SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

### 2. Get Recipe Recommendations

**Python API:**
```python
from recipe_recommender.inference.recipe_scorer import RecipeScorer

# Initialize the scorer (loads trained model automatically)
scorer = RecipeScorer()

# User's interaction history
user_interactions = [
    {"recipe_id": "100", "event_type": "Recipe Viewed", "timestamp": 1755665991},
    {"recipe_id": "100", "event_type": "Recipe Cooked", "timestamp": 1755666000},
    {"recipe_id": "1291", "event_type": "Recipe Favourited", "timestamp": 1755666100}
]

# Get personalized recommendations
recommendations = scorer.get_user_recommendations(
    user_id="user123",
    interaction_history=user_interactions,
    n_recommendations=10
)

# Use the recommendations
for rec in recommendations['recommendations']:
    print(f"{rec['recipe_name']} (Score: {rec['score']:.4f})")
    print(f"  Author: {rec['author_name']}")
    print(f"  Time: {rec['total_time']}min, Servings: {rec['servings']}")
```

**Command Line:**
```bash
# Get recommendations for a specific user from stored events
python run_inference_example.py USER_ID -n 10

# See detailed usage guide
cat INFERENCE_GUIDE.md
```

For complete inference documentation, see [`INFERENCE_GUIDE.md`](INFERENCE_GUIDE.md).

### 3. Configuration Management
The system now uses centralized configuration for easy customization:

```python
from recipe_recommender.config import get_ml_config

# Get default configuration
config = get_ml_config()

# Customize training parameters
config.learning_rate = 0.05
config.num_leaves = 50
config.negative_sampling_ratio = 3

# Enable simple text encodings (optional)
cfg = config.text_encoding
cfg.enable_text_features = True
cfg.tags_encoding = "topk_multi_hot"  # or "hashing"
cfg.tags_top_k = 50
cfg.author_id_encoding = "freq"       # or "target"
cfg.name_encoding = "hashing"         # small dim hashing for recipe name


# Use custom config
from recipe_recommender.models.recipe_ranker import RecipeRanker
recommender = RecipeRanker(config=config)
```

### 4. Integration Example (FastAPI)
```python
from fastapi import FastAPI
from recipe_recommender.inference.recipe_scorer import RecipeScorer

app = FastAPI()
scorer = RecipeScorer()

@app.post("/recommendations")
async def get_recommendations(user_id: str, interactions: List[Dict]):
    return scorer.get_user_recommendations(
        user_id=user_id,
        interaction_history=interactions,
        n_recommendations=20
    )
```

## 📊 Model Performance

Evaluated on held-out test set (temporal split):
- NDCG@5: 0.6545
- NDCG@10: 0.6545
- Recall@5: 0.9555
- Recall@10: 0.9894
- Spearman correlation: 0.9958

Note: We report ranking metrics suited to recommendation tasks.

**Training Data:**
- 20,730 user-recipe pairs (831 users, 1,967 recipes)
- 22 engineered features combining user behavior + recipe content
- Temporal train/validation/test split (80/10/10)

## 🔄 Retraining with New Data

### When to Retrain
- **New user interactions**: Monthly or when interaction volume increases significantly
- **New recipes**: When new recipes are added to the database
- **Performance degradation**: If recommendation quality decreases

### Retraining Process

#### 1. Update Database
```bash
# Ensure new recipe data is in Supabase
# New user interactions should be in your event tracking
```

#### 2. Extract Fresh Data
```python
# Fetch updated recipe data from Supabase
from recipe_recommender.etl.database.fetch_real_recipes import RealRecipeDataFetcher
fetcher = RealRecipeDataFetcher(use_service_role=True)
dataset = fetcher.build_comprehensive_recipe_dataset()
```

#### 3. Rebuild Training Dataset
```python
# Rebuild training data with latest interactions + recipes
from recipe_recommender.models.training_data_builder import TrainingDataBuilder
builder = TrainingDataBuilder()
train_data, val_data, test_data = builder.prepare_training_data()
```

#### 4. Retrain Model
```python
# Train new model
from recipe_recommender.models.recipe_ranker import RecipeRanker
recommender = RecipeRanker()
recommender.load_data()
recommender.train()
recommender.save_model()
```

#### 5. Deploy Updated Model
```python
# The RecipeScorer will automatically load the new model
scorer = RecipeScorer()  # Loads latest trained model
```

### Complete Retraining Script
```bash
# Run the complete retraining pipeline
cd /path/to/PantryPalUtils

# 1. Fetch latest recipe data
uv run python recipe_recommender/etl/database/fetch_real_recipes.py

# 2. Rebuild training dataset  
uv run python recipe_recommender/models/training_data_builder.py

# 3. Train new model
uv run python recipe_recommender/models/recipe_ranker.py

# 4. Test the updated model
uv run python recipe_recommender/inference/recipe_scorer.py
```

## 📁 Project Structure

### Essential Files (Production)
```
recipe_recommender/
├── config.py                       # 🆕 Centralized configuration management
├── utils/                          # 🆕 Common utilities and helpers
│   ├── __init__.py
│   └── common.py                   # Shared functionality
├── etl/                            # 🆕 Consolidated data extraction & transformation
│   ├── __init__.py
│   ├── database/                   # Database operations
│   │   ├── supabase_config.py      # Database connection setup
│   │   └── fetch_real_recipes.py   # Extract recipe data from Supabase
│   └── events/                     # Event processing (formerly transformations)
│       ├── event_transformation.py # Process user interaction events
│       └── helpers.py              # Utility functions
├── models/
│   ├── training_data_builder.py    # Build training dataset
│   └── recipe_ranker.py            # 🔄 Improved train hybrid GBM model
├── inference/
│   └── recipe_scorer.py            # 🔄 Enhanced production recommendation API
├── analysis/
│   ├── exploratory_data_analysis.py # Data exploration (reference)
│   └── feature_engineering.py      # Feature engineering (reference)
└── output/
    ├── hybrid_models/              # Trained model files
    ├── hybrid_train_data.csv       # Training dataset
    ├── enhanced_recipe_features_from_db.csv  # Recipe features
    └── real_*.csv                  # Raw data from Supabase
```

### 🆕 Key Improvements in This Version

**1. Centralized Configuration (`config.py`)**
- No more hardcoded paths throughout the codebase
- Easy to modify training parameters, file paths, and model settings
- Automatic path detection for different environments

**2. Common Utilities (`utils/`)**
- Eliminated code duplication across files
- Consistent error handling and logging
- Safer file operations with proper validation

**3. Enhanced Documentation**
- Beginner-friendly comments explaining ML concepts
- Clear separation of concerns in each class
- Step-by-step workflow documentation

**4. Better Logging**
- Replaced print statements with proper logging
- Consistent logging format across all modules
- Better debugging and monitoring capabilities

**5. Improved Error Handling**
- Graceful handling of missing files and bad data
- Informative error messages for beginners
- Robust validation of inputs and outputs

### Key Output Files
- **`hybrid_models/hybrid_lightgbm_model.txt`**: Trained LightGBM model
- **`enhanced_recipe_features_from_db.csv`**: Complete recipe dataset with features
- **`hybrid_training_metadata.json`**: Model training configuration and stats

## 🔧 Configuration

### Supabase Setup
1. Go to your Supabase project → Settings → API
2. Set environment variables (or create a .env file in the project root):
```bash
export SUPABASE_URL=https://your-project-id.supabase.co
export SUPABASE_ANON_KEY=your-anon-key
export SUPABASE_SERVICE_ROLE_KEY=your-service-role-key  # optional for admin access
```
3. Test the connection:
```bash
uv run python recipe_recommender/etl/database/supabase_config.py
```

### Database Requirements
The system expects these Supabase tables:
- **`recipe`**: Recipe metadata (names, timing, authors, tags)
- **`ingredient`**: Ingredient catalog with categories
- **`ingredients_of_recipe`**: Recipe-ingredient relationships

## 💡 Understanding the Model

### How It Works
1. **User Profiling**: Analyzes interaction patterns to understand preferences
2. **Recipe Content**: Leverages rich recipe metadata (ingredients, complexity, etc.)
3. **Compatibility Scoring**: Matches user preferences with recipe characteristics
4. **Ranking**: Orders recipes by predicted preference likelihood

### Key Insights
- **Most Important Features**: User-recipe compatibility scores matter most
- **Content + Collaborative**: Hybrid approach outperforms pure collaborative filtering
- **Real-time Performance**: Can score thousands of recipes instantly
- **Cold Start Handling**: Works for new users with minimal interaction data

## 🎓 Learning Resources

### For ML Beginners
The codebase includes extensive beginner-friendly documentation:
- **📖 Inline code comments** explain ML concepts in simple terms
- **🔧 Step-by-step workflow** methods with clear explanations
- **💡 Configuration system** makes customization easy to understand
- **🛠️ Comprehensive error handling** with helpful troubleshooting messages

## 🧪 Testing and Validation

### Test the Model
```python
# Quick test
from recipe_recommender.inference.recipe_scorer import RecipeScorer
scorer = RecipeScorer()

# Test with sample interactions
test_interactions = [
    {"recipe_id": "100", "event_type": "Recipe Cooked", "timestamp": 1755666000}
]

result = scorer.get_user_recommendations("test_user", test_interactions)
print(f"Generated {len(result['recommendations'])} recommendations")
```

### Model Evaluation
Run the evaluation script to check model performance:
```bash
uv run python recipe_recommender/models/recipe_ranker.py
```

## 📈 Monitoring and Maintenance

### Key Metrics to Monitor
- **Recommendation CTR**: Click-through rates on recommended recipes
- **Cooking Rate**: How often recommended recipes are actually cooked
- **User Engagement**: Changes in overall recipe interaction rates
- **Model Performance**: NDCG@k, Recall@k, Spearman correlation on held-out data

### Model Refresh Schedule
- **Weekly**: Monitor performance metrics
- **Monthly**: Retrain with new interaction data
- **Quarterly**: Full evaluation and potential architecture updates

## 🔮 Future Enhancements

### Potential Improvements
1. **Seasonal Recommendations**: Incorporate seasonal ingredient preferences
2. **Nutritional Filtering**: Add dietary restrictions and nutrition goals
3. **Social Features**: Include recipe ratings and reviews
4. **Real-time Learning**: Online learning from user feedback
5. **Multi-objective Optimization**: Balance variety, novelty, and relevance

### Data Sources to Consider
- Recipe ratings and reviews from users
- Nutritional information for health-conscious recommendations
- Seasonal ingredient availability
- User pantry/grocery list data for ingredient-based recommendations

## 🐛 Troubleshooting

### Common Issues

**"No model found"**
- Make sure you've run the training pipeline: `training_data_builder.py` → `recipe_ranker.py`

**"No recipe data"**  
- Verify Supabase connection: `python recipe_recommender/etl/database/supabase_config.py`
- Run data fetcher: `python recipe_recommender/etl/database/fetch_real_recipes.py`

**"Low recommendation quality"**
- Check if user has sufficient interaction history (minimum 3-5 interactions recommended)
- Verify training data quality and feature engineering
- Consider retraining with more recent data

**Performance issues**
- For very large recipe catalogs, consider adding recipe filtering by category/cuisine
- Cache user profiles for frequent requests
- Use batch scoring for multiple users

### Support
For technical issues:
1. Check the logs in each script for detailed error messages
2. Verify all dependencies are installed: `uv sync`
3. Ensure Supabase credentials are correctly set
4. Review the `real_database_summary.txt` for data quality issues

---

## 📋 Summary

This system provides a complete solution for personalized recipe recommendations:

✅ **Trained hybrid GBM model** with excellent performance  
✅ **Production-ready API** for real-time recommendations  
✅ **Comprehensive feature engineering** combining user behavior + recipe content  
✅ **Scalable retraining pipeline** for continuous improvement  
✅ **Clean codebase** with essential files only  

**Ready for production deployment and can serve personalized recipe recommendations at scale!** 🎉