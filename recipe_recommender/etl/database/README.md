# Supabase Integration for Recipe Recommendation System

This module integrates your Supabase database to access the complete recipe catalog, enabling content-based and hybrid recommendations.

## 🔧 Setup Instructions

### 1. Install Dependencies
```bash
uv add supabase python-dotenv
```

### 2. Configure Supabase Access

Create a `.env` file in the project root with your Supabase credentials:

```bash
# .env file
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
```

**To get these values:**
1. Go to your Supabase project dashboard
2. Navigate to **Settings** → **API**
3. Copy:
   - **Project URL** → `SUPABASE_URL`
   - **anon/public key** → `SUPABASE_ANON_KEY` 
   - **service_role key** → `SUPABASE_SERVICE_ROLE_KEY`

### 3. Test Connection

```bash
cd recipe_recommender/etl/database
python supabase_config.py
```

You should see: `✅ Supabase connection successful!`

## 📊 Expected Database Schema

The system expects these tables in your Supabase database:

### `recipe` table
- `id` (int/uuid) - Primary key
- `name` (text) - Recipe name
- `author_id` (uuid) - Recipe author reference
- `total_time` (int) - Total cooking time in minutes
- `servings` (int) - Number of servings
- `tags` (jsonb/array) - Recipe tags
- `description` (text) - Recipe description
- `instruction` (text) - Cooking instructions
- `created_at` (timestamp)
- `updated_at` (timestamp)
- Additional recipe metadata...

### `ingredient` table  
- `id` (int/uuid) - Primary key
- `name` (text) - Ingredient name
- `category` (text) - Ingredient category (optional)

### `ingredients_of_recipe` table
- `recipe_id` (foreign key to recipe.id)
- `ingredient_id` (foreign key to ingredient.id)
- `quantity` (text) - Optional quantity
- `unit` (text) - Optional unit

**Note:** Tags are typically stored as JSONB array directly in the `recipe` table rather than in separate tables.

## 🚀 Usage

### Fetch Recipe Data
```python
from recipe_recommender.etl.database.fetch_real_recipes import RealRecipeDataFetcher

# Initialize fetcher (uses centralized config)
fetcher = RealRecipeDataFetcher(use_service_role=True)

# Fetch all recipe data
dataset = fetcher.build_comprehensive_recipe_dataset()

# Access individual DataFrames
recipes = dataset['enhanced_recipes']
ingredients = dataset['recipe_ingredients'] 
raw_ingredients = dataset['ingredients']
```

### Integration with ML Pipeline
```python
# The fetched data automatically integrates with the ML pipeline
from recipe_recommender.models.training_data_builder import TrainingDataBuilder

builder = TrainingDataBuilder()
train_data, val_data, test_data = builder.prepare_training_data()
```

## 🎯 Benefits Over Event-Only Data

### **Current System (Event-Only)**
- Limited to ~1,300 recipes with interactions
- No content-based recommendations
- Cold start problem for new recipes
- Basic recipe features only

### **Enhanced System (Full Catalog)**
- Access to entire recipe database
- **Content-based filtering** via ingredients & tags
- **Hybrid recommendations** (collaborative + content)
- Rich feature engineering:
  - Ingredient similarity matrices
  - Cuisine/diet preferences  
  - Cooking method preferences
  - Nutritional categories
  - Complexity scoring

### **Key Feature Enhancements**

**Ingredient Features:**
- Ingredient similarity (TF-IDF + cosine similarity)
- Ingredient categories (protein, dairy, vegetables)
- Recipe complexity based on ingredient count/diversity

**Tag/Category Features:**
- Dietary restrictions (vegetarian, vegan, gluten-free)
- Cuisine types (Italian, Mexican, Asian, etc.)
- Meal types (breakfast, lunch, dinner, dessert)
- Cooking methods (slow cooker, air fryer, instant pot)

**Content Similarity:**
- Recipe-to-recipe similarity based on ingredients
- User preference matching based on ingredient preferences
- Seasonal/contextual recommendations

## 🔧 Troubleshooting

### Connection Issues
```python
# Test connection
from recipe_recommender.etl.database.supabase_config import SupabaseConfig
config = SupabaseConfig()
config.test_connection()
```

### Schema Issues
Make sure your database schema matches the expected structure. The system will show specific error messages for missing tables or columns.

### Performance
- For large datasets, consider using `limit` parameter initially
- The service role key provides full access but use carefully in production
- Consider adding database indexes on frequently queried columns

## 🧪 Testing

```bash
# Test Supabase connection
python recipe_recommender/etl/database/supabase_config.py

# Fetch latest recipe data
python recipe_recommender/etl/database/fetch_real_recipes.py
```

This will create several output files in `recipe_recommender/output/`:
- `real_*_from_db.csv` - Raw data from Supabase  
- `enhanced_recipe_features_from_db.csv` - Comprehensive recipe features
- `real_database_summary.txt` - Database summary and statistics
- Various metadata files for the ML pipeline
