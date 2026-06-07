import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast


import re
import ast

def clean_ingredients(ingredient_str):
    ingredients = ast.literal_eval(ingredient_str)
    cleaned = ' '.join(ingredients).lower()
    cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned)
    return cleaned

@st.cache_resource
def load_models():
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    model = pickle.load(open(os.path.join(base_path, 'model.pkl'), 'rb'))
    tfidf = pickle.load(open(os.path.join(base_path, 'tfidf.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))
    recipes = pickle.load(open(os.path.join(base_path, 'recipes.pkl'), 'rb'))
    recipes['ingredients_clean'] = recipes['ingredients'].apply(clean_ingredients)
    return model, tfidf, scaler, recipes


def clean_input(text):
    return text.lower().strip()

def get_recommendations(user_ingredients, max_calories, max_minutes, top_n=5):
    # TF-IDF transform user input
    user_input_clean = clean_input(user_ingredients)
    user_vector = tfidf.transform([user_input_clean])
    
    #TF-IDF transform all recipes
    recipe_vectors = tfidf.transform(recipes['ingredients_clean'])
    
    #Cosine similarity
    similarities = cosine_similarity(user_vector, recipe_vectors).flatten()
    
    #Filter by calories and minutes
    filtered = recipes.copy()
    filtered['similarity'] = similarities
    filtered = filtered[
        (filtered['calories'] <= max_calories) &
        (filtered['minutes'] <= max_minutes) &
        (filtered['similarity'] > 0)
    ]
    
    #Take top 50 most similar
    candidates = filtered.nlargest(50, 'similarity')
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    #Logistic Regression re-ranking
    X_tfidf = tfidf.transform(candidates['ingredients_clean'])
    numeric = candidates[['calories', 'minutes', 'avg_recipe_rating']].values
    
    import scipy.sparse as sp
    from sklearn.preprocessing import StandardScaler
    numeric_scaled = scaler.transform(numeric)
    numeric_sparse = sp.csr_matrix(numeric_scaled)
    X = sp.hstack([X_tfidf, numeric_sparse])
    
    # Get probability of "Like"
    proba = model.predict_proba(X)[:, 1]
    candidates = candidates.copy()
    candidates['like_probability'] = proba
    
    #Final score = similarity + like probability
    candidates['final_score'] = (
        candidates['similarity'] * 0.4 + 
        candidates['like_probability'] * 0.6
    )
    
    results = candidates.nlargest(top_n, 'final_score')
    return results



st.set_page_config(
    page_title="Food Recipe Recommender",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Personalized Food Recipe Recommender")
st.write("Enter ingredients you have at home and we'll recommend good recipes!")

st.sidebar.header("Your Preferences")

user_ingredients = st.sidebar.text_input(
    "Ingredients you have (comma separated)",
    placeholder="e.g. chicken, garlic"
)

max_calories = st.sidebar.slider(
    "Max Calories",
    min_value=100,
    max_value=2000,
    value=800,
    step=50
)

max_minutes = st.sidebar.slider(
    "Max Cooking Time (minutes)",
    min_value=10,
    max_value=300,
    value=60,
    step=10
)

top_n = st.sidebar.selectbox(
    "Number of Recipes",
    options=[3, 5, 10],
    index=1
)



if st.sidebar.button("🔍 Find Recipes", use_container_width=True):
    if not user_ingredients:
        st.warning("Please enter at least one ingredient!")
    else:
        with st.spinner("Finding your perfect recipes..."):
            results = get_recommendations(
                user_ingredients,
                max_calories,
                max_minutes,
                top_n
            )
        
        if results.empty:
            st.error("No recipes found! Try different ingredients or adjust your filters.")
        else:
            st.success(f"Found {len(results)} recipes for you!")
            st.divider()
            
            for i, (_, row) in enumerate(results.iterrows()):
                col1, col2 = st.columns([3, 1])
                

                with col1:
                    st.subheader(f"{i+1}. {row['name'].title()}")
                    
                    # Show ingredients
                    try:
                        ingredients_list = ast.literal_eval(row['ingredients'])
                        st.write("**Ingredients:**", ', '.join(ingredients_list))
                    except:
                        st.write("**Ingredients:**", row['ingredients'])
                    
                    # Show cooking steps
                    try:
                        steps_list = ast.literal_eval(row['steps'])
                        with st.expander("📖 View Cooking Steps"):
                            for step_num, step in enumerate(steps_list, 1):
                                st.write(f"**Step {step_num}:** {step.capitalize()}")
                    except:
                        pass
                
                with col2:
                    st.metric("Match Score", f"{round(row['final_score']*100)}%")
                    st.metric("Calories", f"{round(row['calories'])} kcal")
                    st.metric("Cook Time", f"{round(row['minutes'])} mins")
                    st.metric("Avg Rating", f"⭐ {round(row['avg_recipe_rating'], 1)}")
                
                st.divider()

else:

    # Center the content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.divider()
        st.markdown("## How it works:")
        st.write("")
        st.write("**Step 1:** Type the ingredients you have at home in the sidebar")
        st.write("**Step 2:** Set your maximum calorie limit using the slider")
        st.write("**Step 3:** Set your maximum cooking time")
        st.write("**Step 4:** Choose how many recipes you want to see")
        st.write("**Step 5:** Click **🔍 Find Recipes** to get your recommendations!")
        st.caption("ℹ️ Recipes are sourced from Food.com, containing 180,000+ recipes and 1M+ user interactions.")

        # st.info("Recipes are sourced from Food.com, containing 180,000+ recipes and 1M+ user interactions.")