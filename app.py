import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import gdown
import os
from rapidfuzz import process

# -----------------------------
# CONFIG
# -----------------------------
MODEL_DRIVE_ID = "1Gtg0HpRvvX7zAn5N1e4JPJC4sfaU2SwL"  # <-- extracted from your Drive link
MODEL_PATH = "food_recognition_model.h5"
IMG_SIZE = (128, 128)

# Nutrition datasets
nutrition_df = pd.read_csv("FOOD-DATA-GROUP1.csv")
mapping_df = pd.read_csv("healthy_eating_dataset.csv")

# -----------------------------
# DOWNLOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}", MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names (adjust if needed)
class_names = sorted(nutrition_df['food'].dropna().unique().tolist())

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_food(image):
    img = tf.keras.utils.load_img(image, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Clean names
    pred_clean = predicted_class.lower().strip()
    nutrition_df['food_clean'] = nutrition_df['food'].str.lower().str.strip()
    mapping_df['meal_name_clean'] = mapping_df['meal_name'].str.lower().str.strip()

    # Fuzzy match
    match_nutri = process.extractOne(pred_clean, nutrition_df['food_clean'])
    match_map = process.extractOne(pred_clean, mapping_df['meal_name_clean'])

    nutrients = {}
    if match_nutri and match_nutri[1] > 60:
        row = nutrition_df.loc[nutrition_df['food_clean'] == match_nutri[0]].iloc[0]
        nutrients.update({
            "Calories": row.get("Caloric Value"),
            "Protein": row.get("Protein"),
            "Carbohydrates": row.get("Carbohydrates"),
            "Fat": row.get("Fat")
        })

    if match_map and match_map[1] > 60:
        row2 = mapping_df.loc[mapping_df['meal_name_clean'] == match_map[0]].iloc[0]
        nutrients.update({
            "Diet Type": row2.get("diet_type"),
            "Is Healthy": row2.get("is_healthy"),
            "Cuisine": row2.get("cuisine")
        })

    if not nutrients:
        nutrients = "Not found in nutrition DB"

    return predicted_class, confidence, nutrients

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🍲 Food Recognition & Nutrition App")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    food, conf, nutri = predict_food(uploaded_file)
    st.write(f"**Predicted Dish:** {food}")
    st.write(f"**Confidence:** {conf:.2f}%")
    st.write("**Nutrition Info:**")
    st.write(nutri)
