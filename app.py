import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import gdown
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_DRIVE_LINK = "https://drive.google.com/file/d/1Gtg0HpRvvX7zAn5N1e4JPJC4sfaU2SwL/view?usp=sharing"  # <-- replace with your .h5 file link
MODEL_PATH = "food_rogonition_model.h5"
IMG_SIZE = (128, 128)

# Nutrition datasets
nutrition_df = pd.read_csv("/content/drive/MyDrive/ML project/new/FOOD-DATA-GROUP1.csv")
mapping_df = pd.read_csv("/content/drive/MyDrive/ML project/new/healthy_eating_dataset.csv")

# -----------------------------
# DOWNLOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(MODEL_DRIVE_LINK, MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names (adjust if you saved separately)
# Example: if you trained with tf.keras.utils.image_dataset_from_directory
# class_names = train_ds.class_names
# For demo, define manually:
class_names = sorted(nutrition_df['food'].unique().tolist())

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

    # Nutrition lookup
    pred_clean = predicted_class.lower().strip()
    nutrition_df['food_clean'] = nutrition_df['food'].str.lower().str.strip()
    mapping_df['meal_name_clean'] = mapping_df['meal_name'].str.lower().str.strip()

    nutrients = {}
    try:
        row = nutrition_df.loc[nutrition_df['food_clean'] == pred_clean].iloc[0]
        nutrients.update({
            "Calories": row.get("Caloric Value"),
            "Protein": row.get("Protein"),
            "Carbohydrates": row.get("Carbohydrates"),
            "Fat": row.get("Fat")
        })
    except:
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
