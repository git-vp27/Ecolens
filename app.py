import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from bird_info import bird_info
from datetime import datetime
import pytz

import warnings
warnings.filterwarnings("ignore")

timezone = pytz.timezone("Asia/Kolkata")

# ✅ Set page config
st.set_page_config(page_title="Bird Species Detection", layout="wide")

# ✅ Initialize session state for checklist if not already set
if "checklist" not in st.session_state:
    st.session_state.checklist = []

# ✅ Load the model (cached for efficiency)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('densenet_bird_model.h5')  # Update with your model path
    return model

model = load_model()

# ✅ Bird species labels
class_labels = [
    "Asian-Green-Bee-Eater", "Brown-Headed-Barbet", "Cattle-Egret", "Common-Kingfisher",
    "Common-Myna", "Common-Rosefinch", "Common-Tailorbird", "Coppersmith-Barbet",
    "Forest-Wagtail", "Gray-Wagtail", "Hoopoe", "House-Crow",
    "Indian-Grey-Hornbill", "Indian-Peacock", "Indian-Pitta", "Indian-Roller",
    "Jungle-Babbler", "Northern-Lapwing", "Red-Wattled-Lapwing", "Ruddy-Shelduck",
    "Rufous-Treepie", "Sarus-Crane", "White-Breasted-Kingfisher", "White-Breasted-Waterhen",
    "White-Wagtail"
]

# ✅ Home Page
st.title("🦅 Bird Species Detection App")
st.markdown("Choose a feature to continue:")

# ✅ Feature Buttons
feature = st.selectbox("Select a Feature", [
    "Bird Species Prediction Using Image", 
    "Bird Species Prediction Using Audio", 
    "Checklist (Record Bird Sightings)"
])

if feature == "Bird Species Prediction Using Image":
    st.subheader("📷 Upload or Capture an Image")
    option = st.radio("Choose an option:", ["Upload an Image", "Use Camera"])
    
    uploaded_file = None
    if option == "Upload an Image":
        uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    elif option == "Use Camera":
        uploaded_file = st.camera_input("Take a photo")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess image
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        predicted_species = class_labels[predicted_class]
        
        st.success(f"✅ Predicted Bird Species: **{predicted_species}**")
        st.info(f"🎯 Confidence: **{confidence:.2f}%**")

        species_info = bird_info.get(predicted_species, "ℹ️ No information available.")
        st.subheader(f"{predicted_species}:")
        st.markdown(f"""
        - **Description:** {species_info.get('Description', 'N/A')}
        - **Habitat:** {species_info.get('Habitat', 'N/A')}
        - **Diet:** {species_info.get('Diet', 'N/A')}
        - **Conservation Status:** {species_info.get('Conservation Status', 'N/A')}
        """)
        
        # ✅ Get user input for checklist
        date = st.date_input("📅 Date of Sighting")
        time = datetime.now(timezone).strftime("%H:%M")
        st.text_input("⏰ Time of Sighting", time)
        location = st.text_input("📍 Location")
        
        if st.button("Save to Checklist"):
            st.session_state.checklist.append({
                "species": predicted_species,
                "date": str(date),
                "time": str(time),
                "location": location,
                "media": uploaded_file,  # Store image
                "media_type": "image"
            })
            st.success("✅ Bird sighting recorded successfully!")

elif feature == "Bird Species Prediction Using Audio":
    st.subheader("🎙️ Upload Bird Sound")
    uploaded_audio = st.file_uploader("Upload an audio file (WAV, MP3)", type=["wav", "mp3"])
    
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format='audio/wav')
        
        # Placeholder for future model prediction
        predicted_species = "Unknown (Audio Feature Coming Soon)"
        
        st.success(f"✅ Predicted Bird Species: **{predicted_species}**")
        
        # ✅ Get user input for checklist
        date = st.date_input("📅 Date of Sighting")
        location = st.text_input("📍 Location")
        
        if st.button("Save to Checklist"):
            st.session_state.checklist.append({
                "species": predicted_species,
                "date": str(date),
                "location": location,
                "media": uploaded_audio,  # Store audio
                "media_type": "audio"
            })
            st.success("✅ Bird sighting recorded successfully!")

elif feature == "Checklist (Record Bird Sightings)":
    st.subheader("📋 Recorded Bird Sightings")

    if not st.session_state.checklist:
        st.info("No sightings recorded yet. Make a prediction first!")
    else:
        to_delete = None  # Track the index to delete

        for i, sighting in enumerate(st.session_state.checklist):
            with st.expander(f"📌 {sighting['species']} - {sighting['date']}"):
                st.write(f"📅 **Date:** {sighting['date']}")
                st.write(f"⏰ **Time:** {sighting['time']}")
                st.write(f"📍 **Location:** {sighting['location']}")

                if sighting["media"]:
                    if sighting["media_type"] == "image":
                        st.image(sighting["media"], caption=sighting["species"], use_container_width=True)
                    elif sighting["media_type"] == "audio":
                        st.audio(sighting["media"], format='audio/wav')

                # ✅ Use st.form to avoid key conflicts
                with st.form(key=f"delete_form_{i}"):
                    if st.form_submit_button(f"🗑️ Delete {sighting['species']}"):
                        to_delete = i

        # ✅ If an entry is marked for deletion, remove it
        if to_delete is not None:
            del st.session_state.checklist[to_delete]
            st.success("✅ Sighting deleted successfully!")
            st.rerun()


# ✅ Sidebar - About
st.sidebar.header("About")
st.sidebar.write("""
This app predicts bird species from images and audio using a **Deep Learning model (DenseNet121)**.
- 📷 Upload an image or use your **camera**.
- 🎙️ Upload an audio file for bird sound detection.
- 📝 Save and track bird sightings.
""")
