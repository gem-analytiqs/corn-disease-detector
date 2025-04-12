import streamlit as st
import tensorflow as tf
import tarfile
import os
import shutil
import numpy as np

# Configure Streamlit page
st.set_page_config(page_title="Corn Disease Classifier", layout="wide")

@st.cache_resource
def load_model():
    """Load the model from the compressed tar.gz archive"""
    try:
        # Path configuration
        MODEL_ARCHIVE = 'crop_disease_model.tar.gz'
        EXTRACT_DIR = 'temp_model_extract'
        
        # Debug: Show current directory contents
        st.sidebar.write("Current directory files:", os.listdir('.'))
        
        # Verify the archive exists
        if not os.path.exists(MODEL_ARCHIVE):
            raise FileNotFoundError(
                f"Model archive not found at: {os.path.abspath(MODEL_ARCHIVE)}"
            )

        # Create extraction directory
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        
        # Extract the gzipped tar file
        with tarfile.open(MODEL_ARCHIVE, 'r:gz') as tar:
            tar.extractall(path=EXTRACT_DIR)
            
            # Find the model file in extracted contents
            model_file = None
            for root, _, files in os.walk(EXTRACT_DIR):
                for file in files:
                    if file.endswith(('.h5', '.keras')):
                        model_file = os.path.join(root, file)
                        break
                if model_file:
                    break
            
            if not model_file:
                raise ValueError("No .h5 or .keras file found in archive")
            
            # Load the model
            model = tf.keras.models.load_model(model_file)
            st.sidebar.success("Model loaded successfully!")
            return model
            
    except Exception as e:
        st.sidebar.error(f"MODEL LOADING FAILED: {str(e)}")
        return None
    finally:
        # Clean up extracted files
        if os.path.exists(EXTRACT_DIR):
            shutil.rmtree(EXTRACT_DIR)

# --- Main App ---
def main():
    st.title("🌽 Corn Disease Classification")
    
    # Load model (cached)
    model = load_model()
    
    if model:
        # Display model summary
        with st.expander("Model Architecture"):
            st.text(model.summary())
        
        # Prediction UI
        st.subheader("Upload Corn Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False
        )
        
        if uploaded_file:
            try:
                # Preprocess image to match model input requirements
                # (Modify this section based on your model's needs)
                image = tf.image.decode_image(uploaded_file.read(), channels=3)
                image = tf.image.resize(image, [224, 224])  # Example size
                image = np.expand_dims(image, axis=0) / 255.0
                
                # Make prediction
                prediction = model.predict(image)
                st.success(f"Prediction confidence: {prediction[0][0]:.2%}")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    else:
        st.warning("Waiting for model to load...")

if __name__ == "__main__":
    main()