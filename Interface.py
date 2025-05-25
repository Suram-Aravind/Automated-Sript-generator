# interface
import streamlit as st
import tensorflow as tf
import Inference # Your existing script

# --- Configuration for Streamlit Interface ---
MAX_WORDS_OUTPUT = 260 # Fixed max new words to generate as per your request

@st.cache_resource # Cache the loaded model and tokenizer for efficiency
def load_resources():
    """Loads the tokenizer and model."""
    tokenizer = None
    model = None
    
    # Load Tokenizer
    if not Inference.os.path.exists(Inference.TOKENIZER_SAVE_PATH):
        st.error(f"Tokenizer file not found at {Inference.TOKENIZER_SAVE_PATH}. Please ensure it exists.")
        return None, None
    tokenizer = Inference.load_tokenizer(Inference.TOKENIZER_SAVE_PATH)
    if tokenizer is None: # load_tokenizer prints its own error if path exists but load fails
        st.error(f"Failed to load tokenizer from {Inference.TOKENIZER_SAVE_PATH}.")
        return None, None

    
    # Load Model
    if not Inference.os.path.exists(Inference.MODEL_SAVE_PATH):
        st.error(f"Model file not found at {Inference.MODEL_SAVE_PATH}. Please ensure it exists.")
        return tokenizer, None # Return tokenizer even if model fails, to show tokenizer status
        
    try:
        model = tf.keras.models.load_model(Inference.MODEL_SAVE_PATH)
    except Exception as e:
        st.error(f"Error loading Keras model from {Inference.MODEL_SAVE_PATH}: {e}")
        return tokenizer, None # Return tokenizer here as well

    return tokenizer, model

# --- Streamlit App UI ---
st.title("Automated Script Generator")

# Load resources and display status
tokenizer, model = load_resources()

if tokenizer and model:
    #st.header("Generate Text")
    user_prompt = st.text_area("Ask Anything", height=100, placeholder="How can I help you:")
    
    temperature = st.slider(
        "Select Temperature:",
        min_value=0.0,  # 0 will be greedy (uses argmax)
        max_value=2.0,
        value=1.0,      # Default temperature
        step=0.1,
        help="Controls randomness: 0 = greedy/deterministic, higher values = more random."
    )

    if st.button("Generate Paragraph", type="primary"):
        if not user_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner(f"Generating up to {MAX_WORDS_OUTPUT} new words..."):
                response = Inference.generate_text_for_inference(
                    model=model,
                    tokenizer=tokenizer,
                    seed_text=user_prompt,
                    max_generated_len=MAX_WORDS_OUTPUT,
                    temperature=temperature
                )
            
            st.subheader("Generated Text:")
            st.markdown(response)
            st.markdown("---")

elif tokenizer and not model:
    st.error("Tokenizer loaded, but the Model could not be loaded. Please check the errors above.")
elif not tokenizer and not model:
    st.error("Neither Tokenizer nor Model could be loaded. Please check the errors above and ensure files exist.")
else: # Should not happen if logic is correct
    st.error("An unexpected state occurred with resource loading.")

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown(f"Automated Script Generation Model using Transformers.")
st.sidebar.markdown(f"Made by: ")
st.sidebar.markdown(f"S.Aravind")
st.sidebar.markdown(f"Vikas Pal")
st.sidebar.markdown(f"K.Vishnu Vital")
st.sidebar.markdown(f"Tanish chavan")
st.sidebar.markdown(f"Sri Charan")