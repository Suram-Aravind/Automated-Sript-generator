import tensorflow as tf
import numpy as np
import pickle
import os
import re

# --- Hyperparameters (Must match model training) ---
VOCAB_SIZE = 8000        # Must match the VOCAB_SIZE used for training the loaded model
MAX_PARAGRAPH_LEN = 250  # Must match the MAX_PARAGRAPH_LEN used for training

# --- Fixed File Paths (Shared) ---
MODEL_SAVE_PATH = "paragraph_lm.keras"
TOKENIZER_SAVE_PATH = "paragraph_tokenizer.pkl"

# --- Utility Functions (Shared - Copied for script independence) ---
def clean_text(text): # Simplified for inference input3
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_tokenizer(path):
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print(f"Tokenizer loaded from {path}")
        return tokenizer
    print(f"ERROR: Tokenizer not found at {path}. Train a model first.")
    return None

def sample_from_probs(probs, temp):
    probs = np.asarray(probs).astype('float64')
    if temp <= 0 or temp == float('inf'): # temp=0 greedy, temp=inf uniform random (approx)
        return np.argmax(probs)

    # Add a small epsilon to prevent log(0) and ensure sum is not zero after division
    probs = np.log(probs + 1e-9) / temp
    exp_preds = np.exp(probs)
    preds = exp_preds / (np.sum(exp_preds) + 1e-9) # Add epsilon to denominator too

    try:
        # Ensure probabilities sum to 1 for multinomial
        preds = preds / np.sum(preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    except ValueError as e:
        # print(f"Warning: Multinomial sampling failed ('{e}'). Sum of preds: {np.sum(preds)}. Falling back to argmax.")
        return np.argmax(preds) # Fallback to greedy if sampling fails
# --- End Utility Functions ---

def generate_text_for_inference(model, tokenizer, seed_text, max_generated_len=50, temperature=1.0):
    print(f"\nGenerating text from seed: '{seed_text}' with temp {temperature}, max_len {max_generated_len}")

    current_text_for_model = clean_text(seed_text) # Model expects cleaned input
    generated_suffix = "" # Store only newly generated words to append to original seed

    # The model's vocabulary size, including padding (index 0) and OOV
    # This should match the output dimension of the model's dense layer.
    model_vocab_size = VOCAB_SIZE + 1

    for _ in range(max_generated_len):
        token_list = tokenizer.texts_to_sequences([current_text_for_model])[0]

        if not token_list:
            # print("Warning: Token list became empty during generation.")
            break

        # Pad the sequence to the model's expected input length
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=MAX_PARAGRAPH_LEN-1, padding='pre' # MAX_PARAGRAPH_LEN-1 for model input
        )

        predicted_probs = model.predict(padded_sequence, verbose=0)[0]

        # Ensure predicted_probs length matches model_vocab_size.
        # This might not be strictly necessary if model construction is always correct, but good sanity check.
        if len(predicted_probs) != model_vocab_size:
            print(f"Warning: Predicted_probs length ({len(predicted_probs)}) "
                  f"mismatches model_vocab_size ({model_vocab_size}).")
            # Potentially pad or truncate predicted_probs, or error out.
            # For now, we assume it matches.

        predicted_word_index = sample_from_probs(predicted_probs, temperature)

        # Stop if padding (index 0) is predicted
        if predicted_word_index == 0:
            # print("Prediction stopped: Padding token predicted.")
            break

        output_word = tokenizer.index_word.get(predicted_word_index)

        # Stop if OOV token is predicted or word not found (shouldn't happen with valid index)
        if not output_word or output_word == tokenizer.oov_token:
            # print(f"Prediction stopped: OOV ('{tokenizer.oov_token}') or no word for index {predicted_word_index}.")
            break

        generated_suffix += " " + output_word
        current_text_for_model += " " + output_word # Append to the input for the next prediction step

        # Optional: Stop on sentence-ending punctuation
        if output_word in ['.', '!', '?'] and len(generated_suffix.strip()) > 0 : # Check len to avoid stopping on first word if it's punctuation
            # print("Sentence end punctuation detected.")
            break

    return seed_text + generated_suffix # Append generated part to original, uncleaned seed


def run_interactive_inference():
    print("\n--- Starting Interactive Inference Mode ---")

    tokenizer = load_tokenizer(TOKENIZER_SAVE_PATH)
    if not tokenizer:
        return

    # Verify tokenizer's num_words matches VOCAB_SIZE hyperparameter. Critical for consistency.
    if tokenizer.num_words != VOCAB_SIZE:
        print(f"CRITICAL WARNING: Loaded tokenizer has num_words={tokenizer.num_words}, "
              f"but Inference script's VOCAB_SIZE is {VOCAB_SIZE}. These must match the settings "
              f"used during model training. Inference might be incorrect.")
        # Decide whether to proceed or abort. For now, we proceed with a warning.

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at {MODEL_SAVE_PATH}. Train a model first.")
        return

    try:
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print("Model loaded successfully.")
        # model.summary() # Optional: print summary
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nEnter 'quit' to exit inference mode.")
    while True:
        try:
            user_prompt = input("Enter your prompt: ")
            if user_prompt.lower() == 'quit':
                break
            if not user_prompt.strip():
                print("Prompt cannot be empty.")
                continue

            temp_str = input(f"Enter temperature (e.g., 0.7, default 1.0, 0 for greedy): ")
            try:
                temperature = float(temp_str) if temp_str.strip() else 1.0
            except ValueError:
                print("Invalid temperature, using 1.0.")
                temperature = 1.0

            max_len_str = input(f"Max new words to generate (e.g., 50, default 50): ")
            try:
                max_gen_len = int(max_len_str) if max_len_str.strip() else 50
                if max_gen_len <=0: max_gen_len = 50
            except ValueError:
                print("Invalid max length, using 50.")
                max_gen_len = 50

            response = generate_text_for_inference(model, tokenizer, user_prompt,
                                                   max_generated_len=max_gen_len,
                                                   temperature=temperature)
            print(f"\nModel Response:\n{response}\n")

        except KeyboardInterrupt:
            print("\nInference interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            import traceback
            traceback.print_exc()

    print("--- Exiting Inference Mode ---")

if __name__ == "__main__":
    run_interactive_inference()