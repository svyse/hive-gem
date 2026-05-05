import streamlit as st
from transformers import pipeline

# Load a lightweight text generation pipeline
# Using 'distilgpt2' as it's relatively small and good for simple text generation
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

st.title("Interactive Greeter with Hugging Face")
st.markdown("--- Say hello to the AI! ---")

user_name = st.text_input("What's your name?", "Guest")
user_prompt = st.text_area("Enter your message here:", f"Hello, my name is {user_name}. Can you greet me?")

if st.button("Get AI Response"):
    if user_prompt:
        with st.spinner("Generating response..."):
            # Generate text, limiting length to keep it concise and relevant
            # Adjust max_new_tokens as needed for desired response length
            response = generator(
                user_prompt,
                max_new_tokens=50, # Limit the length of the generated response
                num_return_sequences=1,
                truncation=True # Truncate input if it's too long for the model
            )
            # The generated_text often includes the prompt itself, so we might want to trim it.
            # For distilgpt2, it usually appends to the prompt.
            generated_text = response[0]['generated_text']
            # A simple way to try and get only the AI's part is to remove the prompt if it's at the start
            if generated_text.startswith(user_prompt):
                ai_response = generated_text[len(user_prompt):].strip()
            else:
                ai_response = generated_text.strip()
            
            st.success(f"**AI says:** {ai_response}")
    else:
        st.warning("Please enter a message to get a response.")

st.markdown("--- ")
st.markdown("This app uses a lightweight Hugging Face `distilgpt2` model for text generation.")
