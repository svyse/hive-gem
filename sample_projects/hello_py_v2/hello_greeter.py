from transformers import pipeline

# Initialize a lightweight text generation pipeline
# Using 'distilgpt2' as a lightweight option
# This will download the model the first time it's run
try:
    generator = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    print(f"Warning: Could not initialize Hugging Face pipeline ({e}). Ensure you have an internet connection and sufficient memory.")
    generator = None

def get_interactive_greeting(name):
    """
    Generates a personalized and interactive greeting using a Hugging Face transformer.
    Falls back to a simple greeting if the transformer is not available.
    """
    if generator:
        prompt = f"Hello {name}! It's wonderful to connect with you. I'm thinking about what an amazing day you're going to have. Perhaps you will"
        
        try:
            # Generate text based on the prompt
            # max_new_tokens controls the length of the generated response
            # num_return_sequences=1 ensures only one response is generated
            # truncation=True handles long prompts if necessary
            response = generator(prompt, max_new_tokens=50, num_return_sequences=1, truncation=True)
            generated_text = response[0]['generated_text']
            
            # The model might repeat the prompt, so try to extract only the new part
            if generated_text.startswith(prompt):
                return generated_text
            else:
                # Fallback if the model doesn't perfectly prepend the prompt
                return f"Greetings, {name}! {generated_text}"
        except Exception as e:
            print(f"Warning: Error during text generation ({e}). Falling back to simple greeting.")
            return f"Greetings, {name}!"
    else:
        return f"Greetings, {name}!"

if __name__ == "__main__":
    user_name = input("Please enter your name: ")
    print(get_interactive_greeting(user_name))
