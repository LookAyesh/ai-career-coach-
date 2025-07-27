import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import sys

# Load environment variables from .env file
load_dotenv()

# --- Step 1: Check if the API token is loaded ---
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(f"Python version: {sys.version}")
print("-" * 30)

if api_token:
    # We'll just show the first few characters to confirm it's loaded
    print("✅ Hugging Face API token loaded successfully.")
    print(f"   Token starts with: {api_token[:5]}...")
else:
    print("❌ ERROR: Hugging Face API token not found!")
    print("   Please make sure your .env file is in the same directory as this script,")
    print("   and the variable is named HUGGINGFACEHUB_API_TOKEN.")
    exit() # Exit the script if no token is found

print("-" * 30)

# --- Step 2: Try to run a direct inference call ---
print("Attempting to contact Hugging Face Inference API...")
try:
    # Initialize the client directly
    client = InferenceClient(token=api_token)

    # Make a simple API call
    response = client.text_generation(
        model="google/flan-t5-large",
        prompt="The capital of France is",
        max_new_tokens=10
    )

    print("\n✅ SUCCESS! Received a response from the API.")
    print("Response:", response)

except Exception as e:
    print(f"\n❌ FAILED: An error occurred during the API call.")
    print("Error Type:", type(e)._name_)
    print("Error Details:", e)
    import traceback
    traceback.print_exc()