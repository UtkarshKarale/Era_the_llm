import gradio as gr
import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/chat"

def chat_with_llm(message, history):
    try:
        logger.info(f"Sending request to API: {message[:50]}...")
        response = requests.post(
            API_URL,
            json={"message": message},
            timeout=120  # Increased timeout to 120 seconds
        )
        response.raise_for_status()
        result = response.json()
        logger.info("Received response from API")
        return result["response"]
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to the API server. Please make sure the backend is running."
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except requests.exceptions.Timeout:
        error_msg = "Request timed out. The model is taking too long to respond. Please try again with a shorter prompt."
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {error_msg}"

# Create Gradio interface with error handling
demo = gr.ChatInterface(
    chat_with_llm,
    title="Local LLM Chat Interface",
    description="Chat with a local LLM using FastAPI backend. Make sure the backend server is running at http://localhost:8000",
    theme=gr.themes.Soft(),
    examples=[
        "Hello, how are you?",
        "What can you help me with?",
        "Tell me a short story"
    ]
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(share=False) 