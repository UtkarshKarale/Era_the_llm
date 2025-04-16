# Local LLM Chat Application

This project implements a chat interface for a local LLM using FastAPI and Gradio. It uses the Mistral-7B-Instruct model from Hugging Face and includes conversation memory using LangChain.

## Features

- FastAPI backend serving the LLM
- Gradio web interface for easy interaction
- Conversation memory using LangChain
- GPU acceleration support (if available)
- Docker support for easy deployment

## Requirements

- Python 3.9+
- CUDA-compatible GPU (optional, but recommended)
- Docker (optional)

## Installation

### Local Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation

Build and run using Docker:
```bash
docker build -t llm-chat-app .
docker run -p 8000:8000 -p 7860:7860 llm-chat-app
```

## Usage

1. Start the FastAPI backend:
   ```bash
   python app.py
   ```

2. In a separate terminal, start the Gradio interface:
   ```bash
   python frontend.py
   ```

3. Access the chat interface at http://localhost:7860

The API will be available at:
- Chat endpoint: http://localhost:8000/chat
- API documentation: http://localhost:8000/docs

## API Endpoints

### POST /chat
Send a message to the LLM and get a response.

Request body:
```json
{
    "message": "Your message here"
}
```

Response:
```json
{
    "response": "Model's response",
    "conversation_history": [
        {
            "user": "Previous user message",
            "assistant": "Previous assistant response"
        }
    ]
}
```

## Notes

- The application will automatically use GPU acceleration if available
- The model is loaded in 4-bit precision on GPU for memory efficiency
- Conversation history is maintained throughout the session #   E r a _ t h e _ l l m  
 