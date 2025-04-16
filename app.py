# This is a FastAPI application that serves a Large Language Model (LLM) chat interface
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from typing import List, Dict
import os
import logging
import re

# Configure logging to track application events and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(title="LLM Chat API")

# Determine if GPU (CUDA) is available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    # Log GPU information if using CUDA
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Specify the LLM model to use - Mistral 7B Instruct
MODEL_NAME = "facebook/opt-350m"  # Smaller, open-access model
model = None
tokenizer = None
memory = ConversationBufferMemory()  # Store conversation history

# Create a simple prompt template for LangChain
template = """The following is a friendly conversation between a human and an AI. The AI is helpful, creative, clever, and very friendly.

Current conversation:
{history}
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# Define request/response data models using Pydantic
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    conversation_history: List[Dict[str, str]]

# Load model and tokenizer when application starts
@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    try:
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info("Tokenizer loaded successfully")
        
        # Check available memory before loading model
        if DEVICE == "cuda":
            free_memory = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"GPU memory allocated before model load: {free_memory:.2f} GB")
        
        # Different loading strategy based on device
        if DEVICE == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # For CPU, use a simpler loading approach
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map=None,  # Don't use device_map on CPU
                low_cpu_mem_usage=True
            )
            model = model.to(DEVICE)
            
        logger.info("Model loaded successfully!")
    except Exception as e:
        # Comprehensive error logging
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def extract_ai_response(full_response: str) -> str:
    """Extract just the AI's response from the full model output."""
    # Try to find the AI response after "AI:"
    if "AI:" in full_response:
        parts = full_response.split("AI:")
        if len(parts) > 1:
            ai_response = parts[-1].strip()
            # Remove any trailing "Human:" parts
            if "Human:" in ai_response:
                ai_response = ai_response.split("Human:")[0].strip()
            return ai_response
    
    # If we can't find "AI:", try to extract the last part of the response
    # This handles cases where the model doesn't include the "AI:" prefix
    lines = full_response.strip().split("\n")
    if lines:
        return lines[-1].strip()
    
    return full_response.strip()

# Generate response using the loaded model
def generate_response(prompt_text: str) -> str:
    try:
        logger.info(f"Generating response for prompt: {prompt_text[:50]}...")
        inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        
        # Optimize generation parameters for faster responses
        outputs = model.generate(
            **inputs,
            max_length=256,  # Reduced from 512
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,  # Add top_p sampling
            top_k=50,   # Add top_k sampling
            repetition_penalty=1.2,  # Add repetition penalty
            no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
            early_stopping=True  # Enable early stopping
        )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the AI's response part
        response = extract_ai_response(full_response)
        logger.info(f"Generated response: {response[:50]}...")
        return response
    except Exception as e:
        # Error handling for generation process
        logger.error(f"Error generating response: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def parse_conversation_history(history_text: str) -> List[Dict[str, str]]:
    """Parse the conversation history into a structured format."""
    history_entries = []
    
    if not history_text:
        return history_entries
    
    # Split by newlines and process pairs
    parts = history_text.split("\n")
    current_user = ""
    current_ai = ""
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part.startswith("Human:"):
            # If we have a complete pair, add it to history
            if current_user and current_ai:
                history_entries.append({
                    "user": current_user,
                    "assistant": current_ai
                })
                current_user = ""
                current_ai = ""
            
            # Start a new user message
            current_user = part.replace("Human:", "").strip()
        elif part.startswith("AI:"):
            # Complete the AI message
            current_ai = part.replace("AI:", "").strip()
        else:
            # If it doesn't start with Human: or AI:, assume it's a continuation
            if current_user and not current_ai:
                current_user += " " + part
            elif current_ai:
                current_ai += " " + part
    
    # Add the last pair if complete
    if current_user and current_ai:
        history_entries.append({
            "user": current_user,
            "assistant": current_ai
        })
    
    return history_entries

# Chat endpoint that handles conversation
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")
        
        # Get conversation history from memory
        history = memory.load_memory_variables({})
        history_text = history.get("history", "")
        
        # Format the prompt with history
        formatted_prompt = prompt.format(
            history=history_text,
            input=request.message
        )
        
        # Generate response
        response = generate_response(formatted_prompt)
        
        # Store the conversation in memory
        memory.save_context({"input": request.message}, {"output": response})
        
        # Get updated conversation history
        updated_history = memory.load_memory_variables({})
        history_text = updated_history.get("history", "")
        
        # Parse history into structured format
        history_entries = parse_conversation_history(history_text)
        
        logger.info("Successfully processed chat request")
        return ChatResponse(
            response=response,
            conversation_history=history_entries
        )
    except Exception as e:
        # Error handling for chat endpoint
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application using uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 