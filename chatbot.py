import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from peft import PeftModel, PeftConfig
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import HuggingFacePipeline
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import gc
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable LangChain caching
set_llm_cache(InMemoryCache())

app = FastAPI(title="Qwen Inference API")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

# Model initialization
try:
    peft_model_id = "QwenModel"  
    model = AutoModelForCausalLM.from_pretrained(peft_model_id, torch_dtype=torch.bfloat16, device_map='auto')
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}", exc_info=True)
    raise RuntimeError(f"Model loading failed: {str(e)}") from e

# Pipeline configuration
generation_config = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True
}

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_config
)

llm = HuggingFacePipeline(pipeline=pipe)

# LangChain setup
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Adam, a helpful financial assistant"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

memory = ConversationBufferWindowMemory(k=5, return_messages=True)
chain = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

# Memory profiling middleware
@app.middleware("http")
async def memory_profile(request: Request, call_next):
    try:
        mem_before = torch.cuda.memory_allocated() / 1e9
        response = await call_next(request)
        mem_after = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Memory delta: {mem_after - mem_before:.2f}GB")
        return response
    except Exception as e:
        logger.error(f"Middleware error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error in middleware"}
        )

# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Cleanup function
def cleanup_resources():
    torch.cuda.empty_cache()
    gc.collect()

# Startup event

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for model management"""
    # Startup logic
    logger.info("Warming up model...")
    try:
        pipe("Warmup", max_new_tokens=1)
        torch.cuda.empty_cache()
        logger.info("Model warmup complete")
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}", exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Shutdown logic (optional)
    logger.info("Cleaning up model resources...")
    torch.cuda.empty_cache()


# Endpoints
@app.get("/health")
def health_check():
    """Endpoint for service health verification"""
    return {
        "status": "healthy",
        "gpu_mem_allocated": f"{torch.cuda.memory_allocated()/1e9:.2f}GB",
        "gpu_mem_reserved": f"{torch.cuda.max_memory_reserved()/1e9:.2f}GB",
        "conversation_history_length": len(memory.buffer)
    }

@app.post("/validate")
async def validate_model(input: str = "Test prompt"):
    """Endpoint to validate model functionality"""
    try:
        test_output = pipe(input, max_new_tokens=10)[0]['generated_text']
        return {
            "valid": bool(test_output),
            "test_output": test_output
        }
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Model validation failed: {str(e)}")

@app.post("/chat")
async def chat_completion(
    request: PromptRequest,
    background_tasks: BackgroundTasks
):
    """Main chat endpoint"""
    try:
        # Input validation
        if not request.prompt.strip():
            raise HTTPException(400, "Empty prompt")

        # Model inference
        response = chain.invoke({ "input": str(request.prompt)})
        
        # Clean up resources in background
        background_tasks.add_task(cleanup_resources)
        
        # Process response
        cleaned_response = response['response'].split('Assistant:')[-1].strip()
        
        return {
            "response": cleaned_response,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Chat completion failed: {str(e)}")