import os
from typing import List, Optional
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import httpx
from pydantic import BaseModel
import json
from prompt_templates import SENTIMENT_ANALYSIS_PROMPT, RESPONSE_GENERATION_PROMPT

app = FastAPI(title="Customer Feedback Analyzer")

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration for API
WEBUI_ENABLED = True  
WEBUI_BASE_URL = "https://chat.ivislabs.in/api"
API_KEY = "sk-51577380b8794da69ebd97b827d23458"  # Ensure this is securely stored
DEFAULT_MODEL = "gemma2:2b"

OLLAMA_ENABLED = True  
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_API_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api"

class AnalysisRequest(BaseModel):
    feedback: str
    insights: str = ""
    include_insights: bool = True
    generate_response: bool = True
    response_tone: Optional[str] = "empathetic"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_feedback(
    feedback: str = Form(...),
    insights: str = Form(""),
    include_insights: bool = Form(True),
    generate_response: bool = Form(True),
    response_tone: str = Form("empathetic"),
    model: str = Form(DEFAULT_MODEL)
):
    try:
        print(f"Insights Received: {insights}")

        # Sentiment Analysis
        analysis_prompt = SENTIMENT_ANALYSIS_PROMPT.format(feedback=feedback)
        sentiment_analysis = await run_llm_request(model, analysis_prompt)

        response_draft = ""
        if generate_response:
            formatted_insights = f'"{insights.strip()}"' if insights.strip() and include_insights else "No additional insights provided."
            response_prompt = RESPONSE_GENERATION_PROMPT.format(
                feedback=feedback,
                analysis=sentiment_analysis,
                tone=response_tone,
                insights=formatted_insights
            )

            print(f"Response Prompt: {response_prompt}")  
            response_draft = await run_llm_request(model, response_prompt)

        return {
            "feedback": feedback,
            "sentiment_analysis": sentiment_analysis,
            "response_draft": response_draft
        }

    except Exception as e:
        print(f"Error analyzing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing feedback: {str(e)}")

async def run_llm_request(model, prompt):
    if WEBUI_ENABLED:
        try:
            messages = [{"role": "user", "content": prompt}]
            request_payload = {"model": model, "messages": messages}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{WEBUI_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json=request_payload,
                    timeout=60.0
                )

                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        return choice.get("message", {}).get("content", choice.get("text", ""))
                    return result.get("response", "")
        except Exception as e:
            print(f"Open-webui API failed: {str(e)}")

    if OLLAMA_ENABLED:
        print("Falling back to direct Ollama API")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_API_URL}/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60.0
            )
            if response.status_code == 200:
                return response.json().get("response", "")

    raise HTTPException(status_code=500, detail="Failed to generate content from any available LLM API")

@app.get("/models")
async def get_models():
    try:
        if WEBUI_ENABLED:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{WEBUI_BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"})
                    if response.status_code == 200:
                        models_data = response.json()
                        return {"models": [model["id"] for model in models_data.get("data", [])]}
            except Exception as e:
                print(f"Error fetching models from open-webui API: {str(e)}")

        if OLLAMA_ENABLED:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{OLLAMA_API_URL}/tags")
                    if response.status_code == 200:
                        return {"models": [model["name"] for model in response.json().get("models", [])]}
            except Exception as e:
                print(f"Error fetching models from Ollama: {str(e)}")

        return {"models": [DEFAULT_MODEL, "gemma2:2b", "qwen2.5:0.5b", "deepseek-r1:1.5b"]}
    except Exception as e:
        print(f"Unexpected error in get_models: {str(e)}")
        return {"models": [DEFAULT_MODEL]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
