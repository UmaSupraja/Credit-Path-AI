# Add near your other imports at the top of main.py
import os
import httpx
import asyncio
from fastapi import Body

# ---- Chat proxy endpoint: /api/chat ----
# This endpoint expects JSON: { "prompt": "user text" }
# It forwards the prompt to Google Generative Language (Gemini) using a server-side key.
# Set LLM_API_KEY in Vercel environment variables to your Google API key.
@app.post("/api/chat")
async def chat_proxy(payload: dict = Body(...)):
    """
    Proxy POST /api/chat
    Request body: { "prompt": "<text>" }
    Response: { "reply": "<assistant text>" }
    """
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in request body.")

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        logger.error("LLM_API_KEY environment variable not set.")
        raise HTTPException(status_code=500, detail="Server misconfiguration: LLM API key missing.")

    # Model name (you can override with env var if you want)
    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-preview-09-2025")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    system_prompt = (
        "You are an expert AI Credit Coach for the CREDIT PATH-AI application. "
        "Answer user questions about credit scores, loans, debt, and financial health. "
        "Be helpful, encouraging, concise, and avoid off-topic responses."
    )

    request_body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": prompt}]}],
        # Optionally tune parameters here (safety/temperature/length) if supported by the model
    }

    # Do a small exponential-backoff retry loop for transient errors (rate limits / 5xx)
    backoff = 1.0
    max_attempts = 3

    async with httpx.AsyncClient(timeout=30.0) as client:
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.post(url, json=request_body)
                if resp.status_code == 200:
                    result = resp.json()
                    candidate = None
                    # safe extraction of text
                    try:
                        candidate = result.get("candidates", [None])[0]
                        assistant_text = candidate.get("content", {}).get("parts", [None])[0].get("text")
                    except Exception:
                        assistant_text = None

                    if assistant_text:
                        return {"reply": assistant_text}
                    else:
                        # fallback: try other shapes
                        # Combine possible textual fields into reply if present
                        text_candidates = []
                        # try to collect from nested fields if different response shape
                        if isinstance(result.get("candidates"), list):
                            for c in result["candidates"]:
                                if isinstance(c, dict):
                                    for part in (c.get("content", {}).get("parts") or []):
                                        if isinstance(part, dict) and part.get("text"):
                                            text_candidates.append(part.get("text"))
                        if text_candidates:
                            return {"reply": "\n\n".join(text_candidates)}
                        # no textual result found
                        logger.error(f"No text candidate in LLM response: {result}")
                        raise HTTPException(status_code=502, detail="LLM returned an unexpected response shape.")
                elif resp.status_code in (429, 500, 502, 503, 504):
                    # transient — retry
                    last_err = f"LLM transient error {resp.status_code}: {await resp.text()}"
                    logger.warning(f"chat_proxy attempt {attempt} transient error: {last_err}")
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                else:
                    # client error — do not retry
                    err_body = resp.text
                    logger.error(f"LLM client error {resp.status_code}: {err_body}")
                    raise HTTPException(status_code=502, detail=f"LLM API error: {resp.status_code}")
            except httpx.RequestError as re:
                last_err = str(re)
                logger.warning(f"chat_proxy attempt {attempt} request error: {last_err}")
                await asyncio.sleep(backoff)
                backoff *= 2.0
                continue

        # If we get here, all retries failed
        logger.error(f"chat_proxy failed after {max_attempts} attempts. Last error: {last_err}")
        raise HTTPException(status_code=502, detail="Failed to get response from LLM provider. Please try again later.")
