Architected Python solution leveraging Playwright and Gemini APIArchitected Python solution leveraging Playwright and Gemini APILet me read the relevant skill files first, then create a Python script for this task.
The user wants a Python script that:

Takes a URL and a query
Screenshots/captures the webpage as an image
Uses free GenAI to analyze the image and extract relevant information
Outputs results as markdown

Let me think about the tools:

For webpage screenshot: playwright or selenium with headless browser, or pyppeteer
For free GenAI image analysis: Use Claude API (which the user has access to via the Anthropic API in artifacts context), or Google's Gemini free tier, or use pytesseract for OCR
I'll go with Google Gemini since it has a generous free tier and is well-suited for image analysis, paired with playwright for reliable screenshots.

Writing the Python script...Show moreCheck available skillsLet me create a Python script for this task. I'll use:

playwright for webpage screenshots
Google's Gemini API (free tier) for image analysis - it's highly rated for accuracy
Output as markdown

Let me write the script.DoneHere's a well-structured Python script using Playwright for screenshots and Google Gemini (free tier, highly accurate for vision tasks) for analysis:Created a file, read a fileCreated a file, read a filePython script to screenshot a webpage and extract query-relevant info using Gemini visionwebpage_query_extractor.pyPresented fileDoneHow it works
The script has 3 clean stages:
① Screenshot — Playwright launches a headless Chromium browser, loads the page, scrolls to trigger lazy content, then saves a full-page PNG. Ad/tracker routes are blocked for a cleaner render.
② Resize (if needed) — Tall pages are resized proportionally to stay within Gemini's ~8000px / 4MB cap using Pillow.
③ Gemini Vision — The image + query are sent to gemini-1.5-flash (free tier, excellent at tables and structured extraction) with a strict system prompt that tells it to output only relevant content as Markdown — including proper | table formatting — while ignoring nav bars, footers, ads, etc.


# Setup
pip install playwright google-generativeai pillow
playwright install chromium

export GEMINI_API_KEY="your_key"   # free at https://aistudio.google.com/app/apikey

Why these tools?
ToolWhyPlaywrightFree, headless, handles JS-rendered pages, much more reliable than requests+BeautifulSoup for modern sitesGemini 1.5 FlashFree tier (15 RPM, 1M tokens/day), top-rated for vision+table accuracy, returns clean structured textPillowLightweight resize to stay within API limits

## Free Tier
pip install playwright groq ollama pillow
playwright install chromium

### Install Ollama from https://ollama.com, then pull a model:
`https://ollama.com/library`
ollama pull llama3.2-vision   # best accuracy (~6 GB RAM)
ollama pull llava             # lighter    (~4 GB RAM)
ollama pull moondream         # tiny       (~2 GB RAM)
ollama ps
```
The output shows the model, size, and crucially the **processor** column:
```
NAME                  ID      SIZE   PROCESSOR  UNTIL
llama3.2-vision:latest ...     6GB    100% GPU   ...   ✅ good
llama3.2-vision:latest ...     6GB    100% CPU   ...   ❌ no GPU

### NVIDIA
1. Install NVIDIA drivers (470+ recommended)
nvidia-smi   # confirm driver is working

2. Ollama on Linux auto-detects CUDA – no extra install needed
3. On Windows: use the .exe installer from https://ollama.com

### AMD
rocm-smi

### Apple Silicon
sudo powermetrics --samplers gpu_power -n 1


# Options
## Mini OCR + VL Inference Pipelines

Tools like Vision-CAIR / MiniGPT-4 align BLIP-2 vision encoders with an LLM to answer questions about images.
⚠️ Default versions typically need > 8 GB VRAM. But you can:

Quantise to 4-bit and CPU-offload the language model

Fuse with a small backbone LLM (e.g., 3B–7B Tiny) to run locally

## Llama 3.2 Vision with Markdown Extraction

What it is: Open-source vision language capability from Meta (Llama 3.2 supports images).

Use case: You can prompt it with both the image and a text query to get structured output like summaries, lists, bullet points, and Markdown.

Why it fits you:
• Lighter variants (e.g., Llama 3.2 1B / 3B) are much more feasible on 8 GB GPU than huge multimodal models.
• Models like Llama 3.2-11B-Vision can be used with CPU offload/quantisation if needed.

## Run Nanonets-OCR-s on the image → gets Markdown.
https://huggingface.co/nanonets/Nanonets-OCR-s

## PyVisionAI (vision LLM wrapper)

## BLIP-2 + small LLM quantised
https://huggingface.co/docs/transformers/en/model_doc/blip-2