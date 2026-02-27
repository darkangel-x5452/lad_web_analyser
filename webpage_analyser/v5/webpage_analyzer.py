"""
Webpage Image Analyzer using Free GenAI Tools
Captures webpage screenshots and analyzes them using vision models
"""

import os
import base64
from pathlib import Path
from io import BytesIO
import time
from typing import Optional, Literal

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Installing playwright...")
    os.system("pip install playwright --break-system-packages")
    os.system("playwright install chromium")
    from playwright.sync_api import sync_playwright

try:
    import anthropic
except ImportError:
    print("Installing anthropic...")
    os.system("pip install anthropic --break-system-packages")
    import anthropic

try:
    import google.generativeai as genai
except ImportError:
    print("Installing google-generativeai...")
    os.system("pip install google-generativeai --break-system-packages")
    import google.generativeai as genai


class WebpageAnalyzer:
    """Analyzes webpages using screenshots and vision AI models"""
    
    def __init__(self, model: Literal["claude", "gemini", "ollama"] = "claude"):
        """
        Initialize the analyzer
        
        Args:
            model: Which vision model to use ("claude", "gemini", "ollama")
        """
        self.model = model
        self.setup_model()
    
    def setup_model(self):
        """Setup the selected vision model"""
        if self.model == "claude":
            # Claude API - Excellent for vision, free tier available
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("\n⚠️  ANTHROPIC_API_KEY not set!")
                print("Get your free API key at: https://console.anthropic.com/")
                print("Then set it: export ANTHROPIC_API_KEY='your-key-here'")
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = anthropic.Anthropic(api_key=api_key)
            print("✓ Using Claude Sonnet 4 (Excellent accuracy)")
            
        elif self.model == "gemini":
            # Google Gemini - Free tier is very generous
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("\n⚠️  GOOGLE_API_KEY not set!")
                print("Get your free API key at: https://aistudio.google.com/app/apikey")
                print("Then set it: export GOOGLE_API_KEY='your-key-here'")
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✓ Using Gemini 2.0 Flash (Fast & accurate)")
            
        elif self.model == "ollama":
            # Ollama with LLaVA - Completely free, runs locally
            try:
                import ollama
            except ImportError:
                print("Installing ollama...")
                os.system("pip install ollama --break-system-packages")
                import ollama
            
            self.client = ollama
            print("✓ Using Ollama with LLaVA (Local & free)")
            print("   Make sure Ollama is running: ollama serve")
            print("   And LLaVA is pulled: ollama pull llava")
    
    def capture_screenshot(self, url: str, output_path: str = "screenshot.png", 
                          wait_time: int = 2, full_page: bool = True) -> str:
        """
        Capture a screenshot of a webpage
        
        Args:
            url: The webpage URL
            output_path: Where to save the screenshot
            wait_time: Time to wait for page to load (seconds)
            full_page: Capture full scrollable page
            
        Returns:
            Path to the screenshot file
        """
        print(f"\n📸 Capturing screenshot of: {url}")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})
            
            # Navigate to the page
            page.goto(url, wait_until='networkidle')
            time.sleep(wait_time)  # Additional wait for dynamic content
            
            # Take screenshot
            page.screenshot(path=output_path, full_page=full_page)
            browser.close()
        
        print(f"✓ Screenshot saved to: {output_path}")
        return output_path
    
    def analyze_with_claude(self, image_path: str, query: str) -> str:
        """Analyze image using Claude API"""
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        # Determine media type
        ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }
        media_type = media_type_map.get(ext, 'image/png')
        
        # Create the message
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"""Analyze this webpage screenshot and extract the information requested.

USER QUERY: {query}

INSTRUCTIONS:
1. Focus ONLY on the relevant information related to the query
2. Ignore headers, footers, navigation menus, ads, and other irrelevant content
3. Extract data accurately from tables, charts, or text
4. Format the output as clean, readable markdown
5. If the requested information is not visible, clearly state that

Output the extracted information in markdown format."""
                        }
                    ],
                }
            ],
        )
        
        return message.content[0].text
    
    def analyze_with_gemini(self, image_path: str, query: str) -> str:
        """Analyze image using Google Gemini"""
        from PIL import Image
        
        # Load image
        img = Image.open(image_path)
        
        prompt = f"""Analyze this webpage screenshot and extract the information requested.

USER QUERY: {query}

INSTRUCTIONS:
1. Focus ONLY on the relevant information related to the query
2. Ignore headers, footers, navigation menus, ads, and other irrelevant content
3. Extract data accurately from tables, charts, or text
4. Format the output as clean, readable markdown
5. If the requested information is not visible, clearly state that

Output the extracted information in markdown format."""
        
        response = self.client.generate_content([prompt, img])
        return response.text
    
    def analyze_with_ollama(self, image_path: str, query: str) -> str:
        """Analyze image using Ollama with LLaVA"""
        prompt = f"""Analyze this webpage screenshot and extract the information requested.

USER QUERY: {query}

INSTRUCTIONS:
1. Focus ONLY on the relevant information related to the query
2. Ignore headers, footers, navigation menus, ads, and other irrelevant content
3. Extract data accurately from tables, charts, or text
4. Format the output as clean, readable markdown
5. If the requested information is not visible, clearly state that

Output the extracted information in markdown format."""
        
        response = self.client.chat(
            model='llava',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]
        )
        
        return response['message']['content']
    
    def analyze_screenshot(self, image_path: str, query: str) -> str:
        """
        Analyze a screenshot using the selected vision model
        
        Args:
            image_path: Path to the screenshot
            query: What information to extract
            
        Returns:
            Markdown formatted analysis
        """
        print(f"\n🤖 Analyzing with {self.model.upper()}...")
        print(f"📝 Query: {query}")
        
        if self.model == "claude":
            result = self.analyze_with_claude(image_path, query)
        elif self.model == "gemini":
            result = self.analyze_with_gemini(image_path, query)
        elif self.model == "ollama":
            result = self.analyze_with_ollama(image_path, query)
        
        print("✓ Analysis complete!")
        return result
    
    def analyze_webpage(self, url: str, query: str, 
                       screenshot_path: str = "screenshot.png") -> str:
        """
        Complete workflow: capture screenshot and analyze
        
        Args:
            url: Webpage URL
            query: What information to extract
            screenshot_path: Where to save screenshot
            
        Returns:
            Markdown formatted analysis
        """
        # Capture screenshot
        self.capture_screenshot(url, screenshot_path)
        
        # Analyze
        result = self.analyze_screenshot(screenshot_path, query)
        
        return result


def main():
    """Example usage"""
    # Example: Analyze LA Lakers statistics
    
    print("=" * 60)
    print("WEBPAGE IMAGE ANALYZER - Free GenAI Tools")
    print("=" * 60)
    
    # Choose your model (all are free!)
    # "claude" - Most accurate, requires API key from console.anthropic.com
    # "gemini" - Fast & accurate, requires API key from aistudio.google.com
    # "ollama" - Runs locally, completely free, requires Ollama installed
    
    analyzer = WebpageAnalyzer(model="claude")  # Change to "gemini" or "ollama"
    
    # Example URL - NBA Lakers stats
    url = "https://www.nba.com/stats/team/1610612747"
    query = "Get me the LA Lakers team statistics"
    
    # Analyze
    result = analyzer.analyze_webpage(url, query)
    
    # Print result
    print("\n" + "=" * 60)
    print("EXTRACTED INFORMATION")
    print("=" * 60)
    print(result)
    
    # Save to file
    output_file = "analysis_output.md"
    with open(output_file, "w") as f:
        f.write(f"# Webpage Analysis\n\n")
        f.write(f"**URL:** {url}\n\n")
        f.write(f"**Query:** {query}\n\n")
        f.write(f"---\n\n")
        f.write(result)
    
    print(f"\n✓ Saved to: {output_file}")


if __name__ == "__main__":
    main()
