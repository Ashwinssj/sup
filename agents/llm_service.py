import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google AI Studio API configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Use the standard Gemini Pro model which is more stable
GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp-01-21:generateContent"

def query_google_llm(prompt):
    try:
        formatted_prompt = prompt.strip()
        if not formatted_prompt:
            return "Error: Empty prompt provided"
            
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": formatted_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GOOGLE_API_KEY
        }
        
        try:
            # Connect to Google AI API
            response = requests.post(
                GOOGLE_API_URL,
                json=payload,
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                response_data = response.json()
                # Print response structure for debugging
                print(f"Response structure: {json.dumps(response_data, indent=2)[:200]}...")
                
                # Extract text from the response with improved error handling
                if "candidates" in response_data and len(response_data["candidates"]) > 0:
                    candidate = response_data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if parts and len(parts) > 0:
                            # Try different ways to extract text
                            if "text" in parts[0]:
                                return parts[0]["text"]
                            elif isinstance(parts[0], str):
                                return parts[0]
                            elif isinstance(parts[0], dict):
                                # Return any available field that might contain text
                                for key, value in parts[0].items():
                                    if isinstance(value, str) and value:
                                        return value
                
                # If we get here, we couldn't extract text using standard methods
                print(f"Could not extract text from response: {json.dumps(response_data)[:500]}")
                return "No valid response content received from Google AI. Please check API configuration."
            else:
                print(f"Google AI API error: {response.status_code}")
                print(f"Response: {response.text}")
                return f"API Error ({response.status_code}): {response.text[:100]}..."
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"Could not connect to Google AI service: {e}")
            return f"Connection error: {str(e)}"
            
        # Fallback: Generate a simple mock response based on the prompt
        return generate_mock_response(formatted_prompt)
            
    except Exception as e:
        print(f"LLM query error: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        return f"An error occurred while processing your request: {str(e)}"

# Generate a simple mock response when LLM API is unavailable
def generate_mock_response(prompt):
    """Generate a simple response when LLM API is unavailable"""
    # Extract product and question from the prompt
    product = ""
    question = ""
    
    lines = prompt.split('\n')
    for line in lines:
        if "product:" in line.lower():
            product = line.split(':', 1)[1].strip()
        elif "question:" in line.lower():
            question = line.split(':', 1)[1].strip()
    
    # Generate a simple response
    if "delivery" in question.lower() or "shipping" in question.lower():
        return f"For {product}, standard delivery times are 3-5 business days. Express shipping options are available for priority orders."
    elif "inventory" in question.lower() or "stock" in question.lower():
        return f"Current inventory levels for {product} are maintained at optimal levels with our just-in-time supply chain system. We recommend regular monitoring and setting up automated reorder points."
    elif "supplier" in question.lower() or "vendor" in question.lower():
        return f"For {product}, we recommend diversifying your supplier base to mitigate risks. Consider at least 2-3 qualified suppliers with different geographical locations."
    else:
        return f"Based on supply chain best practices for {product}, I recommend implementing a balanced approach that considers cost, quality, and delivery time. Regular supplier evaluations and maintaining safety stock levels are key to successful management."