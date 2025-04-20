import requests
import json
import time
import os
from dotenv import load_dotenv
from agents.llm_service import query_google_llm
from agents.knowledge_base import fetch_web_insight
import html
import re

# Load environment variables
load_dotenv()

# Google AI Studio API configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Use the standard Gemini Pro model which is more stable
GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp-01-21:generateContent"

def process_user_query(product_name, user_question):
    """
    Main interaction function that orchestrates the entire workflow:
    1. Get product profile
    2. Get sourcing advice
    3. Get web insights
    4. Combine all information and generate final response
    """
    try:
        # Import here to avoid circular imports
        from agents.product_profiling import profile_product
        from agents.sourcing import sourcing_advice
        
        # Step 1: Get product profile
        print(f"Getting product profile for: {product_name}")
        product_profile = profile_product(product_name)
        
        # Step 2: Get sourcing advice based on profile
        print(f"Getting sourcing advice for: {product_name}")
        sourcing_info = sourcing_advice(product_profile, product_name)
        
        # Step 3: Get web insights
        print(f"Getting web insights for: {user_question} about {product_name}")
        web_info = fetch_web_insight(f"{user_question} about {product_name}")
        
        # Step 4: Combine all information and generate final response
        print("Generating final response")
        final_prompt = f"""
You are a supply chain expert providing advice to a manager. 
Create a professional, well-structured response to the following question:

PRODUCT: {product_name}

USER QUESTION: {user_question}

PRODUCT PROFILE:
{json.dumps(product_profile, indent=2)}

SOURCING ADVICE:
{sourcing_info}

WEB INSIGHTS:
{web_info}

Format your response in a clear, professional manner with appropriate headings and bullet points.
Focus on directly answering the user's question while incorporating relevant information from the product profile, 
sourcing advice, and web insights.

IMPORTANT: Format your response using HTML tags for better readability:
- Use <h2> for main section headings
- Use <h3> for subsection headings
- Use <p> for paragraphs
- Use <ul> and <li> for bullet points
- Use <strong> for emphasis
- Use <br> for line breaks
- Include a proper salutation and closing
"""
        # Get final response from LLM
        raw_response = query_google_llm(final_prompt)
        
        # Ensure the response has proper HTML formatting
        final_response = format_response_with_html(raw_response, product_name, user_question)
        return final_response
    
    except Exception as e:
        print(f"Error in processing user query: {e}")
        import traceback
        traceback.print_exc()
        return f"<p>An error occurred while processing your request: {str(e)}</p>"

def format_response_with_html(response, product_name, question):
    """Format the response with proper HTML if not already formatted"""
    
    # Check if response already has HTML formatting
    if "<h2>" in response or "<p>" in response:
        # Already has HTML formatting, just ensure it's safe
        return response
    
    # Add basic HTML formatting
    formatted = "<h2>Supply Chain Analysis: " + html.escape(product_name) + "</h2>"
    
    # Add question section
    formatted += "<h3>Regarding: " + html.escape(question) + "</h3>"
    
    # Process the main content - split by newlines and format
    paragraphs = response.split('\n\n')
    for para in paragraphs:
        if para.strip():
            # Check if it's a heading (all caps or ends with colon)
            if para.isupper() or re.match(r'^[A-Z][^a-z]*:', para.strip()):
                formatted += f"<h3>{html.escape(para)}</h3>"
            # Check if it's a bullet point list
            elif para.strip().startswith('* ') or para.strip().startswith('- '):
                formatted += "<ul>"
                for bullet in re.split(r'\n\s*[\*\-]\s+', para):
                    if bullet.strip():
                        formatted += f"<li>{html.escape(bullet.strip())}</li>"
                formatted += "</ul>"
            else:
                formatted += f"<p>{html.escape(para)}</p>"
    
    return formatted

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

# Keep the existing mock response function
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
