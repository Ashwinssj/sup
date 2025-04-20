import requests
import os
from dotenv import load_dotenv
import html

# Load environment variables
load_dotenv()

# Get API key from environment variable or use the hardcoded one as fallback
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")

def fetch_web_insight(query):
    """Fetch insights from the web using Tavily API"""
    try:
        print(f"Fetching web insights for: {query}")
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
        payload = {
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "max_results": 5
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()  # Raises exception for 4XX/5XX errors
        
        result = response.json()
        
        # Extract answer and sources
        answer = result.get("answer", "")
        sources = result.get("results", [])
        
        # Format the response with sources using HTML for better display
        formatted_response = answer.replace("\n", "<br>")
        
        if sources and len(sources) > 0:
            formatted_response += "<br><br><strong>Sources:</strong><ul>"
            for i, source in enumerate(sources[:3], 1):  # Include up to 3 sources
                title = html.escape(source.get("title", "Untitled"))
                url = source.get("url", "")
                image = source.get("image_url", "")
                
                formatted_response += f'<li><a href="{url}" target="_blank">{title}</a>'
                
                # Add image if available
                if image and image.startswith(('http://', 'https://')):
                    formatted_response += f'<br><img src="{image}" alt="{title}" style="max-width:200px; margin:10px 0;">'
                
                formatted_response += '</li>'
            
            formatted_response += "</ul>"
        
        return formatted_response
    
    except requests.exceptions.RequestException as e:
        print(f"Tavily API request error: {e}")
        return f"Unable to fetch web insights: {str(e)}"
    except Exception as e:
        print(f"Knowledge base error: {e}")
        return "No web insight available. Please try a different query or check your internet connection."