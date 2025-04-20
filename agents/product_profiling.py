from agents.llm_service import query_google_llm
import json
import re

def profile_product(product_name):
    """Use LLM to generate a comprehensive product profile based on the product name"""
    if not product_name:
        return {"category": "unknown", "turnover": "unknown", "perishability": "unknown"}
    
    # Modified prompt for better compatibility with Gemini
    prompt = f"""
You are a supply chain expert. Analyze the following product and provide a detailed profile in JSON format:

Product: {product_name}

Create a JSON object with these properties:
- category: the product category (e.g., fragile, perishable, hazardous, general, electronics, etc.)
- turnover: expected inventory turnover rate (high, medium, low)
- perishability: level of perishability (high, medium, low, none)
- storage_requirements: special storage needs (temperature control, humidity control, etc.)
- handling_requirements: special handling needs (if any)
- typical_lead_time: typical lead time for procurement (in days or weeks)
- seasonality: whether the product has seasonal demand patterns (yes/no)

Return ONLY the JSON object without any additional text or markdown formatting.
"""
    
    try:
        # Get profile from Google Gemini LLM
        response = query_google_llm(prompt)
        
        # Try multiple approaches to extract valid JSON
        
        # First, try direct parsing
        try:
            profile = json.loads(response.strip())
            return profile
        except json.JSONDecodeError:
            pass
        
        # Next, try to extract JSON if it's embedded in text
        json_match = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                profile = json.loads(json_str)
                return profile
            except json.JSONDecodeError:
                pass
        
        # Try to extract JSON if it's in a code block
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if code_block_match:
            try:
                json_str = code_block_match.group(1)
                profile = json.loads(json_str)
                return profile
            except json.JSONDecodeError:
                pass
        
        # If we can't parse the response, fall back to basic profiling
        print(f"Could not parse LLM response as JSON: {response}")
        
    except Exception as e:
        print(f"Error in LLM product profiling: {e}")
    
    # Fallback to basic profiling if LLM fails
    if "ceramic" in product_name.lower() or "glass" in product_name.lower() or "porcelain" in product_name.lower():
        return {"category": "fragile", "turnover": "medium", "perishability": "none"}
    elif "food" in product_name.lower() or "fruit" in product_name.lower() or "vegetable" in product_name.lower():
        return {"category": "perishable", "turnover": "high", "perishability": "high"}
    elif "electronics" in product_name.lower() or "computer" in product_name.lower():
        return {"category": "electronics", "turnover": "medium", "perishability": "none"}
    else:
        return {"category": "general", "turnover": "low", "perishability": "none"}
