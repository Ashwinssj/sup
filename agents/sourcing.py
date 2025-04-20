from agents.llm_service import query_google_llm

def sourcing_advice(profile, product_name):
    """Use LLM to generate sourcing advice based on product profile and name"""
    if not profile or not product_name:
        return "Insufficient information to provide sourcing advice."
    
    # Convert profile to string representation for the prompt
    profile_str = "\n".join([f"- {key}: {value}" for key, value in profile.items()])
    
    # Modified prompt for better compatibility with Gemini
    prompt = f"""
You are a supply chain expert. Provide detailed sourcing advice for the following product:

Product: {product_name}

Product Profile:
{profile_str}

Please provide specific sourcing recommendations covering:
1. Supplier selection criteria
2. Geographic sourcing strategy (local, regional, global)
3. Supplier relationship management approach
4. Risk mitigation strategies
5. Optimal order quantities and frequency
6. Quality control considerations

Format your response as a clear, concise report that a supply chain manager can implement.
"""
    
    try:
        # Get advice from Google Gemini LLM
        response = query_google_llm(prompt)
        
        # Clean up response if needed
        response = response.strip()
        
        return response
    except Exception as e:
        print(f"Error in LLM sourcing advice: {e}")
    
    # Fallback to basic advice if LLM fails
    if profile.get("category") == "fragile":
        return "Use local or regional suppliers to reduce damage risk. Implement strict packaging requirements and quality control processes."
    elif profile.get("category") == "perishable":
        return "Source from local suppliers with proven cold chain capabilities. Establish frequent, smaller deliveries to maintain freshness."
    elif profile.get("category") == "electronics":
        return "Consider global sourcing with emphasis on quality certification. Implement robust counterfeit prevention measures."
    else:
        return "Standard global sourcing is suitable. Focus on cost optimization while maintaining quality standards."
