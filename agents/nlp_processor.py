import pandas as pd
import numpy as np
from transformers import pipeline
import re
from datetime import datetime

class NLPProcessor:
    """
    Natural Language Processing capabilities for supply chain management.
    This class provides text analysis functions for processing supplier communications,
    customer feedback, and market reports to extract actionable insights.
    """
    
    def __init__(self):
        # Initialize sentiment analysis pipeline
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.text_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
            self.ner_pipeline = pipeline("ner", grouped_entities=True)
            self.summarizer = pipeline("summarization")
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading NLP models: {e}")
            self.model_loaded = False
    
    def analyze_supplier_communication(self, text):
        """
        Analyze supplier communications to identify potential issues or opportunities.
        
        Args:
            text (str): The text of the supplier communication
            
        Returns:
            dict: Analysis results including sentiment, key issues, and recommendations
        """
        if not self.model_loaded or not text:
            return {"error": "NLP model not loaded or empty text provided"}
        
        # Analyze sentiment
        try:
            sentiment = self.sentiment_analyzer(text)[0]
        except Exception:
            sentiment = {"label": "NEUTRAL", "score": 0.5}
        
        # Extract entities (company names, locations, etc.)
        try:
            entities = self.ner_pipeline(text)
        except Exception:
            entities = []
        
        # Extract dates using regex
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}|\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{2,4})\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        
        # Extract potential issues
        issue_keywords = [
            "delay", "shortage", "problem", "issue", "concern", "disruption", 
            "backorder", "stockout", "quality", "defect", "damage", "late", 
            "cancel", "increase", "decrease", "change"
        ]
        
        issues = []
        for keyword in issue_keywords:
            if re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE):
                # Find the sentence containing the keyword
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sentence in sentences:
                    if re.search(r'\b' + keyword + r'\b', sentence, re.IGNORECASE):
                        issues.append({"keyword": keyword, "context": sentence.strip()})
        
        # Generate recommendations based on sentiment and issues
        recommendations = []
        if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.7:
            recommendations.append("Urgent attention required - highly negative communication")
        
        if any(issue["keyword"] in ["delay", "late", "backorder", "stockout"] for issue in issues):
            recommendations.append("Potential supply disruption - review inventory levels and alternative suppliers")
        
        if any(issue["keyword"] in ["quality", "defect", "damage"] for issue in issues):
            recommendations.append("Quality issues reported - initiate quality review process")
        
        if any(issue["keyword"] in ["increase", "change"] for issue in issues):
            recommendations.append("Pricing or terms changes mentioned - review contract and cost implications")
        
        return {
            "sentiment": {
                "label": sentiment["label"],
                "score": round(sentiment["score"], 2)
            },
            "entities": [{
                "entity": e["entity_group"],
                "text": e["word"],
                "confidence": round(e["score"], 2)
            } for e in entities if "entity_group" in e and "word" in e and "score" in e],
            "dates": dates,
            "issues": issues,
            "recommendations": recommendations
        }
    
    def analyze_market_report(self, text, max_length=150):
        """
        Analyze market reports to extract key insights relevant to supply chain.
        
        Args:
            text (str): The text of the market report
            max_length (int): Maximum length of the summary
            
        Returns:
            dict: Analysis results including summary, key trends, and supply chain implications
        """
        if not self.model_loaded or not text:
            return {"error": "NLP model not loaded or empty text provided"}
        
        # Generate summary
        try:
            # Ensure text is not too long for the model
            max_input_length = 1024  # Most models have limits
            if len(text) > max_input_length:
                text_for_summary = text[:max_input_length]
            else:
                text_for_summary = text
                
            summary = self.summarizer(text_for_summary, max_length=max_length, min_length=30, do_sample=False)
            summary_text = summary[0]["summary_text"]
        except Exception as e:
            print(f"Summarization error: {e}")
            # Fallback to simple extraction
            sentences = re.split(r'(?<=[.!?])\s+', text)
            summary_text = " ".join(sentences[:3]) if len(sentences) > 3 else text
        
        # Extract trends and patterns
        trend_keywords = [
            "increase", "decrease", "growth", "decline", "rise", "fall", "trend", 
            "forecast", "predict", "expect", "project", "estimate", "outlook",
            "shortage", "surplus", "demand", "supply", "market", "price", "cost",
            "inflation", "deflation", "recession", "recovery", "expansion"
        ]
        
        trends = []
        for keyword in trend_keywords:
            if re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE):
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sentence in sentences:
                    if re.search(r'\b' + keyword + r'\b', sentence, re.IGNORECASE):
                        trends.append({"keyword": keyword, "context": sentence.strip()})
        
        # Extract supply chain implications
        sc_keywords = [
            "supply chain", "logistics", "inventory", "warehouse", "transportation", 
            "shipping", "delivery", "procurement", "supplier", "manufacturer", 
            "distribution", "lead time", "stockout", "backorder", "production"
        ]
        
        implications = []
        for keyword in sc_keywords:
            if re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE):
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sentence in sentences:
                    if re.search(r'\b' + keyword + r'\b', sentence, re.IGNORECASE):
                        implications.append({"keyword": keyword, "context": sentence.strip()})
        
        return {
            "summary": summary_text,
            "trends": trends[:5],  # Limit to top 5 trends
            "supply_chain_implications": implications[:5],  # Limit to top 5 implications
            "analysis_date": datetime.now().strftime("%Y-%m-%d")
        }
    
    def classify_customer_feedback(self, feedback_list):
        """
        Classify customer feedback related to product availability and delivery.
        
        Args:
            feedback_list (list): List of customer feedback strings
            
        Returns:
            dict: Classification results with categories and sentiment
        """
        if not self.model_loaded or not feedback_list:
            return {"error": "NLP model not loaded or empty feedback provided"}
        
        results = []
        
        for feedback in feedback_list:
            if not feedback or not isinstance(feedback, str):
                continue
                
            # Analyze sentiment
            try:
                sentiment = self.sentiment_analyzer(feedback)[0]
            except Exception:
                sentiment = {"label": "NEUTRAL", "score": 0.5}
            
            # Classify feedback category
            category = "Other"
            if re.search(r'\b(deliver|delivery|shipping|shipment|arrival|late|delay)\b', feedback, re.IGNORECASE):
                category = "Delivery"
            elif re.search(r'\b(stock|inventory|available|availability|out of stock|in stock|backorder)\b', feedback, re.IGNORECASE):
                category = "Availability"
            elif re.search(r'\b(quality|condition|damage|broken|defect|issue|problem)\b', feedback, re.IGNORECASE):
                category = "Quality"
            elif re.search(r'\b(price|cost|expensive|cheap|value|worth)\b', feedback, re.IGNORECASE):
                category = "Price"
            
            # Extract key phrases (simple approach)
            words = feedback.split()
            key_phrase = " ".join(words[:min(10, len(words))]) + "..." if len(words) > 10 else feedback
            
            results.append({
                "feedback": feedback[:100] + "..." if len(feedback) > 100 else feedback,
                "category": category,
                "sentiment": sentiment["label"],
                "sentiment_score": round(sentiment["score"], 2),
                "key_phrase": key_phrase
            })
        
        # Aggregate results
        categories = {}
        for result in results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {
                    "count": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "examples": []
                }
            
            categories[cat]["count"] += 1
            sentiment = result["sentiment"].lower()
            if sentiment in ["positive", "negative", "neutral"]:
                categories[cat][sentiment] += 1
            
            if len(categories[cat]["examples"]) < 3:  # Limit to 3 examples per category
                categories[cat]["examples"].append({
                    "text": result["feedback"],
                    "sentiment": result["sentiment"]
                })
        
        return {
            "total_feedback": len(results),
            "categories": categories,
            "analysis_date": datetime.now().strftime("%Y-%m-%d")
        }

# Example usage
def analyze_text(text_type, content):
    """
    Analyze text using the appropriate NLP method based on text type.
    
    Args:
        text_type (str): Type of text - 'supplier', 'market', or 'customer'
        content (str or list): Text content to analyze
        
    Returns:
        dict: Analysis results
    """
    processor = NLPProcessor()
    
    if text_type == "supplier":
        return processor.analyze_supplier_communication(content)
    elif text_type == "market":
        return processor.analyze_market_report(content)
    elif text_type == "customer":
        if isinstance(content, str):
            content = [content]  # Convert to list
        return processor.classify_customer_feedback(content)
    else:
        return {"error": "Invalid text type. Use 'supplier', 'market', or 'customer'"}