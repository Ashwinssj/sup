import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from agents.llm_service import query_google_llm
from agents.nlp_processor import analyze_text
from agents.deep_learning import deep_learning_forecast, deep_anomaly_detection
from agents.reinforcement_learning import optimize_inventory_policy, optimize_multi_product_inventory

class MultiAgentWorkflow:
    """
    Orchestrates multiple AI agents to solve complex supply chain problems.
    This class coordinates the workflow between different specialized AI agents,
    each focusing on a specific aspect of supply chain management.
    """
    
    def __init__(self):
        self.agents = {
            "forecasting": {
                "name": "Forecasting Agent",
                "description": "Predicts future demand using deep learning and statistical methods"
            },
            "optimization": {
                "name": "Optimization Agent",
                "description": "Optimizes inventory levels and order quantities using reinforcement learning"
            },
            "anomaly": {
                "name": "Anomaly Detection Agent",
                "description": "Identifies unusual patterns in supply chain data"
            },
            "nlp": {
                "name": "NLP Agent",
                "description": "Analyzes text data from suppliers, market reports, and customer feedback"
            },
            "coordinator": {
                "name": "Coordinator Agent",
                "description": "Orchestrates the workflow and combines insights from other agents"
            }
        }
        self.workflow_history = []
    
    def run_forecasting_agent(self, historical_data, target_col="orders", forecast_days=30):
        """
        Run the forecasting agent to predict future demand.
        
        Args:
            historical_data (list): Historical data for forecasting
            target_col (str): Column to forecast
            forecast_days (int): Number of days to forecast
            
        Returns:
            dict: Forecasting results and agent metadata
        """
        # First try deep learning forecast
        try:
            forecast = deep_learning_forecast(historical_data, target_col, forecast_days)
            method = "deep_learning"
        except Exception as e:
            print(f"Deep learning forecast failed: {e}. Falling back to statistical methods.")
            # Fall back to statistical methods
            from agents.advanced_analytics import advanced_demand_forecast
            forecast = advanced_demand_forecast(historical_data, forecast_days=forecast_days)
            method = "statistical"
        
        result = {
            "agent": "forecasting",
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "forecast": forecast,
            "confidence": self._calculate_forecast_confidence(forecast),
            "recommendations": self._generate_forecast_recommendations(forecast)
        }
        
        self.workflow_history.append({
            "agent": "forecasting",
            "action": "forecast",
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def _calculate_forecast_confidence(self, forecast):
        """
        Calculate confidence score for the forecast based on upper/lower bounds.
        
        Args:
            forecast (list): Forecast data with upper and lower bounds
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not forecast:
            return 0.5
        
        # Calculate average width of prediction intervals relative to predicted value
        total_width_ratio = 0
        for point in forecast:
            predicted = point.get("forecasted_demand", 0)
            if predicted > 0:
                width = point.get("upper_bound", predicted) - point.get("lower_bound", predicted)
                width_ratio = width / predicted
                total_width_ratio += width_ratio
        
        avg_width_ratio = total_width_ratio / len(forecast) if forecast else 1
        
        # Convert to confidence score (narrower intervals = higher confidence)
        confidence = max(0, min(1, 1 - (avg_width_ratio / 2)))
        return round(confidence, 2)
    
    def _generate_forecast_recommendations(self, forecast):
        """
        Generate recommendations based on forecast results.
        
        Args:
            forecast (list): Forecast data
            
        Returns:
            list: Recommendations based on forecast
        """
        if not forecast:
            return ["Insufficient data for recommendations"]
        
        recommendations = []
        
        # Calculate trend
        if len(forecast) >= 7:
            first_week = sum(point.get("forecasted_demand", 0) for point in forecast[:7])
            last_week = sum(point.get("forecasted_demand", 0) for point in forecast[-7:])
            
            if last_week > first_week * 1.2:
                recommendations.append("Strong upward trend detected. Consider increasing inventory levels and production capacity.")
            elif last_week < first_week * 0.8:
                recommendations.append("Downward trend detected. Consider reducing order quantities and running promotions to clear inventory.")
        
        # Check for seasonality
        if len(forecast) >= 14:
            weekly_totals = []
            for i in range(0, len(forecast) - 7, 7):
                weekly_total = sum(point.get("forecasted_demand", 0) for point in forecast[i:i+7])
                weekly_totals.append(weekly_total)
            
            if len(weekly_totals) >= 2:
                week_over_week_change = [(weekly_totals[i] - weekly_totals[i-1]) / weekly_totals[i-1] 
                                        for i in range(1, len(weekly_totals))]
                
                if any(change > 0.15 for change in week_over_week_change):
                    recommendations.append("Weekly seasonality detected. Adjust inventory and staffing levels accordingly.")
        
        # Add general recommendations
        recommendations.append("Regularly update forecast with new data to improve accuracy.")
        
        return recommendations
    
    def run_optimization_agent(self, products, historical_data=None, budget_constraint=None):
        """
        Run the optimization agent to optimize inventory levels and order quantities.
        
        Args:
            products (list): List of products to optimize
            historical_data (list): Historical data for optimization
            budget_constraint (float): Budget constraint for ordering
            
        Returns:
            dict: Optimization results and agent metadata
        """
        try:
            # If multiple products, use multi-product optimization
            if len(products) > 1 and budget_constraint:
                optimization_results = optimize_multi_product_inventory(products, budget_constraint)
                method = "multi_product_rl"
            else:
                # Single product optimization
                product = products[0] if products else {"inventory": 5000}
                optimization_results = optimize_inventory_policy(initial_inventory=product.get("inventory", 5000))
                method = "single_product_rl"
        except Exception as e:
            print(f"Reinforcement learning optimization failed: {e}. Falling back to heuristic methods.")
            # Fall back to heuristic methods
            from agents.analytics import optimize_order_quantities
            optimization_results = {"policy": {}, "performance": {}}
            for product in products:
                opt_qty = optimize_order_quantities([product])[0]
                optimization_results[product["name"]] = {"recommended_order": opt_qty}
            method = "heuristic"
        
        result = {
            "agent": "optimization",
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "optimization_results": optimization_results,
            "recommendations": self._generate_optimization_recommendations(optimization_results)
        }
        
        self.workflow_history.append({
            "agent": "optimization",
            "action": "optimize",
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def _generate_optimization_recommendations(self, optimization_results):
        """
        Generate recommendations based on optimization results.
        
        Args:
            optimization_results (dict): Optimization results
            
        Returns:
            list: Recommendations based on optimization
        """
        recommendations = []
        
        # Extract performance metrics if available
        performance = optimization_results.get("performance", {})
        
        if "average_inventory" in performance:
            avg_inventory = performance["average_inventory"]
            if avg_inventory > 3000:
                recommendations.append(f"Average inventory level ({avg_inventory:.0f} units) is high. Consider reducing safety stock.")
            elif avg_inventory < 1000:
                recommendations.append(f"Average inventory level ({avg_inventory:.0f} units) is low. Consider increasing safety stock to prevent stockouts.")
        
        if "stockout_days" in performance:
            stockout_days = performance["stockout_days"]
            if stockout_days > 0:
                recommendations.append(f"Stockouts detected on {stockout_days} days. Increase reorder points or safety stock.")
        
        # Add general recommendations
        recommendations.append("Implement the suggested ordering policy to optimize inventory costs.")
        recommendations.append("Review and adjust optimization parameters periodically based on changing business conditions.")
        
        return recommendations
    
    def run_anomaly_detection_agent(self, historical_data, sensitivity=1.0):
        """
        Run the anomaly detection agent to identify unusual patterns in supply chain data.
        
        Args:
            historical_data (list): Historical data to analyze
            sensitivity (float): Sensitivity level for anomaly detection
            
        Returns:
            dict: Anomaly detection results and agent metadata
        """
        try:
            # Try deep learning anomaly detection first
            anomalies = deep_anomaly_detection(historical_data, sensitivity)
            method = "deep_learning"
        except Exception as e:
            print(f"Deep learning anomaly detection failed: {e}. Falling back to statistical methods.")
            # Fall back to statistical methods
            from agents.advanced_analytics import detect_supply_chain_anomalies
            anomalies = detect_supply_chain_anomalies(historical_data, sensitivity=0.05)
            method = "statistical"
        
        result = {
            "agent": "anomaly",
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "anomalies": anomalies,
            "recommendations": self._generate_anomaly_recommendations(anomalies)
        }
        
        self.workflow_history.append({
            "agent": "anomaly",
            "action": "detect_anomalies",
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def _generate_anomaly_recommendations(self, anomalies):
        """
        Generate recommendations based on detected anomalies.
        
        Args:
            anomalies (list): Detected anomalies
            
        Returns:
            list: Recommendations based on anomalies
        """
        if not anomalies:
            return ["No anomalies detected. Supply chain is operating within normal parameters."]
        
        recommendations = []
        
        # Count anomalies by type
        metrics_count = {}
        for anomaly in anomalies:
            for metric in anomaly.get("anomalous_metrics", []):
                metrics_count[metric] = metrics_count.get(metric, 0) + 1
        
        # Generate recommendations based on most common anomalous metrics
        if metrics_count.get("orders", 0) > 2:
            recommendations.append("Multiple order anomalies detected. Review demand forecasting methods and market conditions.")
        
        if metrics_count.get("deliveries", 0) > 2:
            recommendations.append("Multiple delivery anomalies detected. Investigate supplier performance and logistics operations.")
        
        if metrics_count.get("inventory", 0) > 2:
            recommendations.append("Multiple inventory anomalies detected. Review inventory management policies and warehouse operations.")
        
        if metrics_count.get("costs", 0) > 2:
            recommendations.append("Multiple cost anomalies detected. Audit expenses and negotiate with suppliers.")
        
        # Add general recommendations
        recommendations.append(f"Investigate {len(anomalies)} detected anomalies to identify root causes and prevent recurrence.")
        recommendations.append("Consider adjusting anomaly detection sensitivity based on business needs.")
        
        return recommendations
    
    def run_nlp_agent(self, text_data, text_type):
        """
        Run the NLP agent to analyze text data from various sources.
        
        Args:
            text_data (str or list): Text data to analyze
            text_type (str): Type of text - 'supplier', 'market', or 'customer'
            
        Returns:
            dict: NLP analysis results and agent metadata
        """
        try:
            analysis = analyze_text(text_type, text_data)
            method = "transformer_nlp"
        except Exception as e:
            print(f"Advanced NLP analysis failed: {e}. Falling back to basic text analysis.")
            # Fall back to basic text analysis
            analysis = self._basic_text_analysis(text_data, text_type)
            method = "basic_nlp"
        
        result = {
            "agent": "nlp",
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "text_type": text_type,
            "analysis": analysis,
            "recommendations": self._generate_nlp_recommendations(analysis, text_type)
        }
        
        self.workflow_history.append({
            "agent": "nlp",
            "action": f"analyze_{text_type}_text",
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def _basic_text_analysis(self, text_data, text_type):
        """
        Perform basic text analysis when advanced NLP fails.
        
        Args:
            text_data (str or list): Text data to analyze
            text_type (str): Type of text
            
        Returns:
            dict: Basic analysis results
        """
        if text_type == "supplier":
            # Basic supplier communication analysis
            keywords = ["delay", "shortage", "issue", "problem", "increase", "decrease"]
            issues = []
            
            for keyword in keywords:
                if keyword in text_data.lower():
                    issues.append({"keyword": keyword, "context": "Found in text"})
            
            return {
                "sentiment": {"label": "NEUTRAL", "score": 0.5},
                "issues": issues,
                "recommendations": ["Review communication for potential supply chain issues"]
            }
            
        elif text_type == "market":
            # Basic market report analysis
            return {
                "summary": text_data[:200] + "..." if len(text_data) > 200 else text_data,
                "trends": [],
                "supply_chain_implications": []
            }
            
        elif text_type == "customer":
            # Basic customer feedback analysis
            if isinstance(text_data, str):
                text_data = [text_data]
                
            categories = {}
            for feedback in text_data:
                category = "Other"
                if "delivery" in feedback.lower() or "shipping" in feedback.lower():
                    category = "Delivery"
                elif "stock" in feedback.lower() or "available" in feedback.lower():
                    category = "Availability"
                
                if category not in categories:
                    categories[category] = {"count": 0, "examples": []}
                
                categories[category]["count"] += 1
                if len(categories[category]["examples"]) < 2:
                    categories[category]["examples"].append({"text": feedback, "sentiment": "NEUTRAL"})
            
            return {
                "total_feedback": len(text_data),
                "categories": categories
            }
        
        return {"error": "Invalid text type"}
    
    def _generate_nlp_recommendations(self, analysis, text_type):
        """
        Generate recommendations based on NLP analysis results.
        
        Args:
            analysis (dict): NLP analysis results
            text_type (str): Type of text analyzed
            
        Returns:
            list: Recommendations based on NLP analysis
        """
        recommendations = []
        
        if text_type == "supplier":
            # Recommendations based on supplier communication analysis
            if "recommendations" in analysis:
                return analysis["recommendations"]
                
            if "sentiment" in analysis and analysis["sentiment"].get("label") == "NEGATIVE":
                recommendations.append("Negative sentiment detected in supplier communication. Schedule follow-up discussion.")
                
            if "issues" in analysis and len(analysis["issues"]) > 0:
                issue_keywords = [issue["keyword"] for issue in analysis["issues"]]
                
                if any(keyword in ["delay", "late", "postpone"] for keyword in issue_keywords):
                    recommendations.append("Potential delivery delays mentioned. Update inventory plans and notify affected departments.")
                    
                if any(keyword in ["price", "cost", "increase"] for keyword in issue_keywords):
                    recommendations.append("Potential price changes mentioned. Review contract terms and budget impact.")
        
        elif text_type == "market":
            # Recommendations based on market report analysis
            if "supply_chain_implications" in analysis and len(analysis["supply_chain_implications"]) > 0:
                recommendations.append("Market report contains supply chain implications. Review for strategic planning.")
                
            if "trends" in analysis and len(analysis["trends"]) > 0:
                trend_keywords = [trend["keyword"] for trend in analysis.get("trends", [])]
                
                if any(keyword in ["shortage", "disruption", "constraint"] for keyword in trend_keywords):
                    recommendations.append("Market report indicates potential supply constraints. Develop contingency plans.")
                    
                if any(keyword in ["growth", "increase", "expansion"] for keyword in trend_keywords):
                    recommendations.append("Market growth trends identified. Prepare for potential demand increases.")
        
        elif text_type == "customer":
            # Recommendations based on customer feedback analysis
            categories = analysis.get("categories", {})
            
            if "Delivery" in categories and categories["Delivery"].get("count", 0) > 0:
                recommendations.append("Customer feedback mentions delivery issues. Review logistics operations and carrier performance.")
                
            if "Availability" in categories and categories["Availability"].get("count", 0) > 0:
                recommendations.append("Product availability issues mentioned in customer feedback. Review inventory management and forecasting.")
        
        # Add general recommendation if none specific
        if not recommendations:
            recommendations.append(f"Review {text_type} analysis for potential supply chain insights.")
        
        return recommendations
    
    def run_coordinator_agent(self, agent_results):
        """
        Run the coordinator agent to combine insights from other agents.
        
        Args:
            agent_results (dict): Results from other agents
            
        Returns:
            dict: Coordinated insights and recommendations
        """
        # Extract recommendations from all agents
        all_recommendations = []
        for agent_name, result in agent_results.items():
            if "recommendations" in result:
                for rec in result["recommendations"]:
                    all_recommendations.append({
                        "source_agent": agent_name,
                        "recommendation": rec
                    })
        
        # Prioritize recommendations
        prioritized_recommendations = self._prioritize_recommendations(all_recommendations)
        
        # Generate coordinated action plan
        action_plan = self._generate_action_plan(agent_results, prioritized_recommendations)
        
        result = {
            "agent": "coordinator",
            "timestamp": datetime.now().isoformat(),
            "insights": self._generate_cross_agent_insights(agent_results),
            "prioritized_recommendations": prioritized_recommendations,
            "action_plan": action_plan
        }
        
        self.workflow_history.append({
            "agent": "coordinator",
            "action": "coordinate",
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def _prioritize_recommendations(self, recommendations):
        """
        Prioritize recommendations based on urgency and impact.
        
        Args:
            recommendations (list): All recommendations from agents
            
        Returns:
            list: Prioritized recommendations
        """
        # Define priority keywords
        high_priority_keywords = ["critical", "urgent", "immediate", "stockout", "disruption"]
        medium_priority_keywords = ["review", "consider", "investigate", "potential", "monitor"]
        
        # Assign priority scores
        prioritized = []
        for rec in recommendations:
            priority = "low"
            text = rec["recommendation"].lower()
            
            if any(keyword in text for keyword in high_priority_keywords):
                priority = "high"
            elif any(keyword in text for keyword in medium_priority_keywords):
                priority = "medium"
            
            prioritized.append({
                "source_agent": rec["source_agent"],
                "recommendation": rec["recommendation"],
                "priority": priority
            })
        
        # Sort by priority
        return sorted(prioritized, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]])
    
    def _generate_cross_agent_insights(self, agent_results):
        """
        Generate insights by combining results from multiple agents.
        
        Args:
            agent_results (dict): Results from all agents
            
        Returns:
            list: Cross-agent insights
        """
        insights = []
        
        # Check for forecast and anomaly correlation
        if "forecasting" in agent_results and "anomaly" in agent_results:
            forecast = agent_results["forecasting"].get("forecast", [])
            anomalies = agent_results["anomaly"].get("anomalies", [])
            
            if forecast and anomalies:
                # Check if anomalies might affect forecast
                anomaly_dates = [a.get("date") for a in anomalies]
                recent_anomalies = [a for a in anomalies if a.get("date") in anomaly_dates[-30:]] if anomaly_dates else []
                
                if recent_anomalies:
                    insights.append("Recent anomalies detected may affect forecast accuracy. Consider adjusting forecast based on anomaly analysis.")
        
        # Check for optimization and forecast alignment
        if "optimization" in agent_results and "forecasting" in agent_results:
            forecast = agent_results["forecasting"].get("forecast", [])
            optimization = agent_results["optimization"].get("optimization_results", {})
            
            if forecast and optimization:
                insights.append("Align inventory optimization with latest demand forecast for best results.")
        
        # Check for NLP insights that might affect other areas
        if "nlp" in agent_results:
            nlp_result = agent_results["nlp"]
            text_type = nlp_result.get("text_type")
            analysis = nlp_result.get("analysis", {})
            
            if text_type == "supplier" and "issues" in analysis and len(analysis["issues"]) > 0:
                insights.append("Supplier communication issues detected may require adjustments to inventory and sourcing strategies.")
            
            if text_type == "market" and "trends" in analysis and len(analysis.get("trends", [])) > 0:
                insights.append("Market trends identified should be incorporated into demand forecasting and inventory planning.")
        
        # Add general insights if none specific
        if not insights:
            insights.append("Coordinate actions across all supply chain functions for optimal performance.")
        
        return insights
    
    def _generate_action_plan(self, agent_results, prioritized_recommendations):
        """
        Generate a coordinated action plan based on all agent results.
        
        Args:
            agent_results (dict): Results from all agents
            prioritized_recommendations (list): Prioritized recommendations
            
        Returns:
            dict: Structured action plan
        """
        # Group actions by timeframe
        immediate_actions = []
        short_term_actions = []
        long_term_actions = []
        
        # Assign actions based on priority
        for rec in prioritized_recommendations:
            action = rec["recommendation"]
            
            if rec["priority"] == "high":
                immediate_actions.append(action)
            elif rec["priority"] == "medium":
                short_term_actions.append(action)
            else:
                long_term_actions.append(action)
        
        # Ensure we have at least one action in each timeframe
        if not immediate_actions and short_term_actions:
            immediate_actions.append(short_term_actions.pop(0))
        
        if not short_term_actions and long_term_actions:
            short_term_actions.append(long_term_actions.pop(0))
        
        # Create action plan structure
        action_plan = {
            "immediate_actions": immediate_actions[:3],  # Top 3 immediate actions
            "short_term_actions": short_term_actions[:5],  # Top 5 short-term actions
            "long_term_actions": long_term_actions[:5],  # Top 5 long-term actions
            "monitoring_metrics": self._suggest_monitoring_metrics(agent_results)
        }
        
        return action_plan
    
    def _suggest_monitoring_metrics(self, agent_results):
        """
        Suggest metrics to monitor based on agent results.
        
        Args:
            agent_results (dict): Results from all agents
            
        Returns:
            list: Suggested monitoring metrics
        """
        metrics = [
            "Forecast Accuracy (MAPE)",
            "Inventory Turnover Rate",
            "Order Fill Rate",
            "Lead Time Variability"
        ]
        
        # Add specific metrics based on agent results
        if "anomaly" in agent_results and agent_results["anomaly"].get("anomalies"):
            anomalous_metrics = set()
            for anomaly in agent_results["anomaly"].get("anomalies", []):
                anomalous_metrics.update(anomaly.get("anomalous_metrics", []))
            
            for metric in anomalous_metrics:
                if metric == "orders":
                    metrics.append("Daily Order Volume Variance")
                elif metric == "deliveries":
                    metrics.append("Supplier Delivery Performance")
                elif metric == "inventory":
                    metrics.append("Days of Supply")
                elif metric == "costs":
                    metrics.append("Total Supply Chain Cost as % of Revenue")
        
        return metrics

# Example usage function
def run_multi_agent_workflow(historical_data, products, text_data=None):
    """
    Run a complete multi-agent workflow for supply chain management.
    
    Args:
        historical_data (list): Historical supply chain data
        products (list): List of products to analyze
        text_data (dict): Optional text data for NLP analysis
        
    Returns:
        dict: Comprehensive results from all agents and coordinator
    """
    workflow = MultiAgentWorkflow()
    results = {}
    
    # Run forecasting agent
    results["forecasting"] = workflow.run_forecasting_agent(historical_data)
    
    # Run optimization agent
    results["optimization"] = workflow.run_optimization_agent(products, historical_data)
    
    # Run anomaly detection agent
    results["anomaly"] = workflow.run_anomaly_detection_agent(historical_data)
    
    # Run NLP agent if text data provided
    if text_data and "type" in text_data and "content" in text_data:
        results["nlp"] = workflow.run_nlp_agent(text_data["content"], text_data["type"])
    
    # Run coordinator agent to combine insights
    results["coordinator"] = workflow.run_coordinator_agent(results)
    
    return results