from flask import Flask, render_template, request, jsonify, request
from agents.llm_agent import process_user_query
import json
import os
import random
from datetime import datetime, timedelta
from agents.analytics import analyze_inventory_levels, forecast_demand, calculate_risk_score, optimize_order_quantities
from agents.advanced_analytics import advanced_demand_forecast, detect_supply_chain_anomalies, optimize_inventory_allocation
from agents.deep_learning import deep_learning_forecast, deep_anomaly_detection
from agents.reinforcement_learning import optimize_inventory_policy, optimize_multi_product_inventory
from agents.nlp_processor import analyze_text
from agents.multi_agent_workflow import run_multi_agent_workflow

app = Flask(__name__, template_folder='templates')

# Sample data for demonstration purposes
SAMPLE_SUPPLIERS = [
    {"id": 1, "name": "Global Electronics", "location": {"lat": 37.7749, "lng": -122.4194}, "reliability": 0.92, "products": ["Electronics", "Computers"]},
    {"id": 2, "name": "Fresh Produce Co.", "location": {"lat": 40.7128, "lng": -74.0060}, "reliability": 0.88, "products": ["Food", "Perishables"]},
    {"id": 3, "name": "Industrial Materials", "location": {"lat": 34.0522, "lng": -118.2437}, "reliability": 0.95, "products": ["Raw Materials", "Chemicals"]},
    {"id": 4, "name": "Asian Manufacturing", "location": {"lat": 31.2304, "lng": 121.4737}, "reliability": 0.90, "products": ["Electronics", "Textiles"]},
    {"id": 5, "name": "European Precision", "location": {"lat": 48.8566, "lng": 2.3522}, "reliability": 0.94, "products": ["Machinery", "Automotive Parts"]}
]

SAMPLE_WAREHOUSES = [
    {"id": 1, "name": "West Coast Distribution", "location": {"lat": 37.3382, "lng": -121.8863}, "capacity": 10000, "utilization": 0.75},
    {"id": 2, "name": "East Coast Hub", "location": {"lat": 39.9526, "lng": -75.1652}, "capacity": 15000, "utilization": 0.82},
    {"id": 3, "name": "Central Storage", "location": {"lat": 41.8781, "lng": -87.6298}, "capacity": 12000, "utilization": 0.68},
    {"id": 4, "name": "Southern Facility", "location": {"lat": 29.7604, "lng": -95.3698}, "capacity": 8000, "utilization": 0.91}
]

SAMPLE_PRODUCTS = [
    {"id": 1, "name": "Laptop", "category": "Electronics", "inventory": 1250, "demand": "High", "leadTime": "2-3 weeks"},
    {"id": 2, "name": "Smartphone", "category": "Electronics", "inventory": 3000, "demand": "High", "leadTime": "1-2 weeks"},
    {"id": 3, "name": "Fresh Apples", "category": "Perishables", "inventory": 5000, "demand": "Medium", "leadTime": "3-5 days"},
    {"id": 4, "name": "Steel Sheets", "category": "Raw Materials", "inventory": 10000, "demand": "Low", "leadTime": "4-6 weeks"},
    {"id": 5, "name": "Cotton Fabric", "category": "Textiles", "inventory": 7500, "demand": "Medium", "leadTime": "2-4 weeks"}
]

# Generate sample historical data
def generate_historical_data(days=90):
    data = []
    start_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        data.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "orders": random.randint(50, 200),
            "deliveries": random.randint(40, 180),
            "inventory": random.randint(1000, 5000),
            "costs": random.randint(5000, 15000)
        })
    
    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    response = None
    error = None
    
    if request.method == 'POST':
        try:
            product = request.form['product']
            question = request.form['question']
            
            # Use the centralized process_user_query function
            response = process_user_query(product, question)
            
            if not response or "API error" in response:
                error = "Sorry, I couldn't generate a response. Please try again."
                response = None
                
        except Exception as e:
            print(f"Error: {e}")
            error = "An error occurred while processing your request."
            response = None

    try:
        return render_template('index.html', response=response, error=error)
    except Exception as e:
        print(f"Template error: {e}")
        return "The index page is currently unavailable. Please try again later.", 500

@app.route('/dashboard')
def dashboard():
    try:
        return render_template('dashboard.html')
    except Exception as e:
        return f"Dashboard template error: {str(e)}", 500

# API endpoints for the dashboard
@app.route('/api/multi-agent-workflow', methods=['POST'])
def run_multi_agent_workflow_api():
    try:
        # Generate sample data for demonstration
        historical_data = generate_historical_data(90)
        
        # Run the multi-agent workflow
        results = run_multi_agent_workflow(
            historical_data=historical_data,
            products=SAMPLE_PRODUCTS,
            text_data={"type": "market", "content": "Recent supply chain disruptions in Asia have led to increased lead times for electronics components."}
        )
        
        return jsonify({"success": True, "results": results})
    except Exception as e:
        print(f"Multi-agent workflow error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/deep-learning-forecast')
def deep_learning_forecast_api():
    try:
        # Generate sample data for demonstration
        historical_data = generate_historical_data(90)
        
        # Run deep learning forecast
        forecast = deep_learning_forecast(historical_data, forecast_days=30)
        
        return jsonify({"success": True, "forecast": forecast})
    except Exception as e:
        print(f"Deep learning forecast error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/anomaly-detection')
def anomaly_detection_api():
    try:
        # Generate sample data for demonstration
        historical_data = generate_historical_data(90)
        
        # Run deep anomaly detection
        anomalies = deep_anomaly_detection(historical_data, sensitivity=1.0)
        
        return jsonify({"success": True, "anomalies": anomalies})
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/nlp-analysis', methods=['POST'])
def nlp_analysis_api():
    try:
        # Get text data from request
        data = request.json
        text = data.get('text', '')
        text_type = data.get('type', 'supplier')
        
        # Run NLP analysis
        analysis = analyze_text(text_type, text)
        
        return jsonify({"success": True, "analysis": analysis})
    except Exception as e:
        print(f"NLP analysis error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/optimize-inventory')
def optimize_inventory_api():
    try:
        # Run reinforcement learning optimization
        optimization_results = optimize_inventory_policy(initial_inventory=5000)
        
        return jsonify({"success": True, "results": optimization_results})
    except Exception as e:
        print(f"Inventory optimization error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/supply-chain-map')
def supply_chain_map():
    """Return supply chain map data"""
    return jsonify({
        "suppliers": SAMPLE_SUPPLIERS,
        "warehouses": SAMPLE_WAREHOUSES
    })

@app.route('/api/inventory')
def inventory():
    """Return inventory data"""
    return jsonify(SAMPLE_PRODUCTS)

@app.route('/api/historical-data')
def historical_data():
    """Return historical performance data"""
    return jsonify(generate_historical_data())

@app.route('/api/advanced-forecast')
def advanced_forecast():
    """Return advanced demand forecast using AI models"""
    historical = generate_historical_data()
    category = request.args.get('category')
    days = request.args.get('days', 30, type=int)
    
    # Use advanced forecasting with fallback to simple forecasting
    try:
        forecast = advanced_demand_forecast(historical, category, days)
        return jsonify(forecast)
    except Exception as e:
        print(f"Error in advanced forecast: {e}")
        # Fallback to simpler forecasting
        return jsonify(forecast_demand(historical, category, days))

@app.route('/api/anomaly-detection')
def anomaly_detection():
    """Detect anomalies in supply chain data using AI"""
    historical = generate_historical_data()
    sensitivity = request.args.get('sensitivity', 0.05, type=float)
    
    try:
        anomalies = detect_supply_chain_anomalies(historical, sensitivity)
        return jsonify(anomalies)
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return jsonify([])

@app.route('/api/inventory-optimization')
def inventory_optimization():
    """Optimize inventory allocation across warehouses"""
    try:
        # Get forecast for next 30 days
        historical = generate_historical_data()
        forecast = advanced_demand_forecast(historical, None, 30)
        
        # Optimize inventory allocation
        optimization = optimize_inventory_allocation(SAMPLE_PRODUCTS, SAMPLE_WAREHOUSES, forecast)
        return jsonify(optimization)
    except Exception as e:
        print(f"Error in inventory optimization: {e}")
        return jsonify([])

@app.route('/api/risk-assessment')
def risk_assessment():
    """Return risk assessment data"""
    risks = [
        {"id": 1, "type": "Supplier Disruption", "probability": 0.3, "impact": 0.8, "mitigation": "Diversify supplier base"},
        {"id": 2, "type": "Transportation Delay", "probability": 0.5, "impact": 0.6, "mitigation": "Increase buffer inventory"},
        {"id": 3, "type": "Demand Surge", "probability": 0.2, "impact": 0.7, "mitigation": "Flexible production capacity"},
        {"id": 4, "type": "Quality Issue", "probability": 0.4, "impact": 0.9, "mitigation": "Enhance quality control processes"},
        {"id": 5, "type": "Regulatory Change", "probability": 0.1, "impact": 0.5, "mitigation": "Monitor regulatory environment"}
    ]
    return jsonify(risks)

@app.route('/api/simulate-scenario', methods=['POST'])
def simulate_scenario():
    """Simulate a supply chain scenario"""
    scenario = request.json
    
    # In a real application, this would run a complex simulation
    # For demo purposes, we'll return some mock results
    results = {
        "inventoryImpact": random.uniform(-0.3, 0.1),
        "costImpact": random.uniform(0.05, 0.3),
        "deliveryTimeImpact": random.uniform(0.1, 0.5),
        "customerSatisfactionImpact": random.uniform(-0.4, -0.1),
        "recommendation": "Based on the simulation, we recommend increasing safety stock by 20% and identifying alternative suppliers."
    }
    
    return jsonify(results)

@app.route('/api/ai-query', methods=['POST'])
def ai_query():
    """Handle AI assistant queries from the dashboard"""
    try:
        data = request.json
        response = process_user_query(data['product'], data['question'])
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"<p class='error'>Error processing request: {str(e)}</p>"}), 500

if __name__ == '__main__':
    app.run(debug=True)
