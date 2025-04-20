import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def analyze_inventory_levels(products):
    """
    Analyze inventory levels and provide recommendations based on demand and lead time
    """
    results = []
    
    for product in products:
        analysis = {
            "product_name": product["name"],
            "category": product["category"],
            "current_inventory": product["inventory"],
            "status": "",
            "recommendation": ""
        }
        
        # Convert lead time to days for calculation
        lead_time_days = 0
        if "week" in product["leadTime"].lower():
            # Extract the number from strings like "2-3 weeks"
            parts = product["leadTime"].split("-")
            if len(parts) > 1:
                lead_time_days = int(float(parts[1].split()[0]) * 7)  # Convert weeks to days
            else:
                lead_time_days = int(float(parts[0].split()[0]) * 7)  # Convert weeks to days
        elif "day" in product["leadTime"].lower():
            parts = product["leadTime"].split("-")
            if len(parts) > 1:
                lead_time_days = int(parts[1].split()[0])  # Already in days
            else:
                lead_time_days = int(parts[0].split()[0])  # Already in days
        
        # Determine status based on inventory and demand
        if product["demand"] == "High" and product["inventory"] < 2000:
            analysis["status"] = "Critical - Low Stock"
            analysis["recommendation"] = f"Immediate reorder recommended. Consider expedited shipping and increasing order quantity by 30% to account for high demand during lead time of {product['leadTime']}."
        elif product["demand"] == "High" and product["inventory"] < 3500:
            analysis["status"] = "Warning - Stock Declining"
            analysis["recommendation"] = f"Place order within {max(lead_time_days // 3, 1)} days to avoid potential stockout. Monitor daily consumption rates."
        elif product["demand"] == "Low" and product["inventory"] > 8000:
            analysis["status"] = "Overstocked"
            analysis["recommendation"] = "Consider promotional activities or redistribution to other warehouses. Delay next order cycle."
        elif product["demand"] == "Medium" and lead_time_days > 21 and product["inventory"] < 6000:
            analysis["status"] = "Caution - Long Lead Time"
            analysis["recommendation"] = "Maintain higher safety stock due to extended lead time. Explore alternative suppliers with shorter lead times."
        else:
            analysis["status"] = "Optimal"
            analysis["recommendation"] = "Inventory levels are within optimal range. Continue regular ordering cycle."
        
        results.append(analysis)
    
    return results

def forecast_demand(historical_data, product_category=None, forecast_days=30):
    """
    Generate a simple demand forecast based on historical data
    Optionally filter by product category
    """
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Calculate average daily demand from historical data
    avg_daily_demand = df['orders'].mean()
    std_daily_demand = df['orders'].std()
    
    # Generate forecast dates
    last_date = df.index.max()
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    # Simple forecast with some randomness to simulate real-world variability
    forecast = []
    trend = 0.005  # Slight upward trend
    
    for i, date in enumerate(forecast_dates):
        # Add trend and seasonality factors
        trend_factor = 1 + (trend * i)  # Increasing trend over time
        day_of_week = date.weekday()
        
        # Weekend effect (lower demand on weekends)
        weekday_factor = 0.8 if day_of_week >= 5 else 1.0
        
        # Monthly seasonality (higher demand at beginning of month)
        monthly_factor = 1.2 if date.day < 5 else 1.0
        
        # Calculate forecasted demand with some randomness
        base_demand = avg_daily_demand * trend_factor * weekday_factor * monthly_factor
        random_factor = np.random.normal(1, 0.1)  # Add some noise
        forecasted_demand = int(base_demand * random_factor)
        
        forecast.append({
            "date": date.strftime("%Y-%m-%d"),
            "forecasted_demand": max(forecasted_demand, 0),  # Ensure non-negative
            "lower_bound": max(int(forecasted_demand - 0.5 * std_daily_demand), 0),
            "upper_bound": int(forecasted_demand + 0.5 * std_daily_demand)
        })
    
    return forecast

def calculate_risk_score(suppliers, warehouses, products):
    """
    Calculate overall supply chain risk score and identify key risk factors
    """
    risk_factors = []
    total_risk_score = 0
    max_possible_score = 0
    
    # Supplier reliability risks
    low_reliability_suppliers = [s for s in suppliers if s["reliability"] < 0.9]
    if low_reliability_suppliers:
        risk_score = len(low_reliability_suppliers) * 10
        total_risk_score += risk_score
        max_possible_score += len(suppliers) * 10
        risk_factors.append({
            "factor": "Supplier Reliability",
            "score": risk_score,
            "max_score": len(suppliers) * 10,
            "description": f"{len(low_reliability_suppliers)} suppliers have reliability below 90%",
            "mitigation": "Develop backup suppliers and implement supplier improvement programs"
        })
    
    # Warehouse utilization risks
    high_utilization_warehouses = [w for w in warehouses if w["utilization"] > 0.85]
    if high_utilization_warehouses:
        risk_score = len(high_utilization_warehouses) * 15
        total_risk_score += risk_score
        max_possible_score += len(warehouses) * 15
        risk_factors.append({
            "factor": "Warehouse Capacity",
            "score": risk_score,
            "max_score": len(warehouses) * 15,
            "description": f"{len(high_utilization_warehouses)} warehouses are above 85% capacity utilization",
            "mitigation": "Expand warehouse space or redistribute inventory to lower-utilization facilities"
        })
    
    # Inventory risks
    low_stock_products = [p for p in products if p["demand"] == "High" and p["inventory"] < 2000]
    if low_stock_products:
        risk_score = len(low_stock_products) * 20
        total_risk_score += risk_score
        max_possible_score += len([p for p in products if p["demand"] == "High"]) * 20
        risk_factors.append({
            "factor": "Inventory Levels",
            "score": risk_score,
            "max_score": len([p for p in products if p["demand"] == "High"]) * 20,
            "description": f"{len(low_stock_products)} high-demand products have critically low inventory",
            "mitigation": "Expedite orders and implement safety stock policies"
        })
    
    # Geographic concentration risk
    supplier_regions = {}
    for supplier in suppliers:
        # Simplistic region determination based on longitude
        region = "Asia" if supplier["location"]["lng"] > 100 else "Europe" if supplier["location"]["lng"] > 0 else "Americas"
        supplier_regions[region] = supplier_regions.get(region, 0) + 1
    
    max_region_concentration = max(supplier_regions.values()) / len(suppliers) if suppliers else 0
    if max_region_concentration > 0.5:  # More than 50% suppliers in one region
        risk_score = int(max_region_concentration * 25)
        total_risk_score += risk_score
        max_possible_score += 25
        risk_factors.append({
            "factor": "Geographic Concentration",
            "score": risk_score,
            "max_score": 25,
            "description": f"Over {int(max_region_concentration*100)}% of suppliers concentrated in one region",
            "mitigation": "Diversify supplier base across different geographic regions"
        })
    
    # Calculate overall risk percentage
    risk_percentage = (total_risk_score / max_possible_score * 100) if max_possible_score > 0 else 0
    
    return {
        "overall_risk_score": total_risk_score,
        "risk_percentage": risk_percentage,
        "risk_level": "High" if risk_percentage > 70 else "Medium" if risk_percentage > 40 else "Low",
        "risk_factors": risk_factors
    }

def optimize_order_quantities(products, historical_data):
    """
    Calculate optimal order quantities based on demand patterns and lead times
    """
    results = []
    
    # Convert historical data to DataFrame for analysis
    hist_df = pd.DataFrame(historical_data)
    
    # Calculate average daily demand and its standard deviation
    avg_daily_demand = hist_df['orders'].mean()
    std_daily_demand = hist_df['orders'].std()
    
    for product in products:
        # Extract lead time in days
        lead_time_days = 0
        if "week" in product["leadTime"].lower():
            parts = product["leadTime"].split("-")
            if len(parts) > 1:
                lead_time_days = int(float(parts[1].split()[0]) * 7)  # Convert weeks to days
            else:
                lead_time_days = int(float(parts[0].split()[0]) * 7)  # Convert weeks to days
        elif "day" in product["leadTime"].lower():
            parts = product["leadTime"].split("-")
            if len(parts) > 1:
                lead_time_days = int(parts[1].split()[0])  # Already in days
            else:
                lead_time_days = int(parts[0].split()[0])  # Already in days
        
        # Adjust demand based on product category
        demand_multiplier = 1.0
        if product["demand"] == "High":
            demand_multiplier = 1.5
        elif product["demand"] == "Medium":
            demand_multiplier = 1.0
        else:  # Low demand
            demand_multiplier = 0.5
        
        # Calculate product-specific demand
        product_daily_demand = avg_daily_demand * demand_multiplier
        
        # Calculate safety stock (covers 95% of demand variations during lead time)
        safety_factor = 1.645  # 95% service level
        safety_stock = safety_factor * std_daily_demand * np.sqrt(lead_time_days) * demand_multiplier
        
        # Calculate reorder point
        reorder_point = (product_daily_demand * lead_time_days) + safety_stock
        
        # Calculate economic order quantity (simplified)
        holding_cost_ratio = 0.25  # 25% of item value per year
        ordering_cost = 100  # Fixed cost per order
        annual_demand = product_daily_demand * 365
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / (product["inventory"] * holding_cost_ratio))
        
        # Round to nearest 100
        eoq = round(eoq / 100) * 100
        
        # Calculate order frequency
        order_frequency_days = (eoq / product_daily_demand) if product_daily_demand > 0 else 90
        
        results.append({
            "product_name": product["name"],
            "category": product["category"],
            "economic_order_quantity": int(max(eoq, 100)),  # Minimum order of 100
            "reorder_point": int(reorder_point),
            "safety_stock": int(safety_stock),
            "order_frequency": f"Every {int(order_frequency_days)} days",
            "next_order_date": (datetime.now() + timedelta(days=max(0, (product["inventory"] - reorder_point) / product_daily_demand))).strftime("%Y-%m-%d") if product_daily_demand > 0 else "Not scheduled"
        })
    
    return results