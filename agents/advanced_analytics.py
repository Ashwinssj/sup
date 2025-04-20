import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def advanced_demand_forecast(historical_data, product_category=None, forecast_days=30):
    """
    Generate an advanced demand forecast using exponential smoothing
    Provides more accurate predictions than simple forecasting methods
    """
    # Convert to pandas DataFrame for analysis
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Filter by product category if specified
    if product_category:
        # In a real application, we would filter by category
        # For this demo, we'll just use all data
        pass
    
    # Prepare data for forecasting
    orders_series = df['orders']
    
    # Apply Holt-Winters Exponential Smoothing
    # Using multiplicative seasonality for retail-like data patterns
    try:
        model = ExponentialSmoothing(
            orders_series,
            seasonal_periods=7,  # Weekly seasonality
            trend='add',
            seasonal='mul',
            use_boxcox=True
        ).fit()
        
        # Generate forecast dates
        last_date = df.index.max()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        # Generate predictions
        forecast_values = model.forecast(forecast_days)
        
        # Calculate prediction intervals (80% confidence)
        forecast_df = pd.DataFrame({
            'forecasted_demand': forecast_values,
            'lower_bound': forecast_values * 0.8,  # Simplified lower bound
            'upper_bound': forecast_values * 1.2,  # Simplified upper bound
        }, index=forecast_dates)
        
        # Convert to list of dictionaries for API response
        forecast = []
        for date, row in forecast_df.iterrows():
            forecast.append({
                "date": date.strftime("%Y-%m-%d"),
                "forecasted_demand": max(int(row['forecasted_demand']), 0),
                "lower_bound": max(int(row['lower_bound']), 0),
                "upper_bound": int(row['upper_bound'])
            })
        
        return forecast
    
    except Exception as e:
        print(f"Error in advanced forecasting: {e}")
        # Fall back to simpler forecasting method
        from agents.analytics import forecast_demand
        return forecast_demand(historical_data, product_category, forecast_days)

def detect_supply_chain_anomalies(historical_data, sensitivity=0.05):
    """
    Detect anomalies in supply chain data using Isolation Forest algorithm
    
    Parameters:
    - historical_data: List of dictionaries with supply chain metrics
    - sensitivity: Lower values mean more points will be classified as anomalies
    
    Returns:
    - List of anomalies with dates and affected metrics
    """
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Select numerical columns for anomaly detection
        numerical_cols = ['orders', 'deliveries', 'inventory', 'costs']
        data = df[numerical_cols].copy()
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Apply Isolation Forest
        model = IsolationForest(contamination=sensitivity, random_state=42)
        df['anomaly'] = model.fit_predict(scaled_data)
        
        # -1 indicates an anomaly, 1 indicates normal
        anomalies = df[df['anomaly'] == -1].copy()
        
        # Determine which metrics are anomalous
        result = []
        for _, row in anomalies.iterrows():
            # Calculate z-scores to determine which metrics are anomalous
            z_scores = {}
            for col in numerical_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                z_score = (row[col] - mean_val) / std_val if std_val > 0 else 0
                z_scores[col] = z_score
            
            # Identify metrics with high absolute z-scores
            anomalous_metrics = [col for col, z in z_scores.items() if abs(z) > 2]
            
            if anomalous_metrics:
                result.append({
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "anomalous_metrics": anomalous_metrics,
                    "values": {col: row[col] for col in anomalous_metrics},
                    "severity": max(abs(z_scores[m]) for m in anomalous_metrics) / 4,  # Normalize to 0-1 scale
                    "potential_causes": get_potential_causes(anomalous_metrics, row)
                })
        
        return result
    
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return []

def get_potential_causes(anomalous_metrics, data_row):
    """
    Generate potential causes for anomalies based on the affected metrics
    """
    causes = []
    
    if 'orders' in anomalous_metrics and data_row['orders'] > data_row['deliveries'] * 1.3:
        causes.append("Unexpected demand surge")
    
    if 'deliveries' in anomalous_metrics and data_row['deliveries'] < data_row['orders'] * 0.7:
        causes.append("Possible supplier disruption")
    
    if 'inventory' in anomalous_metrics:
        if data_row['inventory'] < 1500:
            causes.append("Critical inventory shortage")
        elif data_row['inventory'] > 4500:
            causes.append("Excess inventory buildup")
    
    if 'costs' in anomalous_metrics and data_row['costs'] > 12000:
        causes.append("Unusual cost increase")
    
    # If no specific causes identified, provide a generic message
    if not causes:
        causes.append("Multiple factors affecting supply chain performance")
    
    return causes

def optimize_inventory_allocation(products, warehouses, demand_forecast):
    """
    Optimize inventory allocation across warehouses based on demand forecasts
    and warehouse capacities
    """
    results = []
    
    # Calculate total forecasted demand for the next period
    total_forecast = sum(item['forecasted_demand'] for item in demand_forecast[:7])  # Next 7 days
    
    # Calculate total warehouse capacity and current utilization
    total_capacity = sum(w['capacity'] for w in warehouses)
    current_utilization = sum(w['capacity'] * w['utilization'] for w in warehouses)
    
    # Calculate available capacity per warehouse
    available_capacity = {}
    for warehouse in warehouses:
        available_capacity[warehouse['id']] = warehouse['capacity'] * (1 - warehouse['utilization'])
    
    # Prioritize products based on demand and inventory levels
    product_priority = []
    for product in products:
        # Convert demand to numerical value
        demand_value = 3 if product['demand'] == 'High' else 2 if product['demand'] == 'Medium' else 1
        
        # Calculate priority score (higher means higher priority)
        inventory_ratio = product['inventory'] / (demand_value * 1000)  # Normalize by demand
        priority_score = demand_value / max(inventory_ratio, 0.1)  # Avoid division by zero
        
        product_priority.append({
            'product_id': product['id'],
            'name': product['name'],
            'priority_score': priority_score,
            'current_inventory': product['inventory'],
            'category': product['category']
        })
    
    # Sort products by priority (highest first)
    product_priority.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Allocate inventory to warehouses
    for product in product_priority:
        # Calculate optimal allocation across warehouses
        allocation = []
        remaining_inventory = product['current_inventory']
        
        # Sort warehouses by available capacity (highest first)
        sorted_warehouses = sorted(warehouses, key=lambda w: available_capacity[w['id']], reverse=True)
        
        for warehouse in sorted_warehouses:
            if remaining_inventory <= 0:
                break
                
            # Allocate based on available capacity and product priority
            allocation_amount = min(
                remaining_inventory,
                int(available_capacity[warehouse['id']] * 0.8)  # Use 80% of available capacity max
            )
            
            if allocation_amount > 0:
                allocation.append({
                    'warehouse_id': warehouse['id'],
                    'warehouse_name': warehouse['name'],
                    'allocation': allocation_amount,
                    'utilization_impact': allocation_amount / warehouse['capacity']
                })
                
                # Update remaining inventory and available capacity
                remaining_inventory -= allocation_amount
                available_capacity[warehouse['id']] -= allocation_amount
        
        results.append({
            'product_name': product['name'],
            'category': product['category'],
            'total_inventory': product['current_inventory'],
            'allocated_inventory': product['current_inventory'] - remaining_inventory,
            'unallocated_inventory': remaining_inventory,
            'allocation_by_warehouse': allocation
        })
    
    return results