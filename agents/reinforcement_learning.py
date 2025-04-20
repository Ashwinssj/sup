import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class SupplyChainEnvironment:
    """
    A reinforcement learning environment for supply chain optimization.
    This environment simulates inventory management, order placement, and demand fulfillment.
    """
    def __init__(self, initial_inventory=5000, holding_cost=1, stockout_cost=10, order_cost=100):
        self.inventory = initial_inventory
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.order_cost = order_cost
        self.day = 0
        self.demand_history = []
        self.inventory_history = []
        self.order_history = []
        self.reward_history = []
        self.state_history = []
        
    def reset(self):
        """Reset the environment to initial state"""
        self.inventory = 5000
        self.day = 0
        self.demand_history = []
        self.inventory_history = []
        self.order_history = []
        self.reward_history = []
        self.state_history = []
        return self._get_state()
    
    def _generate_demand(self):
        """Generate random demand with seasonality and trend"""
        # Base demand
        base_demand = 500
        
        # Add trend (slight increase over time)
        trend = self.day * 0.5
        
        # Add seasonality (weekly pattern)
        day_of_week = self.day % 7
        seasonality = 100 if day_of_week < 5 else -100  # Higher on weekdays
        
        # Add randomness
        noise = np.random.normal(0, 50)
        
        # Calculate final demand and ensure it's positive
        demand = max(0, int(base_demand + trend + seasonality + noise))
        return demand
    
    def _get_state(self):
        """Return the current state representation"""
        # State includes current inventory and recent demand history
        recent_demands = self.demand_history[-7:] if len(self.demand_history) >= 7 else [0] * (7 - len(self.demand_history)) + self.demand_history
        
        # Add day of week as a feature
        day_of_week = self.day % 7
        
        state = [self.inventory] + recent_demands + [day_of_week]
        return state
    
    def step(self, action):
        """Take an action (order quantity) and return new state, reward, done"""
        # Action is the order quantity
        order_quantity = action
        
        # Generate demand for the day
        demand = self._generate_demand()
        self.demand_history.append(demand)
        
        # Update inventory: add new order, subtract demand
        self.inventory += order_quantity
        fulfilled_demand = min(demand, self.inventory)
        unfulfilled_demand = demand - fulfilled_demand
        self.inventory -= fulfilled_demand
        
        # Calculate costs/rewards
        holding_cost = self.holding_cost * self.inventory  # Cost of holding inventory
        stockout_cost = self.stockout_cost * unfulfilled_demand  # Cost of stockouts
        order_cost = self.order_cost if order_quantity > 0 else 0  # Fixed cost for placing an order
        variable_order_cost = 2 * order_quantity  # Variable cost based on order size
        
        # Total reward is negative of total cost
        reward = -(holding_cost + stockout_cost + order_cost + variable_order_cost)
        
        # Update histories
        self.inventory_history.append(self.inventory)
        self.order_history.append(order_quantity)
        self.reward_history.append(reward)
        
        # Increment day
        self.day += 1
        
        # Get new state
        new_state = self._get_state()
        self.state_history.append(new_state)
        
        # Check if episode is done (e.g., after 30 days)
        done = self.day >= 30
        
        return new_state, reward, done

def optimize_inventory_policy(initial_inventory=5000, episodes=100):
    """
    Use a simple reinforcement learning approach to optimize inventory policy.
    Returns the optimal policy and performance metrics.
    """
    env = SupplyChainEnvironment(initial_inventory=initial_inventory)
    
    # Simple Q-learning parameters
    learning_rate = 0.1
    discount_factor = 0.95
    exploration_rate = 1.0
    max_exploration_rate = 1.0
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001
    
    # Discretize the state space for simplicity
    # We'll use inventory levels and average recent demand as state
    q_table = {}
    
    # Possible actions: order quantities
    actions = [0, 500, 1000, 1500, 2000]
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        # Simplify state representation for Q-table
        simple_state = (state[0] // 500, sum(state[1:8]) // 700)  # (inventory_level, avg_demand_level)
        
        while not done:
            # Exploration-exploitation tradeoff
            if np.random.uniform(0, 1) < exploration_rate:
                action_index = np.random.choice(len(actions))
            else:
                if simple_state not in q_table:
                    q_table[simple_state] = [0] * len(actions)
                action_index = np.argmax(q_table[simple_state])
            
            action = actions[action_index]
            
            # Take action and observe new state and reward
            new_state, reward, done = env.step(action)
            
            # Simplify new state for Q-table
            new_simple_state = (new_state[0] // 500, sum(new_state[1:8]) // 700)
            
            # Q-learning update
            if simple_state not in q_table:
                q_table[simple_state] = [0] * len(actions)
                
            if new_simple_state not in q_table:
                q_table[new_simple_state] = [0] * len(actions)
            
            # Update Q-value
            q_table[simple_state][action_index] = q_table[simple_state][action_index] + learning_rate * \
                (reward + discount_factor * max(q_table[new_simple_state]) - q_table[simple_state][action_index])
            
            # Update state
            simple_state = new_simple_state
        
        # Decay exploration rate
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
    # Extract policy from Q-table
    policy = {}
    for state in q_table:
        policy[state] = actions[np.argmax(q_table[state])]
    
    # Run one episode with the learned policy
    env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state = env._get_state()
        simple_state = (state[0] // 500, sum(state[1:8]) // 700)
        
        # Use policy if state exists in Q-table, otherwise use default action
        if simple_state in policy:
            action = policy[simple_state]
        else:
            action = 1000  # Default order quantity
        
        _, reward, done = env.step(action)
        total_reward += reward
    
    # Calculate performance metrics
    avg_inventory = sum(env.inventory_history) / len(env.inventory_history)
    stockout_days = sum(1 for inv in env.inventory_history if inv <= 0)
    total_ordering_cost = sum(env.order_cost if order > 0 else 0 for order in env.order_history)
    
    results = {
        "policy": policy,
        "performance": {
            "total_reward": total_reward,
            "average_inventory": avg_inventory,
            "stockout_days": stockout_days,
            "total_ordering_cost": total_ordering_cost,
            "inventory_history": env.inventory_history,
            "demand_history": env.demand_history,
            "order_history": env.order_history
        }
    }
    
    return results

def optimize_multi_product_inventory(products, budget_constraint=10000, episodes=50):
    """
    Optimize inventory policies for multiple products with a budget constraint.
    
    Args:
        products: List of product dictionaries with name, initial_inventory, holding_cost, etc.
        budget_constraint: Total budget available for ordering across all products
        episodes: Number of training episodes
        
    Returns:
        Dictionary with optimized policies and performance metrics for each product
    """
    # Initialize environments for each product
    envs = {}
    for product in products:
        envs[product["name"]] = SupplyChainEnvironment(
            initial_inventory=product.get("inventory", 5000),
            holding_cost=product.get("holding_cost", 1),
            stockout_cost=product.get("stockout_cost", 10),
            order_cost=product.get("order_cost", 100)
        )
    
    # Possible actions (order quantities)
    actions = [0, 500, 1000, 1500, 2000]
    
    # Initialize Q-tables for each product
    q_tables = {name: {} for name in envs.keys()}
    
    # Training loop
    for episode in range(episodes):
        # Reset all environments
        states = {name: env.reset() for name, env in envs.items()}
        simple_states = {name: (state[0] // 500, sum(state[1:8]) // 700) for name, state in states.items()}
        
        done = {name: False for name in envs.keys()}
        remaining_budget = budget_constraint
        
        # Run one step for all products
        while not all(done.values()):
            # Determine actions for all products based on exploration-exploitation
            exploration_rate = 0.5 * (1 - episode / episodes)  # Decay exploration rate
            
            # First pass: determine desired actions
            desired_actions = {}
            for name, env in envs.items():
                if done[name]:
                    continue
                    
                if np.random.uniform(0, 1) < exploration_rate:
                    action_index = np.random.choice(len(actions))
                else:
                    if simple_states[name] not in q_tables[name]:
                        q_tables[name][simple_states[name]] = [0] * len(actions)
                    action_index = np.argmax(q_tables[name][simple_states[name]])
                
                desired_actions[name] = actions[action_index]
            
            # Second pass: allocate budget
            # Simple proportional allocation if budget is constrained
            total_desired = sum(desired_actions.values())
            if total_desired > remaining_budget:
                scale_factor = remaining_budget / max(total_desired, 1)
                actual_actions = {name: int(action * scale_factor) for name, action in desired_actions.items()}
                remaining_budget = 0
            else:
                actual_actions = desired_actions
                remaining_budget -= total_desired
            
            # Take actions and update Q-tables
            for name, env in envs.items():
                if done[name]:
                    continue
                    
                action = actual_actions[name]
                action_index = actions.index(action) if action in actions else 0
                
                # Take action
                new_state, reward, done[name] = env.step(action)
                
                # Simplify new state
                new_simple_state = (new_state[0] // 500, sum(new_state[1:8]) // 700)
                
                # Ensure states exist in Q-tables
                if simple_states[name] not in q_tables[name]:
                    q_tables[name][simple_states[name]] = [0] * len(actions)
                if new_simple_state not in q_tables[name]:
                    q_tables[name][new_simple_state] = [0] * len(actions)
                
                # Update Q-value
                q_tables[name][simple_states[name]][action_index] = \
                    q_tables[name][simple_states[name]][action_index] + 0.1 * \
                    (reward + 0.95 * max(q_tables[name][new_simple_state]) - \
                     q_tables[name][simple_states[name]][action_index])
                
                # Update state
                simple_states[name] = new_simple_state
    
    # Extract policies and evaluate performance
    results = {}
    for name, env in envs.items():
        # Extract policy
        policy = {}
        for state in q_tables[name]:
            policy[state] = actions[np.argmax(q_tables[name][state])]
        
        # Evaluate policy
        env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state = env._get_state()
            simple_state = (state[0] // 500, sum(state[1:8]) // 700)
            
            if simple_state in policy:
                action = policy[simple_state]
            else:
                action = 1000  # Default
            
            _, reward, done = env.step(action)
            total_reward += reward
        
        # Calculate metrics
        avg_inventory = sum(env.inventory_history) / len(env.inventory_history)
        stockout_days = sum(1 for inv in env.inventory_history if inv <= 0)
        
        results[name] = {
            "policy": policy,
            "performance": {
                "total_reward": total_reward,
                "average_inventory": avg_inventory,
                "stockout_days": stockout_days,
                "inventory_history": env.inventory_history,
                "demand_history": env.demand_history,
                "order_history": env.order_history
            }
        }
    
    return results