document.addEventListener('DOMContentLoaded', function() {
    // Initialize the supply chain map
    initializeMap();
    
    // Load historical performance data
    loadHistoricalData();
    
    // Load inventory data
    loadInventoryData();
    
    // Initialize risk matrix
    initializeRiskMatrix();
    
    // Set up simulation controls
    setupSimulationControls();
    
    // Load advanced analytics
    loadAdvancedForecast();
    loadAnomalyDetection();
    
    // Set up AI assistant form
    setupAIAssistant();
    
    // Add event listeners for AI action buttons
    document.getElementById('run-ai-workflow').addEventListener('click', runMultiAgentWorkflow);
    document.getElementById('view-ai-insights').addEventListener('click', viewAIInsights);
});

// Supply Chain Map
function initializeMap() {
    // Create the map
    const map = L.map('supply-chain-map').setView([39.8283, -98.5795], 4); // Center on US
    
    // Add the tile layer (OpenStreetMap)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    
    // Fetch supply chain map data
    fetch('/api/supply-chain-map')
        .then(response => response.json())
        .then(data => {
            // Add suppliers to the map
            data.suppliers.forEach(supplier => {
                const marker = L.marker([supplier.location.lat, supplier.location.lng], {
                    icon: L.divIcon({
                        className: 'supplier-marker',
                        html: `<div style="background-color: #e74c3c; width: 12px; height: 12px; border-radius: 50%;"></div>`,
                        iconSize: [12, 12]
                    })
                }).addTo(map);
                
                marker.bindPopup(`
                    <strong>${supplier.name}</strong><br>
                    Reliability: ${supplier.reliability * 100}%<br>
                    Products: ${supplier.products.join(', ')}
                `);
            });
            
            // Add warehouses to the map
            data.warehouses.forEach(warehouse => {
                const marker = L.marker([warehouse.location.lat, warehouse.location.lng], {
                    icon: L.divIcon({
                        className: 'warehouse-marker',
                        html: `<div style="background-color: #3498db; width: 14px; height: 14px; border-radius: 50%;"></div>`,
                        iconSize: [14, 14]
                    })
                }).addTo(map);
                
                marker.bindPopup(`
                    <strong>${warehouse.name}</strong><br>
                    Capacity: ${warehouse.capacity} units<br>
                    Utilization: ${warehouse.utilization * 100}%
                `);
            });
            
            // Draw routes between suppliers and warehouses
            data.suppliers.forEach(supplier => {
                data.warehouses.forEach(warehouse => {
                    // Only draw routes for some supplier-warehouse pairs (for demonstration)
                    if (Math.random() > 0.5) {
                        const polyline = L.polyline([
                            [supplier.location.lat, supplier.location.lng],
                            [warehouse.location.lat, warehouse.location.lng]
                        ], {
                            color: '#2ecc71',
                            weight: 2,
                            opacity: 0.7,
                            dashArray: '5, 5'
                        }).addTo(map);
                        
                        polyline.bindPopup(`
                            <strong>Route</strong><br>
                            From: ${supplier.name}<br>
                            To: ${warehouse.name}<br>
                            Transit Time: ${Math.floor(Math.random() * 10) + 2} days
                        `);
                    }
                });
            });
        })
        .catch(error => console.error('Error loading supply chain map data:', error));
}

// Historical Performance Chart
function loadHistoricalData() {
    fetch('/api/historical-data')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('performance-chart').getContext('2d');
            
            // Extract dates and metrics
            const dates = data.map(item => item.date);
            const orders = data.map(item => item.orders);
            const deliveries = data.map(item => item.deliveries);
            const inventory = data.map(item => item.inventory);
            
            // Create the chart
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [
                        {
                            label: 'Orders',
                            data: orders,
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Deliveries',
                            data: deliveries,
                            borderColor: '#2ecc71',
                            backgroundColor: 'rgba(46, 204, 113, 0.1)',
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Inventory',
                            data: inventory,
                            borderColor: '#f39c12',
                            backgroundColor: 'rgba(243, 156, 18, 0.1)',
                            tension: 0.4,
                            fill: true,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            ticks: {
                                maxTicksLimit: 10
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Orders & Deliveries'
                            }
                        },
                        y1: {
                            position: 'right',
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Inventory'
                            },
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        },
                        legend: {
                            position: 'top'
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error loading historical data:', error));
}

// Inventory Table
function loadInventoryData() {
    fetch('/api/inventory')
        .then(response => response.json())
        .then(data => {
            const tableBody = document.querySelector('#inventory-table tbody');
            
            // Clear existing rows
            tableBody.innerHTML = '';
            
            // Add rows for each product
            data.forEach(product => {
                // Determine status based on inventory and demand
                let status = '';
                let statusClass = '';
                
                if (product.demand === 'High' && product.inventory < 2000) {
                    status = 'Low Stock';
                    statusClass = 'status-warning';
                } else if (product.demand === 'Low' && product.inventory > 5000) {
                    status = 'Overstocked';
                    statusClass = 'status-warning';
                } else {
                    status = 'Optimal';
                    statusClass = 'status-good';
                }
                
                // Create the row
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${product.name}</td>
                    <td>${product.category}</td>
                    <td>${product.inventory.toLocaleString()}</td>
                    <td>${product.demand}</td>
                    <td>${product.leadTime}</td>
                    <td class="${statusClass}">${status}</td>
                `;
                
                tableBody.appendChild(row);
            });
            
            // Add CSS for status classes
            const style = document.createElement('style');
            style.textContent = `
                .status-good { color: #2ecc71; font-weight: bold; }
                .status-warning { color: #f39c12; font-weight: bold; }
                .status-critical { color: #e74c3c; font-weight: bold; }
            `;
            document.head.appendChild(style);
        })
        .catch(error => console.error('Error loading inventory data:', error));
}

// Risk Matrix
function initializeRiskMatrix() {
    fetch('/api/risk-assessment')
        .then(response => response.json())
        .then(risks => {
            const riskMatrix = document.getElementById('risk-matrix');
            const riskDetails = document.getElementById('risk-details');
            
            // Clear existing content
            riskMatrix.innerHTML = '';
            
            // Create the risk matrix cells (5x5 grid)
            for (let i = 5; i >= 1; i--) { // Impact (y-axis, top to bottom)
                for (let j = 1; j <= 5; j++) { // Probability (x-axis, left to right)
                    const cell = document.createElement('div');
                    cell.className = 'risk-cell';
                    
                    // Determine cell color based on risk level
                    let color;
                    const riskLevel = i * j; // Simple calculation for risk level
                    
                    if (riskLevel >= 16) {
                        color = '#e74c3c'; // High risk (red)
                    } else if (riskLevel >= 8) {
                        color = '#f39c12'; // Medium risk (orange)
                    } else {
                        color = '#2ecc71'; // Low risk (green)
                    }
                    
                    cell.style.backgroundColor = color;
                    
                    // Find risks that fall into this cell
                    const cellRisks = risks.filter(risk => {
                        const riskProbability = Math.ceil(risk.probability * 5);
                        const riskImpact = Math.ceil(risk.impact * 5);
                        return riskProbability === j && riskImpact === i;
                    });
                    
                    // Add risk indicators if there are risks in this cell
                    if (cellRisks.length > 0) {
                        cell.innerHTML = `<span class="risk-count">${cellRisks.length}</span>`;
                        cell.style.cursor = 'pointer';
                        
                        // Show risk details on click
                        cell.addEventListener('click', () => {
                            let detailsHTML = '<h4>Risks in this category:</h4><ul>';
                            
                            cellRisks.forEach(risk => {
                                detailsHTML += `
                                    <li>
                                        <strong>${risk.type}</strong><br>
                                        Probability: ${(risk.probability * 100).toFixed(0)}%<br>
                                        Impact: ${(risk.impact * 100).toFixed(0)}%<br>
                                        Mitigation: ${risk.mitigation}
                                    </li>
                                `;
                            });
                            
                            detailsHTML += '</ul>';
                            riskDetails.innerHTML = detailsHTML;
                        });
                    }
                    
                    riskMatrix.appendChild(cell);
                }
            }
            
            // Add CSS for the risk matrix
            const style = document.createElement('style');
            style.textContent = `
                #risk-matrix {
                    display: grid;
                    grid-template-columns: repeat(5, 1fr);
                    grid-template-rows: repeat(5, 1fr);
                    gap: 5px;
                    height: 100%;
                }
                .risk-cell {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 4px;
                }
                .risk-count {
                    color: white;
                    font-weight: bold;
                    font-size: 16px;
                    text-shadow: 0 0 2px rgba(0, 0, 0, 0.5);
                }
            `;
            document.head.appendChild(style);
            
            // Show default risk details
            riskDetails.innerHTML = '<p>Click on a cell in the risk matrix to view details.</p>';
        })
        .catch(error => console.error('Error loading risk assessment data:', error));
}

// Simulation Controls
function setupSimulationControls() {
    const scenarioType = document.getElementById('scenario-type');
    const scenarioSeverity = document.getElementById('scenario-severity');
    const severityValue = document.getElementById('severity-value');
    const scenarioDuration = document.getElementById('scenario-duration');
    const runSimulationButton = document.getElementById('run-simulation');
    const simulationResults = document.getElementById('simulation-results');
    
    // Update severity value display when slider changes
    scenarioSeverity.addEventListener('input', () => {
        severityValue.textContent = scenarioSeverity.value;
    });
    
    // Run simulation when button is clicked
    runSimulationButton.addEventListener('click', () => {
        // Show loading state
        runSimulationButton.disabled = true;
        runSimulationButton.textContent = 'Running Simulation...';
        simulationResults.innerHTML = '<p>Processing simulation data...</p>';
        simulationResults.style.display = 'block';
        
        // Prepare simulation parameters
        const simulationParams = {
            scenarioType: scenarioType.value,
            severity: parseInt(scenarioSeverity.value),
            duration: parseInt(scenarioDuration.value)
        };
        
        // Send simulation request to the server
        fetch('/api/simulate-scenario', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(simulationParams)
        })
        .then(response => response.json())
        .then(results => {
            // Format the impact values as percentages
            const inventoryImpact = (results.inventoryImpact * 100).toFixed(1);
            const costImpact = (results.costImpact * 100).toFixed(1);
            const deliveryTimeImpact = (results.deliveryTimeImpact * 100).toFixed(1);
            const customerSatisfactionImpact = (results.customerSatisfactionImpact * 100).toFixed(1);
            
            // Determine impact classes for styling
            const getImpactClass = (value) => {
                if (value > 0) return 'negative-impact';
                if (value < 0) return 'positive-impact';
                return 'neutral-impact';
            };
            
            // Display the results
            simulationResults.innerHTML = `
                <h4>Simulation Results: ${formatScenarioType(scenarioType.value)}</h4>
                <div class="impact-grid">
                    <div class="impact-item">
                        <div class="impact-label">Inventory Impact</div>
                        <div class="impact-value ${getImpactClass(results.inventoryImpact)}">
                            ${inventoryImpact}%
                        </div>
                    </div>
                    <div class="impact-item">
                        <div class="impact-label">Cost Impact</div>
                        <div class="impact-value ${getImpactClass(results.costImpact)}">
                            +${costImpact}%
                        </div>
                    </div>
                    <div class="impact-item">
                        <div class="impact-label">Delivery Time Impact</div>
                        <div class="impact-value ${getImpactClass(results.deliveryTimeImpact)}">
                            +${deliveryTimeImpact}%
                        </div>
                    </div>
                    <div class="impact-item">
                        <div class="impact-label">Customer Satisfaction</div>
                        <div class="impact-value ${getImpactClass(results.customerSatisfactionImpact)}">
                            ${customerSatisfactionImpact}%
                        </div>
                    </div>
                </div>
                <div class="recommendation">
                    <h4>Recommendation</h4>
                    <p>${results.recommendation}</p>
                </div>
            `;
            
            // Add CSS for the impact grid
            const style = document.createElement('style');
            style.textContent = `
                .impact-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 10px;
                    margin-bottom: 15px;
                }
                .impact-item {
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                }
                .impact-label {
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-bottom: 5px;
                }
                .impact-value {
                    font-size: 18px;
                    font-weight: bold;
                }
                .positive-impact {
                    color: #2ecc71;
                }
                .negative-impact {
                    color: #e74c3c;
                }
                .neutral-impact {
                    color: #7f8c8d;
                }
                .recommendation {
                    background-color: #eaf2f8;
                    padding: 10px;
                    border-radius: 5px;
                    border-left: 4px solid #3498db;
                }
                .recommendation h4 {
                    margin-top: 0;
                    color: #2980b9;
                }
            `;
            document.head.appendChild(style);
            
            // Reset button state
            runSimulationButton.disabled = false;
            runSimulationButton.textContent = 'Run Simulation';
        })
        .catch(error => {
            console.error('Error running simulation:', error);
            simulationResults.innerHTML = `<p class="error">Error running simulation: ${error.message}</p>`;
            runSimulationButton.disabled = false;
            runSimulationButton.textContent = 'Run Simulation';
        });
    });
    
    // Helper function to format scenario type for display
    function formatScenarioType(type) {
        return type.split('-').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
}

// Add a function to toggle between dark and light mode
function setupThemeToggle() {
    // Create theme toggle button
    const themeToggle = document.createElement('button');
    themeToggle.id = 'theme-toggle';
    themeToggle.className = 'theme-toggle';
    themeToggle.innerHTML = '<span class="toggle-icon">🌙</span>';
    document.body.appendChild(themeToggle);
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        themeToggle.innerHTML = '<span class="toggle-icon">☀️</span>';
    }
    
    // Toggle theme on button click
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-theme');
        
        if (document.body.classList.contains('dark-theme')) {
            localStorage.setItem('theme', 'dark');
            themeToggle.innerHTML = '<span class="toggle-icon">☀️</span>';
        } else {
            localStorage.setItem('theme', 'light');
            themeToggle.innerHTML = '<span class="toggle-icon">🌙</span>';
        }
        
        // Refresh charts if they exist
        if (window.charts) {
            window.charts.forEach(chart => chart.update());
        }
    });
    
    // Add CSS for theme toggle and dark theme
    const style = document.createElement('style');
    style.textContent = `
        .theme-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background-color: #3498db;
            color: white;
            border: none;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.3s;
        }
        
        .theme-toggle:hover {
            background-color: #2980b9;
        }
        
        .toggle-icon {
            font-size: 20px;
        }
        
        .dark-theme {
            background-color: #1a1a2e;
            color: #f0f0f0;
        }
        
        .dark-theme .dashboard-container {
            background-color: #1a1a2e;
        }
        
        .dark-theme .dashboard-card {
            background-color: #16213e;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .dark-theme .kpi-item,
        .dark-theme .data-table th,
        .dark-theme #risk-details,
        .dark-theme .simulation-results {
            background-color: #0f3460;
        }
        
        .dark-theme h1, 
        .dark-theme h2, 
        .dark-theme h3, 
        .dark-theme h4 {
            color: #e94560;
        }
        
        .dark-theme .kpi-value {
            color: #f0f0f0;
        }
        
        .dark-theme .data-table td {
            border-bottom: 1px solid #16213e;
        }
        
        .dark-theme .data-table tr:hover {
            background-color: #0f3460;
        }
    `;
    document.head.appendChild(style);
}

// Call the theme toggle setup
document.addEventListener('DOMContentLoaded', function() {
    // Existing initialization code...
    
    // Advanced Demand Forecast
function loadAdvancedForecast() {
    fetch('/api/advanced-forecast')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('forecast-chart');
            if (!ctx) {
                console.error('Forecast chart canvas not found');
                return;
            }
            
            // Extract dates and forecast values
            const dates = data.map(item => item.date);
            const forecastValues = data.map(item => item.forecasted_demand);
            const lowerBounds = data.map(item => item.lower_bound);
            const upperBounds = data.map(item => item.upper_bound);
            
            // Create the chart
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [
                        {
                            label: 'Forecasted Demand',
                            data: forecastValues,
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            tension: 0.4,
                            fill: false
                        },
                        {
                            label: 'Lower Bound',
                            data: lowerBounds,
                            borderColor: 'rgba(52, 152, 219, 0.5)',
                            backgroundColor: 'transparent',
                            borderDash: [5, 5],
                            tension: 0.4,
                            fill: false
                        },
                        {
                            label: 'Upper Bound',
                            data: upperBounds,
                            borderColor: 'rgba(52, 152, 219, 0.5)',
                            backgroundColor: 'transparent',
                            borderDash: [5, 5],
                            tension: 0.4,
                            fill: '-1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            ticks: {
                                maxTicksLimit: 10
                            },
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Demand (Units)'
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        },
                        legend: {
                            position: 'top'
                        },
                        title: {
                            display: true,
                            text: 'AI-Powered Demand Forecast (30 Days)'
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error loading forecast data:', error));
}

// Anomaly Detection
function loadAnomalyDetection() {
    fetch('/api/anomaly-detection')
        .then(response => response.json())
        .then(anomalies => {
            const anomalyContainer = document.getElementById('anomaly-container');
            if (!anomalyContainer) {
                console.error('Anomaly container not found');
                return;
            }
            
            if (anomalies.length === 0) {
                anomalyContainer.innerHTML = '<p class="no-anomalies">No anomalies detected in recent data.</p>';
                return;
            }
            
            // Sort anomalies by severity (highest first)
            anomalies.sort((a, b) => b.severity - a.severity);
            
            let html = '<h3>Detected Anomalies</h3><div class="anomaly-list">';
            
            anomalies.forEach(anomaly => {
                const severityClass = anomaly.severity > 0.7 ? 'high' : 
                                     anomaly.severity > 0.4 ? 'medium' : 'low';
                                     
                html += `
                    <div class="anomaly-item severity-${severityClass}">
                        <div class="anomaly-date">${anomaly.date}</div>
                        <div class="anomaly-metrics">
                            <strong>Affected metrics:</strong> ${anomaly.anomalous_metrics.join(', ')}
                        </div>
                        <div class="anomaly-values">
                            <strong>Values:</strong> 
                            <ul>
                `;
                            
                for (const [metric, value] of Object.entries(anomaly.values)) {
                    html += `<li>${metric}: ${value}</li>`;
                }
                
                html += `
                            </ul>
                        </div>
                        <div class="anomaly-causes">
                            <strong>Potential causes:</strong>
                            <ul>
                `;
                            
                anomaly.potential_causes.forEach(cause => {
                    html += `<li>${cause}</li>`;
                });
                
                html += `
                            </ul>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            anomalyContainer.innerHTML = html;
            
            // Add CSS for anomaly items
            const style = document.createElement('style');
            style.textContent = `
                .anomaly-list {
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                    margin-top: 15px;
                }
                .anomaly-item {
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 5px solid #ccc;
                }
                .severity-high {
                    border-left-color: #e74c3c;
                    background-color: rgba(231, 76, 60, 0.1);
                }
                .severity-medium {
                    border-left-color: #f39c12;
                    background-color: rgba(243, 156, 18, 0.1);
                }
                .severity-low {
                    border-left-color: #3498db;
                    background-color: rgba(52, 152, 219, 0.1);
                }
                .anomaly-date {
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .no-anomalies {
                    color: #27ae60;
                    font-weight: bold;
                }
            `;
            document.head.appendChild(style);
        })
        .catch(error => console.error('Error loading anomaly data:', error));
}

function setupAIAssistant() {
    const aiForm = document.getElementById('ai-assistant-form');
    const aiResponse = document.getElementById('ai-response');
    
    if (!aiForm || !aiResponse) {
        console.error('AI assistant form or response container not found');
        return;
    }
    
    aiForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const product = document.getElementById('ai-product').value;
        const question = document.getElementById('ai-question').value;
        
        if (!product || !question) {
            aiResponse.innerHTML = '<p class="error">Please enter both product name and question.</p>';
            return;
        }
        
        // Show loading indicator
        aiResponse.innerHTML = '<div class="loading">Processing your request...</div>';
        
        // Send request to AI assistant API
        fetch('/api/ai-query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                product: product,
                question: question
            })
        })
        .then(response => response.json())
        .then(data => {
            aiResponse.innerHTML = data.response;
        })
        .catch(error => {
            console.error('Error querying AI assistant:', error);
            aiResponse.innerHTML = '<p class="error">An error occurred while processing your request. Please try again.</p>';
        });
    });
    // Add this to the DOMContentLoaded event listener
    document.getElementById('ai-assistant-form').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const product = document.getElementById('ai-product').value;
        const question = document.getElementById('ai-question').value;
        const responseDiv = document.getElementById('ai-response');
        
        responseDiv.innerHTML = '<p>Analyzing supply chain data...</p>';
        responseDiv.style.display = 'block';
        
        fetch('/api/ai-query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                product: product,
                question: question
            })
        })
        .then(response => response.json())
        .then(data => {
            responseDiv.innerHTML = data.response;
        })
        .catch(error => {
            responseDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        });
    });
}
    
    // Add theme toggle functionality
    setupThemeToggle();
});