# Maintenance Optimization System

A comprehensive web-based application for simulating and optimizing maintenance strategies for multi-component systems with degradation. This system helps analyze different maintenance policies, visualize component degradation patterns, and determine optimal intervention strategies.

## 🚀 Features

- **Interactive Web Interface**: User-friendly setup and configuration
- **Signal-Based Maintenance**: Green/Yellow/Red signal system for maintenance decisions
- **Optimal Policy Calculation**: Mathematical optimization using linear programming
- **Real-time Simulation**: Monte Carlo simulation of maintenance strategies
- **Rich Visualizations**: Component states, degradation heatmaps, cost analysis
- **Flexible Configuration**: Customizable component parameters and cost structures

## 📋 System Overview

### Signal System
- **🟢 Green Signal (0)**: All components in perfect condition
- **🟡 Yellow Signal (1)**: Some components degraded but not failed
- **🔴 Red Signal (2)**: At least one component has failed

### Maintenance Strategy
- **Preventive Maintenance**: Triggered after consecutive yellow signals exceed threshold
- **Corrective Maintenance**: Immediate action required on red signal
- **Component Selection**: All components, degraded only, or custom number

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
Flask 2.0+
NumPy
Matplotlib
Seaborn
Gurobipy (for optimal policy calculation)
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd maintenance-optimization-system

# Install dependencies
pip install -r requirements.txt

# Install Gurobi (optional, for optimal policy)
pip install gurobipy

# Run the application
python app.py
```

### Access the Application
Open your browser and navigate to: `http://localhost:5000`

## 📁 Project Structure

```
maintenance-optimization-system/
├── app.py                          # Main Flask application
├── optimal_policy.py               # Optimal policy calculation module
├── static/
│   ├── css/
│   │   └── custom.css              # Custom styling
│   ├── js/
│   │   └── custom.js               # Client-side functionality
│   └── img/
│       └── favicon.svg             # Application icon
├── templates/
│   ├── base.html                   # Base template
│   ├── index.html                  # Home page
│   ├── setup.html                  # Parameter configuration
│   ├── visualization.html          # Results visualization
│   └── policy.html                 # Policy description
└── README.md                       # This file
```

## 🎯 Usage Guide

### 1. Configure System Parameters
- **Components (C)**: Number of system components (1-10)
- **Deterioration Levels (K)**: Maximum degradation before failure (2-15)
- **Degradation Probability**: Likelihood of component degradation per time step

### 2. Set Policy Parameters
- **Yellow Threshold**: Consecutive yellow signals before preventive maintenance
- **Component Selection**: Strategy for selecting components during maintenance

### 3. Define Cost Structure
- **Preventive/Corrective Costs**: Fixed maintenance operation costs
- **Component Costs**: Transfer, replacement, shortage, and excess costs

### 4. Calculate Optimal Policy
Generate mathematically optimal maintenance strategy using linear programming

### 5. Run Simulation
Execute Monte Carlo simulation to analyze policy performance

### 6. Analyze Results
View comprehensive visualizations and performance metrics

## 📊 Visualizations

- **Signal History**: System state transitions over time
- **Degradation Heatmap**: Component deterioration patterns
- **Cost Analysis**: Breakdown of maintenance costs
- **Signal Distribution**: Time spent in each signal state

## 🧮 Mathematical Model

### State Space
- **Component States**: Discrete deterioration levels [0, K]
- **System State**: Joint state of all components
- **Signal Generation**: Deterministic mapping from component states

### Optimization Problem
- **Objective**: Minimize long-term expected cost
- **Constraints**: Maintenance policy restrictions and yellow threshold limits
- **Method**: Linear programming with Markov Decision Process formulation

## 💻 Code Architecture

### Main Application (app.py) - Pseudo Code

```python
class MaintenanceOptimizationSimulator:
    def __init__(self, params):
        # Initialize system parameters
        # Set up component configurations
        # Configure cost structure
    
    def determine_signal(self, state, K):
        # IF all components at level 0: return GREEN
        # ELIF any component at level K: return RED  
        # ELSE: return YELLOW
    
    def run_simulation(self):
        # INITIALIZE simulation variables
        # FOR each time step:
        #   current_signal = determine_signal(component_states)
        #   IF signal is RED:
        #       perform_corrective_maintenance()
        #   ELIF yellow_counter >= threshold:
        #       perform_preventive_maintenance()
        #   ELSE:
        #       degrade_components_probabilistically()
        #   record_costs_and_metrics()
        # RETURN simulation_results
    
    def calculate_performance_metrics(self):
        # Calculate uptime, MTBF, total costs
        # Generate summary statistics
    
    def generate_visualizations(self):
        # Create component state plots
        # Generate heatmaps and cost analysis
        # Produce signal distribution charts

# Flask Routes
@app.route('/setup', methods=['GET', 'POST'])
def setup():
    # Handle parameter configuration
    # Validate input parameters
    # Store in session

@app.route('/calculate_policy', methods=['POST'])
def calculate_policy():
    # Get parameters from session
    # Call optimal_policy.calculate_optimal_policy()
    # Generate policy visualization
    # Return results

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    # Create simulator instance
    # Execute simulation
    # Generate all visualizations
    # Return comprehensive results
```

### Optimal Policy Module (optimal_policy.py) - Pseudo Code

```python
def StateIndex(StateDescV, Dim, SetSize):
    # Convert multi-dimensional state to linear index
    # RETURN linearized state index

def StateDesc(Ind, Dim, SetSize):
    # Convert linear index to multi-dimensional state
    # RETURN state description vector

def PreProcessing(alpha, K, C, Eps):
    # INITIALIZE state space and transition matrices
    # BUILD transition probabilities between states
    # 
    # t = 0
    # ProbReachingT = 1
    # WHILE ProbReachingT > Eps:
    #   t = t + 1
    #   calculate_state_distributions(t)
    #   update_observation_transition_matrix()
    #   ProbReachingT *= probability_of_staying_non_red
    # 
    # RETURN (NumberNonGreenV, DistributionV, ObsTransM, U)

def ReliabilityLP(Code, K, C, U, alpha, costs, yellow_threshold):
    # CREATE linear programming model
    # DEFINE decision variables P[signal, counter, action]
    # 
    # ADD CONSTRAINTS:
    #   - No intervention at perfect state (0,0)
    #   - Probability conservation constraints
    #   - Mandatory intervention in red states
    #   - Mandatory intervention at truncation point
    #   - Yellow threshold constraints (NEW)
    # 
    # IF yellow_threshold is specified:
    #   FOR counter > yellow_threshold:
    #     FORCE P[yellow, counter, no_action] = 0
    #   FORCE intervention at yellow_threshold
    # 
    # SOLVE optimization model
    # RETURN (solution_matrix, objective_value)

def calculate_optimal_policy(params):
    # EXTRACT parameters from input
    # 
    # TRY:
    #   preprocessing_results = PreProcessing(alpha, K, C, Eps)
    #   solution = ReliabilityLP(0, K, C, U, alpha, costs, yellow_threshold)
    #   policy_info = extract_policy_info(solution)
    #   RETURN success_result
    # EXCEPT Exception as e:
    #   RETURN error_result

def extract_policy_info(SolutionMat, C, U, yellow_threshold):
    # INITIALIZE intervention lists
    # 
    # FOR each counter value n ≤ yellow_threshold:
    #   FOR each action a:
    #     IF P[yellow, n, a] > tolerance:
    #       ADD to yellow_interventions
    # 
    # FOR each counter value n:
    #   FOR each action a:
    #     IF P[red, n, a] > tolerance:
    #       ADD to red_interventions
    # 
    # CALCULATE performance probabilities
    # GENERATE policy description
    # RETURN policy_information
```

## 🔧 Configuration Parameters

### System Parameters
| Parameter | Description | Range | Default |
|-----------|-------------|--------|---------|
| C | Number of components | 1-10 | 3 |
| K | Maximum deterioration level | 2-15 | 3 |
| α | Non-degradation probability | 0.05-0.95 | 0.75 |
| Yellow Threshold | Consecutive yellow signals | 1-20 | 5 |

### Cost Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| c₁ | Preventive maintenance cost | 100 |
| c₂ | Corrective maintenance cost | 200 |
| cₜ | Transfer cost per component | 30 |
| cᵣ | Replacement cost per component | 50 |
| cₛ | Shortage cost per component | 60 |
| cₑ | Excess cost per component | 30 |

## 📈 Performance Metrics

- **Total Cost**: Cumulative maintenance expenses
- **Uptime Percentage**: System availability
- **MTBF**: Mean Time Between Failures
- **Intervention Counts**: Preventive vs. corrective maintenance
- **Signal Distribution**: Time spent in each state

## 🚨 Troubleshooting

### Common Issues

1. **Gurobi Installation**
   ```bash
   pip install gurobipy
   # For academic license: https://www.gurobi.com/academia/
   ```

2. **Memory Issues with Large Systems**
   - Reduce number of components (C)
   - Lower maximum deterioration level (K)
   - Increase precision threshold (Eps)

3. **Slow Optimization**
   - Check yellow threshold settings
   - Reduce simulation steps
   - Verify cost parameter ranges

## 📝 Mathematical Background

The system uses a **Semi-Markov Decision Process** formulation where:

- **States**: Joint deterioration levels of all components
- **Actions**: Number of components to maintain
- **Transitions**: Probabilistic degradation model
- **Costs**: Comprehensive maintenance cost structure
- **Objective**: Minimize long-run average cost per unit time

The optimization problem ensures that maintenance interventions respect the user's yellow threshold constraint while finding the mathematically optimal policy within those bounds.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request



## 📚 References

- Markov Decision Processes in Maintenance Optimization
- Signal-Based Maintenance Strategies
- Multi-Component System Reliability
- Linear Programming Applications in Operations Research

## 💡 Future Enhancements

- [ ] Machine learning-based degradation prediction
- [ ] Multi-objective optimization (cost vs. reliability)
- [ ] Real-time data integration
- [ ] Advanced uncertainty modeling
- [ ] Export functionality for simulation results
- [ ] Batch processing for parameter sensitivity analysis

---

**Created by**: Maintenance Optimization Team  
**Last Updated**: December 2024  
**Version**: 1.0.0
