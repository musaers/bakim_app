from flask import Flask, render_template, request, jsonify, session
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.figure import Figure
import random
import time
import json
import os
from flask_session import Session

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

class MaintenanceOptimizationSimulator:
    def __init__(self, params):
        # System parameters
        self.C = params.get('C', 3)  # Number of components
        self.K = params.get('K', 3)  # Maximum deterioration level
        self.alpha = params.get('alpha', 0.25)  # Prob. of degradation (1-alpha in paper)
        self.simulation_steps = params.get('simulation_steps', 100)
        self.yellow_threshold = params.get('yellow_threshold', 5)  # Maximum acceptable yellow states
        
        # Cost parameters
        self.c1 = params.get('c1', 100)  # Preventive maintenance cost
        self.c2 = params.get('c2', 200)  # Corrective maintenance cost
        self.ct = params.get('ct', 30)   # Transfer cost per component
        self.cr = params.get('cr', 50)   # Replacement cost per component
        self.cs = params.get('cs', 60)   # Shortage cost per component
        self.ce = params.get('ce', 30)   # Excess cost per component
        
        # Component parameters
        self.component_params = params.get('component_params', [])
        if not self.component_params:
            self.update_component_params()
            
        # Results placeholders
        self.optimal_policy = None
        self.simulation_results = None
        self.maintenance_events = []
        
    def update_component_params(self):
        """Update component parameters"""
        num_components = self.C
        
        # Reset the component_params list with the current number of components
        old_params = self.component_params.copy() if hasattr(self, 'component_params') and self.component_params else []
        self.component_params = []
        
        for i in range(num_components):
            if i < len(old_params):
                # Keep existing parameters
                self.component_params.append(old_params[i])
            else:
                # Create new parameters with default values
                self.component_params.append({
                    'name': f"Component {i+1}",
                    'k': self.K,  # Maximum deterioration level
                    'p': 1 - self.alpha,  # Degradation probability
                    'current_state': 0  # Initial state
                })
        
        return self.component_params

    def determine_signal(self, state, K):
        """Determine the signal based on component states"""
        # Green (0): All components are at level 0
        if all(d == 0 for d in state):
            return 0
        
        # Red (2): At least one component is at level K (failed)
        if any(d >= K for d in state):
            return 2
        
        # Yellow (1): Some degradation but no failures
        return 1

    def run_simulation(self):
        """Run a simulation based on the current parameters and policy"""
        # Get system parameters
        C = self.C
        K = self.K
        alpha = self.alpha  # Probability of NOT degrading
        simulation_steps = self.simulation_steps
        yellow_threshold = self.yellow_threshold  # Maximum acceptable yellow states
        
        # Make sure component_params is updated
        self.update_component_params()
        
        # Initialize simulation variables
        component_states = np.zeros((C, simulation_steps + 1), dtype=int)
        current_states = np.zeros(C, dtype=int)
        signal_history = np.zeros(simulation_steps + 1, dtype=int)
        signal_history[0] = 0  # Initial signal is green
        
        # Initialize yellow counter for tracking consecutive yellow signals
        yellow_counter = 0
        yellow_threshold_reached_count = 0
        
        # Reset metrics
        self.intervention_count = 0
        self.preventive_maintenance_count = 0
        self.corrective_maintenance_count = 0
        self.failure_count = 0
        self.downtime_steps = 0
        self.maintenance_events = []
        
        # Initialize cost tracking
        maintenance_costs = np.zeros(simulation_steps + 1)
        transfer_costs = np.zeros(simulation_steps + 1)
        shortage_costs = np.zeros(simulation_steps + 1)
        excess_costs = np.zeros(simulation_steps + 1)
        component_costs = np.zeros(simulation_steps + 1)
        cumulative_costs = np.zeros(simulation_steps + 1)
        
        # Initialize logs
        logs = []
        
        # Main simulation loop
        for t in range(simulation_steps):
            # Store current component states
            component_states[:, t] = current_states
            
            # Current signal is determined by component states
            current_signal = self.determine_signal(current_states, K)
            signal_history[t] = current_signal
            
            # Update yellow counter based on current signal
            if current_signal == 1:  # Yellow signal
                yellow_counter += 1
            else:
                yellow_counter = 0  # Reset counter if not yellow
            
            # Determine if maintenance is needed based on:
            # 1. Red signal (component failure) - immediate maintenance
            # 2. Yellow counter exceeding threshold
            maintenance_needed = False
            maintenance_type = None
            
            if current_signal == 2:  # Red signal - immediate maintenance
                maintenance_needed = True
                maintenance_type = "corrective"
            elif yellow_counter >= yellow_threshold:  # Yellow threshold reached
                maintenance_needed = True
                maintenance_type = "preventive"
                yellow_threshold_reached_count += 1
            
            # Execute maintenance if needed
            if maintenance_needed:
                self.intervention_count += 1
                self.maintenance_events.append(t)
                
                # Count by type
                if maintenance_type == "corrective":
                    self.corrective_maintenance_count += 1
                    maintenance_costs[t] = self.c2  # Corrective maintenance cost
                else:  # preventive
                    self.preventive_maintenance_count += 1
                    maintenance_costs[t] = self.c1  # Preventive maintenance cost
                
                # For maintenance we take all components
                action = C
                
                # Transfer cost
                transfer_costs[t] = action * self.ct
                
                # Count actual degraded components
                degraded_count = np.sum(current_states > 0)
                
                # Replacement cost
                component_costs[t] = degraded_count * self.cr
                
                # Shortage or excess cost
                if action < degraded_count:
                    shortage_costs[t] = (degraded_count - action) * self.cs
                else:
                    excess_costs[t] = (action - degraded_count) * self.ce
                
                # Reset all components to perfect condition
                current_states.fill(0)
                
                # Reset yellow counter after maintenance
                yellow_counter = 0
                
                # Log maintenance action
                logs.append(f"Time {t}: {maintenance_type.capitalize()} maintenance performed. Reset all components.")
            else:
                # No intervention, degrade components according to their probabilities
                for i in range(C):
                    # Component-specific degradation rate
                    comp_alpha = self.component_params[i]['p']  # This is the degradation probability
                    
                    # Only degrade if not already at maximum level
                    if current_states[i] < K and random.random() < comp_alpha:
                        current_states[i] += 1
                
                # Check for failures
                if np.any(current_states >= K):
                    self.failure_count += 1
                    self.downtime_steps += 1
            
            # Calculate cumulative costs
            if t > 0:
                cumulative_costs[t] = (cumulative_costs[t-1] + maintenance_costs[t] + 
                                      transfer_costs[t] + component_costs[t] + 
                                      shortage_costs[t] + excess_costs[t])
            else:
                cumulative_costs[t] = (maintenance_costs[t] + transfer_costs[t] + 
                                      component_costs[t] + shortage_costs[t] + excess_costs[t])
        
        # Store final state
        component_states[:, simulation_steps] = current_states
        
        # Store simulation results
        self.simulation_results = {
            'component_states': component_states.tolist(),
            'signal_history': signal_history.tolist(),
            'maintenance_events': self.maintenance_events,
            'yellow_threshold_reached_count': yellow_threshold_reached_count,
            'logs': logs,
            'costs': {
                'maintenance': maintenance_costs.tolist(),
                'transfer': transfer_costs.tolist(),
                'component': component_costs.tolist(),
                'shortage': shortage_costs.tolist(),
                'excess': excess_costs.tolist(),
                'cumulative': cumulative_costs.tolist()
            }
        }
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(simulation_steps)
        
        return {
            'results': self.simulation_results,
            'metrics': metrics
        }

    def calculate_performance_metrics(self, simulation_steps):
        """Calculate performance metrics based on simulation results"""
        # Uptime percentage
        uptime_percentage = ((simulation_steps - self.downtime_steps) / simulation_steps) * 100
        
        # Mean time between failures (MTBF)
        if self.failure_count > 0:
            mtbf = simulation_steps / self.failure_count
        else:
            mtbf = simulation_steps  # No failures
        
        # Total cost
        if self.simulation_results:
            total_cost = self.simulation_results['costs']['cumulative'][-1]
        else:
            total_cost = 0
        
        # Return metrics
        metrics = {
            'total_cost': round(total_cost, 2),
            'uptime_percentage': round(uptime_percentage, 2),
            'mtbf': round(mtbf, 2),
            'intervention_count': self.intervention_count,
            'preventive_maintenance_count': self.preventive_maintenance_count,
            'corrective_maintenance_count': self.corrective_maintenance_count,
            'failure_count': self.failure_count,
            'yellow_threshold_reached_count': self.simulation_results['yellow_threshold_reached_count'] if self.simulation_results else 0
        }
        
        return metrics

    def generate_component_state_plot(self):
        """Generate plot for component states"""
        if not self.simulation_results:
            return None
            
        signal_history = np.array(self.simulation_results['signal_history'])
        maintenance_events = self.maintenance_events
        
        # Create figure
        fig = Figure(figsize=(10, 6), dpi=100)
        
        # Create a single subplot for signal history
        ax = fig.add_subplot(111)
        
        # Create a legend handler manually
        legend_elements = []
        
        # Plot signal history
        for signal, color, label in [(0, 'green', 'Green'), (1, 'gold', 'Yellow'), (2, 'red', 'Red')]:
            mask = signal_history == signal
            if np.any(mask):
                points = ax.scatter(np.where(mask)[0], [signal] * np.sum(mask), 
                                   color=color, label=f"{label} ({signal})")
                legend_elements.append(points)
        
        # Add maintenance event line to legend if there are any events
        if maintenance_events:
            line = ax.axvline(x=maintenance_events[0], color='blue', linestyle='-', alpha=0.5, 
                             label='Maintenance')
            legend_elements.append(line)
            
            # Add the rest of maintenance events without adding to legend
            for event in maintenance_events[1:]:
                ax.axvline(x=event, color='blue', linestyle='-', alpha=0.5)
        
        # Set limits and labels
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Green (0)', 'Yellow (1)', 'Red (2)'])
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Signal", fontsize=12)
        ax.set_title("System Signal Over Time", fontsize=14)
        
        # Add legend manually
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add yellow threshold annotation
        yellow_threshold = self.yellow_threshold
        ax.text(0.02, 0.02, f"Yellow Threshold: {yellow_threshold}", transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Adjust layout
        fig.tight_layout()
        
        # Convert to base64 string for embedding in HTML
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return img_str

    def generate_heatmap_plot(self):
        """Generate heatmap visualization of component degradation"""
        if not self.simulation_results:
            return None
            
        component_states = np.array(self.simulation_results['component_states'])
        maintenance_events = self.maintenance_events
        C = self.C
        K = self.K
        simulation_steps = self.simulation_steps
        
        # Create figure
        fig = Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # If too many time steps, sample at regular intervals
        if simulation_steps > 50:
            sample_points = np.linspace(0, simulation_steps, 50, dtype=int)
            sampled_states = component_states[:, sample_points]
            time_labels = sample_points
        else:
            sampled_states = component_states
            time_labels = range(simulation_steps + 1)
        
        # Normalize states by component-specific thresholds
        normalized_states = np.zeros_like(sampled_states, dtype=float)
        for i in range(C):
            comp_k = self.component_params[i]['k']
            normalized_states[i, :] = sampled_states[i, :] / comp_k
        
        # Create heatmap
        im = ax.imshow(normalized_states, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        
        # Add color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Degradation Level (% of threshold)')
        
        # Set labels and ticks
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Component')
        ax.set_title('Component Degradation Heatmap')
        
        # Set y-ticks (components)
        ax.set_yticks(range(C))
        ax.set_yticklabels([self.component_params[i]['name'] for i in range(C)])
        
        # Set x-ticks (time steps) - show fewer ticks if many steps
        if len(time_labels) > 20:
            tick_indices = np.linspace(0, len(time_labels)-1, 20, dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([time_labels[i] for i in tick_indices])
        else:
            ax.set_xticks(range(len(time_labels)))
            ax.set_xticklabels(time_labels)
        
        # Mark maintenance events
        for event in maintenance_events:
            if simulation_steps > 50:
                # Find nearest sample point
                event_idx = np.abs(sample_points - event).argmin()
                ax.axvline(x=event_idx, color='blue', linestyle='--', alpha=0.7)
            else:
                ax.axvline(x=event, color='blue', linestyle='--', alpha=0.7)
        
        # Add annotation for maintenance events
        if maintenance_events:
            ax.text(0.98, 0.02, 'Blue lines: Maintenance events', transform=ax.transAxes, 
                   color='blue', fontsize=10, ha='right', va='bottom', 
                   bbox=dict(facecolor='white', alpha=0.7))
        
        # Adjust layout
        fig.tight_layout()
        
        # Convert to base64 string for embedding in HTML
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return img_str

    def generate_cost_analysis_plot(self):
        """Generate cost analysis visualization"""
        if not self.simulation_results:
            return None
            
        costs = self.simulation_results['costs']
        maintenance_events = self.maintenance_events
        simulation_steps = self.simulation_steps
        
        # Create figure
        fig = Figure(figsize=(12, 10), dpi=100)
        
        # 1. Stacked bar chart for cost breakdown
        ax1 = fig.add_subplot(211)
        
        # Create time points
        time_points = np.arange(simulation_steps + 1)
        
        # Convert costs to numpy arrays for easier manipulation
        maintenance_costs = np.array(costs['maintenance'])
        transfer_costs = np.array(costs['transfer'])
        component_costs = np.array(costs['component'])
        shortage_costs = np.array(costs['shortage'])
        excess_costs = np.array(costs['excess'])
        
        # Plot cost components
        ax1.bar(time_points, maintenance_costs, label='Fixed Maintenance')
        ax1.bar(time_points, transfer_costs, bottom=maintenance_costs, label='Transfer')
        ax1.bar(time_points, component_costs, 
                bottom=maintenance_costs + transfer_costs, label='Component Replacement')
        ax1.bar(time_points, shortage_costs, 
                bottom=maintenance_costs + transfer_costs + component_costs, label='Shortage')
        ax1.bar(time_points, excess_costs, 
                bottom=maintenance_costs + transfer_costs + component_costs + shortage_costs, 
                label='Excess')
        
        # Mark maintenance events
        for event in maintenance_events:
            ax1.axvline(x=event, color='black', linestyle='--', alpha=0.3)
        
        # Set labels and title
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Cost')
        ax1.set_title('Cost Breakdown by Type')
        ax1.legend()
        
        # 2. Cumulative cost plot
        ax2 = fig.add_subplot(212)
        
        # Plot cumulative cost
        ax2.plot(time_points, costs['cumulative'], 'b-', linewidth=2)
        
        # Mark maintenance events
        for event in maintenance_events:
            ax2.axvline(x=event, color='r', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Cumulative Cost')
        ax2.set_title('Cumulative Cost Over Time')
        
        # Add cost summary text
        total_costs = {
            'Maintenance': np.sum(maintenance_costs),
            'Transfer': np.sum(transfer_costs),
            'Component': np.sum(component_costs),
            'Shortage': np.sum(shortage_costs),
            'Excess': np.sum(excess_costs),
            'Total': costs['cumulative'][-1]
        }
        
        summary_text = '\n'.join([f'{k}: {v:.2f}' for k, v in total_costs.items()])
        
        ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        fig.tight_layout()
        
        # Convert to base64 string for embedding in HTML
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return img_str

    def generate_signal_history_plot(self):
        """Generate signal history visualization with pie chart"""
        if not self.simulation_results:
            return None
            
        signal_history = np.array(self.simulation_results['signal_history'])
        
        # Create figure
        fig = Figure(figsize=(10, 8), dpi=100)
        
        # Create a single subplot for the pie chart
        ax = fig.add_subplot(111)
        
        # Count occurrences of each signal
        signal_counts = {
            'Green (0)': np.sum(signal_history == 0),
            'Yellow (1)': np.sum(signal_history == 1),
            'Red (2)': np.sum(signal_history == 2)
        }
        
        # Create labels and percentages
        labels = list(signal_counts.keys())
        sizes = list(signal_counts.values())
        
        # Calculate percentages
        total = sum(sizes)
        percentages = [100 * s / total for s in sizes]
        
        # Create customized labels to avoid overlapping
        # Just use percentages without text labels on the pie
        # The labels will be in the legend instead
        autopct_labels = ['%1.1f%%' % p for p in percentages]
        
        # Create pie chart
        colors = ['green', 'gold', 'red']
        wedges, texts, autotexts = ax.pie(sizes, colors=colors, 
                                         autopct='%1.1f%%',
                                         textprops={'fontsize': 12},
                                         startangle=90)
        
        # Customize text appearance to avoid overlapping
        for text in autotexts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
            
        # Create a legend outside the pie chart
        ax.legend(wedges, [f'{l} ({p:.1f}%)' for l, p in zip(labels, percentages)],
                loc='upper right', bbox_to_anchor=(1.0, 0.9))
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Add title
        ax.set_title('Signal Distribution', fontsize=16, pad=20)
        
        # Remove border
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add extra space around the pie chart
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Convert to base64 string for embedding in HTML
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return img_str

    def generate_maintenance_rules_plot(self):
        """Generate maintenance rules plot"""
        # Create figure
        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Draw the state machine
        ax.plot([0, 1], [0, 1], 'go-', markersize=15, label='Green')
        ax.plot([1, 2], [1, 1], 'yo-', markersize=15, label='Yellow')
        ax.plot([2, 3], [1, 0], 'ro-', markersize=15, label='Red')
        ax.plot([3, 0], [0, 0], 'ko--', alpha=0.5)
        
        # Add annotations
        ax.annotate('Start', xy=(0, 0), xytext=(0, -0.2), ha='center')
        ax.annotate('Degradation', xy=(0.5, 0.5), xytext=(0.5, 0.7), ha='center')
        ax.annotate(f'Consecutive Yellow\nCount >= {self.yellow_threshold}', xy=(1.5, 1), xytext=(1.5, 1.2), ha='center')
        ax.annotate('Component\nFailure', xy=(2.5, 0.5), xytext=(2.5, 0.7), ha='center')
        ax.annotate('Maintenance\n(Reset All Components)', xy=(1.5, 0), xytext=(1.5, -0.2), ha='center')
        
        # Set limits and remove axes
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add title
        ax.set_title('System State Transitions and Maintenance Policy')
        
        # Adjust layout
        fig.tight_layout()
        
        # Convert to base64 string for embedding in HTML
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return img_str

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if request.method == 'POST':
        # Get form data
        data = request.get_json()
        
        # Store parameters in session
        session['params'] = data
        
        # Create simulator instance
        simulator = MaintenanceOptimizationSimulator(data)
        
        # Update component params
        component_params = simulator.update_component_params()
        
        return jsonify({
            'status': 'success',
            'component_params': component_params
        })
    
    # For GET request, render setup form
    return render_template('setup.html')

@app.route('/component_params', methods=['POST'])
def component_params():
    data = request.get_json()
    
    # Update component parameters in session
    if 'params' in session:
        session['params']['component_params'] = data['component_params']
    
    return jsonify({'status': 'success'})

@app.route('/calculate_policy', methods=['POST'])
def calculate_policy():
    # Get parameters from session
    params = session.get('params', {})
    
    # Create simulator instance
    simulator = MaintenanceOptimizationSimulator(params)
    
    # Generate maintenance rules plot
    policy_plot = simulator.generate_maintenance_rules_plot()
    
    # Get maintenance rules text
    rules_text = f"""
The system follows these maintenance rules:

1. RED SIGNAL (Failure Detection):
   - If ANY component reaches its maximum deterioration level (K={simulator.K}), 
     the system emits a RED signal.
   - Immediate maintenance is performed, restoring all components to perfect condition.
   - All components ({simulator.C} total) are taken for the maintenance operation.

2. YELLOW SIGNAL (Degradation Detection):
   - If ANY component is degraded but none have failed, the system emits a YELLOW signal.
   - If the system remains in YELLOW state for {simulator.yellow_threshold} consecutive time steps,
     preventive maintenance is performed.
   - The yellow counter resets after each maintenance operation.

3. GREEN SIGNAL (Perfect Condition):
   - When ALL components are in perfect condition, the system emits a GREEN signal.
   - No maintenance action is taken in this state.

Cost Parameters:
- Preventive Maintenance (Yellow-triggered): {simulator.c1} units
- Corrective Maintenance (Red-triggered): {simulator.c2} units
- Transfer Cost per Component: {simulator.ct} units
- Replacement Cost per Component: {simulator.cr} units
- Shortage Cost per Component: {simulator.cs} units
- Excess Cost per Component: {simulator.ce} units

Degradation Model:
- Each component has a {(1-simulator.alpha)*100:.1f}% chance to degrade by one level each time step.
- Components degrade independently of each other.
"""
    
    # Store in session that policy was calculated
    session['policy_calculated'] = True
    
    return jsonify({
        'status': 'success',
        'rules_text': rules_text,
        'policy_plot': policy_plot
    })

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    # Get parameters from session
    params = session.get('params', {})
    
    # Create simulator instance
    simulator = MaintenanceOptimizationSimulator(params)
    
    # Run simulation
    simulation_data = simulator.run_simulation()
    
    # Generate plots
    plots = {
        'component_state': simulator.generate_component_state_plot(),
        'heatmap': simulator.generate_heatmap_plot(),
        'cost_analysis': simulator.generate_cost_analysis_plot(),
        'signal_history': simulator.generate_signal_history_plot()
    }
    
    # Add plots to simulation data
    simulation_data['plots'] = plots
    
    # Store simulation results in session
    session['simulation_results'] = simulation_data
    
    return jsonify({
        'status': 'success',
        'data': simulation_data
    })

@app.route('/reset', methods=['POST'])
def reset_simulation():
    # Clear session data
    session.pop('params', None)
    session.pop('simulation_results', None)
    session.pop('policy_calculated', None)
    
    return jsonify({'status': 'success'})

@app.route('/visualization')
def visualization():
    # Check if simulation has been run
    if 'simulation_results' not in session:
        return render_template('visualization.html', no_data=True)
    
    # Get simulation results from session
    simulation_results = session.get('simulation_results', {})
    
    return render_template('visualization.html', 
                          simulation_results=simulation_results,
                          no_data=False)

@app.route('/policy')
def policy():
    # Check if policy has been calculated
    if not session.get('policy_calculated', False):
        return render_template('policy.html', no_policy=True)
    
    # Get parameters from session
    params = session.get('params', {})
    
    # Create simulator instance
    simulator = MaintenanceOptimizationSimulator(params)
    
    # Generate maintenance rules plot
    policy_plot = simulator.generate_maintenance_rules_plot()
    
    # Get maintenance rules text
    rules_text = f"""
The system follows these maintenance rules:

1. RED SIGNAL (Failure Detection):
   - If ANY component reaches its maximum deterioration level (K={simulator.K}), 
     the system emits a RED signal.
   - Immediate maintenance is performed, restoring all components to perfect condition.
   - All components ({simulator.C} total) are taken for the maintenance operation.

2. YELLOW SIGNAL (Degradation Detection):
   - If ANY component is degraded but none have failed, the system emits a YELLOW signal.
   - If the system remains in YELLOW state for {simulator.yellow_threshold} consecutive time steps,
     preventive maintenance is performed.
   - The yellow counter resets after each maintenance operation.

3. GREEN SIGNAL (Perfect Condition):
   - When ALL components are in perfect condition, the system emits a GREEN signal.
   - No maintenance action is taken in this state.

Cost Parameters:
- Preventive Maintenance (Yellow-triggered): {simulator.c1} units
- Corrective Maintenance (Red-triggered): {simulator.c2} units
- Transfer Cost per Component: {simulator.ct} units
- Replacement Cost per Component: {simulator.cr} units
- Shortage Cost per Component: {simulator.cs} units
- Excess Cost per Component: {simulator.ce} units

Degradation Model:
- Each component has a {(1-simulator.alpha)*100:.1f}% chance to degrade by one level each time step.
- Components degrade independently of each other.
"""
    
    return render_template('policy.html', 
                          no_policy=False,
                          rules_text=rules_text,
                          policy_plot=policy_plot)

if __name__ == '__main__':
    app.run(debug=True)