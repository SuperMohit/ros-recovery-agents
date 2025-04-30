import numpy as np
import random
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set, Optional, Union

class AdvancedMaintenanceEnvironment:
    """
    Environment for multi-agent maintenance robot system modeled as an MDP.
    Includes realistic fault types for industrial maintenance applications.
    """
    
    def __init__(self, num_robots: int, num_machines: int, grid_size: int = 10):
        """
        Initialize the maintenance environment.
        
        Args:
            num_robots: Number of maintenance robots
            num_machines: Number of machines to maintain
            grid_size: Size of the grid environment (grid_size x grid_size)
        """
        self.num_robots = num_robots
        self.num_machines = num_machines
        self.grid_size = grid_size
        
        # Initialize robot state
        self.robot_locations = np.zeros((num_robots, 2), dtype=int)  # (x, y) coordinates
        self.battery_levels = np.ones(num_robots) * 100  # Full battery (100%)
        self.robot_status = ["idle"] * num_robots
        self.robot_task_timers = [0] * num_robots
        self.current_assignments = [-1] * num_robots  # -1 means no assignment
        
        # Initialize robot fault states
        self.robot_faults = {
            # Power/battery faults
            "battery_degradation": np.zeros(num_robots),  # 0-1 severity
            "power_brownout_prob": np.zeros(num_robots),  # probability of brownout
            
            # Communication faults
            "comm_delay": np.zeros(num_robots),  # seconds of delay
            "packet_loss_prob": np.zeros(num_robots),  # probability of packet loss
        }
        
        # Initialize machine state
        self.machine_locations = np.zeros((num_machines, 2), dtype=int)
        self.machine_states = np.zeros(num_machines, dtype=int)  # 0=operational, 1=needs maintenance, 2=broken
        self.maintenance_queue = []  # Priority queue for maintenance
        
        # Initialize detailed machine fault states
        self.machine_faults = {
            # Motor/Actuator Faults
            "motor_wear": np.zeros(num_machines),  # 0-1 severity
            "motor_instability": np.zeros(num_machines),  # 0-1 severity
            "joint_backlash": np.zeros(num_machines),  # 0-1 severity
            
            # Sensor Faults
            "sensor_drift": np.zeros(num_machines),  # 0-1 severity
            "sensor_noise": np.zeros(num_machines),  # 0-1 severity
            "sensor_dropout_prob": np.zeros(num_machines),  # probability of dropout
        }
        
        # Fault groups for easier tracking
        self.fault_groups = {
            "motor": ["motor_wear", "motor_instability", "joint_backlash"],
            "sensor": ["sensor_drift", "sensor_noise", "sensor_dropout_prob"],
            "power": ["battery_degradation", "power_brownout_prob"],
            "comm": ["comm_delay", "packet_loss_prob"]
        }
        
        # Fault threshold for maintenance need
        self.fault_threshold = 0.5  # When any fault exceeds this, machine state goes to 1
        self.critical_threshold = 0.8  # When any fault exceeds this, risk of breakdown increases
        
        # Place robots and machines randomly
        self._initialize_locations()
        
        # Charging station location
        self.charging_station = (grid_size // 2, grid_size // 2)
        
        # Task durations (in time steps)
        self.task_durations = {
            "inspect": 1,
            "analyze_motor": 2,
            "analyze_sensor": 2,
            "analyze_power": 2,
            "analyze_comm": 2,
            "repair_motor": 4,
            "repair_sensor": 3,
            "repair_power": 3,
            "repair_comm": 2,
            "replace_part": 5
        }
        
        # Repair success probabilities
        self.repair_success_prob = {
            "repair_motor": 0.85,
            "repair_sensor": 0.9,
            "repair_power": 0.8,
            "repair_comm": 0.95,
            "replace_part": 0.98
        }
        
        # Failure probabilities
        self.move_failure_prob = 0.1
        
        # Fault development rates
        self.fault_development_rate = 0.01  # Base rate for fault severity increase
        
        # Time tracking
        self.time_step = 0
        
        # History tracking for visualization
        self.history = {
            "machine_states": [],
            "battery_levels": [],
            "rewards": [],
            "fault_severities": []
        }
    
    def _initialize_locations(self):
        """Initialize random locations for robots and machines."""
        # Generate random positions for robots
        for i in range(self.num_robots):
            self.robot_locations[i] = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            ]
        
        # Generate random positions for machines
        for i in range(self.num_machines):
            self.machine_locations[i] = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            ]
    
    def _update_machine_states(self):
        """Update machine states based on fault severities."""
        for i in range(self.num_machines):
            # Check if any fault exceeds the maintenance threshold
            needs_maintenance = False
            critical_faults = False
            
            # Check motor faults
            for fault in self.fault_groups["motor"]:
                if self.machine_faults[fault][i] >= self.fault_threshold:
                    needs_maintenance = True
                if self.machine_faults[fault][i] >= self.critical_threshold:
                    critical_faults = True
            
            # Check sensor faults
            for fault in self.fault_groups["sensor"]:
                if self.machine_faults[fault][i] >= self.fault_threshold:
                    needs_maintenance = True
                if self.machine_faults[fault][i] >= self.critical_threshold:
                    critical_faults = True
            
            # Update machine state based on faults
            if self.machine_states[i] == 0 and needs_maintenance:
                self.machine_states[i] = 1  # Needs maintenance
                if i not in self.maintenance_queue:
                    self.maintenance_queue.append(i)
            
            # If critical faults exist, there's a chance of breakdown
            if critical_faults and self.machine_states[i] == 1:
                if random.random() < 0.05 * sum(self.machine_faults[fault][i] for fault in self.fault_groups["motor"] + self.fault_groups["sensor"]):
                    self.machine_states[i] = 2  # Broken
    
    def _update_fault_severities(self):
        """Gradually increase fault severities based on machine usage and time."""
        # Increase motor wear based on machine usage (simplified)
        for i in range(self.num_machines):
            if self.machine_states[i] == 0:  # If operational
                # Motor faults development
                for fault in self.fault_groups["motor"]:
                    # More wear for operational machines with existing wear
                    increase = self.fault_development_rate * (1 + self.machine_faults[fault][i])
                    self.machine_faults[fault][i] = min(1.0, self.machine_faults[fault][i] + increase * random.uniform(0.5, 1.5))
                
                # Sensor faults development
                for fault in self.fault_groups["sensor"]:
                    increase = self.fault_development_rate * random.uniform(0.3, 1.2)
                    self.machine_faults[fault][i] = min(1.0, self.machine_faults[fault][i] + increase)
        
        # Update robot fault states
        for i in range(self.num_robots):
            # Battery degradation increases with usage
            if self.robot_status[i] != "idle" and self.robot_status[i] != "charging":
                self.robot_faults["battery_degradation"][i] = min(1.0, 
                    self.robot_faults["battery_degradation"][i] + self.fault_development_rate * 0.5)
            
            # Communication faults fluctuate randomly
            if random.random() < 0.05:  # 5% chance of communication degradation
                self.robot_faults["comm_delay"][i] = min(1.0, 
                    self.robot_faults["comm_delay"][i] + random.uniform(0, 0.1))
                self.robot_faults["packet_loss_prob"][i] = min(1.0, 
                    self.robot_faults["packet_loss_prob"][i] + random.uniform(0, 0.05))
    
    def get_state(self) -> dict:
        """Return the current state of the environment."""
        return {
            "robot_locations": self.robot_locations.copy(),
            "machine_states": self.machine_states.copy(),
            "battery_levels": self.battery_levels.copy(),
            "maintenance_queue": self.maintenance_queue.copy(),
            "machine_faults": {k: v.copy() for k, v in self.machine_faults.items()},
            "robot_faults": {k: v.copy() for k, v in self.robot_faults.items()},
            "robot_status": self.robot_status.copy(),
            "time_step": self.time_step,
            "current_assignments": self.current_assignments.copy()
        }
    
    def get_available_actions(self, robot_id: int) -> List[Tuple[str, Union[int, Tuple[int, int]]]]:
        """
        Get available actions for a specific robot.
        
        Returns:
            List of (action_type, target_id) tuples
        """
        actions = []
        
        # If robot is busy, only continue action is available
        if self.robot_status[robot_id] != "idle" and self.robot_task_timers[robot_id] > 0:
            return [("continue", -1)]
        
        # Movement actions - can move to any machine
        for machine_id in range(self.num_machines):
            actions.append(("move", machine_id))
        
        # Charging action
        actions.append(("charge", -1))
        
        # Current location of the robot
        robot_loc = tuple(self.robot_locations[robot_id])
        
        # Check if robot is at a machine location
        for machine_id in range(self.num_machines):
            machine_loc = tuple(self.machine_locations[machine_id])
            if robot_loc == machine_loc:
                # Basic inspection
                actions.append(("inspect", machine_id))
                
                # Detailed analysis actions
                actions.append(("analyze_motor", machine_id))
                actions.append(("analyze_sensor", machine_id))
                actions.append(("analyze_power", machine_id))
                actions.append(("analyze_comm", machine_id))
                
                # Repair actions
                actions.append(("repair_motor", machine_id))
                actions.append(("repair_sensor", machine_id))
                actions.append(("repair_power", machine_id))
                actions.append(("repair_comm", machine_id))
                
                # Complete replacement
                actions.append(("replace_part", machine_id))
        
        # Communication/delegation actions - can delegate any task to any other robot
        for other_robot in range(self.num_robots):
            if other_robot != robot_id:
                for machine_id in range(self.num_machines):
                    # Can delegate inspection tasks
                    actions.append(("delegate", (other_robot, machine_id, "inspect")))
                    
                    # Can delegate repair tasks if machine needs repair
                    if self.machine_states[machine_id] > 0:
                        actions.append(("delegate", (other_robot, machine_id, "repair_motor")))
                        actions.append(("delegate", (other_robot, machine_id, "repair_sensor")))
        
        return actions
    
    def step(self, actions: List[Tuple[str, Union[int, Tuple[int, int, str]]]]) -> Tuple[dict, List[float], bool]:
        """
        Take a step in the environment based on joint actions.
        
        Args:
            actions: List of (action_type, target_id) tuples for each robot
            
        Returns:
            new_state, rewards, done
        """
        self.time_step += 1
        rewards = [0] * self.num_robots
        
        # Update all fault severities gradually
        self._update_fault_severities()
        
        # Process each robot's action
        for robot_id, (action_type, target) in enumerate(actions):
            # Apply communication faults
            if self.robot_faults["packet_loss_prob"][robot_id] > 0:
                if random.random() < self.robot_faults["packet_loss_prob"][robot_id]:
                    # Action is lost - robot does nothing this turn
                    rewards[robot_id] -= 15  # Penalty for packet loss
                    continue
            
            # Add communication delay (simplified)
            if self.robot_faults["comm_delay"][robot_id] > 0.5:
                # Significant delay might cause the robot to miss this step
                if random.random() < self.robot_faults["comm_delay"][robot_id] * 0.3:
                    rewards[robot_id] -= 5  # Small penalty for delay
                    continue
            
            # Check for power brownout
            if random.random() < self.robot_faults["power_brownout_prob"][robot_id]:
                rewards[robot_id] -= 20  # Penalty for brownout
                # Skip action and possibly reset robot status
                if random.random() < 0.5:  # 50% chance of resetting task
                    self.robot_status[robot_id] = "idle"
                    self.robot_task_timers[robot_id] = 0
                continue
            
            # Skip if robot is busy with a task
            if self.robot_status[robot_id] != "idle" and self.robot_task_timers[robot_id] > 0:
                self.robot_task_timers[robot_id] -= 1
                if self.robot_task_timers[robot_id] == 0:
                    # Task completed
                    task = self.robot_status[robot_id]
                    machine_id = self.current_assignments[robot_id]
                    
                    self.robot_status[robot_id] = "idle"
                    
                    # Handle completion of repair tasks
                    if task in ["repair_motor", "repair_sensor", "repair_power", "repair_comm", "replace_part"] and machine_id >= 0:
                        success_prob = self.repair_success_prob.get(task, 0.8)
                        
                        # Battery degradation affects repair success
                        success_prob *= (1 - self.robot_faults["battery_degradation"][robot_id] * 0.3)
                        
                        if random.random() < success_prob:
                            # Successful repair
                            if task == "repair_motor":
                                for fault in self.fault_groups["motor"]:
                                    self.machine_faults[fault][machine_id] = max(0, self.machine_faults[fault][machine_id] - 0.7)
                                rewards[robot_id] += 120
                            
                            elif task == "repair_sensor":
                                for fault in self.fault_groups["sensor"]:
                                    self.machine_faults[fault][machine_id] = max(0, self.machine_faults[fault][machine_id] - 0.7)
                                rewards[robot_id] += 100
                            
                            elif task == "repair_power":
                                rewards[robot_id] += 110
                            
                            elif task == "repair_comm":
                                rewards[robot_id] += 90
                            
                            elif task == "replace_part":
                                # Reset all faults for this machine
                                for fault_group in self.fault_groups.values():
                                    for fault in fault_group:
                                        if fault in self.machine_faults:
                                            self.machine_faults[fault][machine_id] = 0
                                rewards[robot_id] += 150
                        else:
                            # Failed repair attempt
                            rewards[robot_id] -= 20
                    
                    self.current_assignments[robot_id] = -1
                continue
            
            # Process action based on type
            if action_type == "move":
                machine_id = target
                # Calculate distance to target machine
                target_loc = self.machine_locations[machine_id]
                distance = np.linalg.norm(self.robot_locations[robot_id] - target_loc)
                
                # Movement has a chance of failure
                if random.random() > self.move_failure_prob:
                    # Move towards the target (simplified - direct movement)
                    self.robot_locations[robot_id] = target_loc
                    rewards[robot_id] -= 1  # Small cost for movement
                else:
                    rewards[robot_id] -= 2  # Penalty for failed movement
                
                # Battery consumption based on distance and degradation
                battery_factor = 1 + self.robot_faults["battery_degradation"][robot_id]
                self.battery_levels[robot_id] -= min(5, distance) * battery_factor
            
            elif action_type == "charge":
                # Move towards charging station if not already there
                if not np.array_equal(self.robot_locations[robot_id], self.charging_station):
                    self.robot_locations[robot_id] = np.array(self.charging_station)
                    rewards[robot_id] -= 1  # Cost to move to charging station
                
                # Charge battery - affected by degradation
                self.robot_status[robot_id] = "charging"
                charge_efficiency = 1 - self.robot_faults["battery_degradation"][robot_id] * 0.5
                old_level = self.battery_levels[robot_id]
                self.battery_levels[robot_id] = min(100, self.battery_levels[robot_id] + 20 * charge_efficiency)
                rewards[robot_id] += (self.battery_levels[robot_id] - old_level) * 0.1  # Small reward for charging
            
            elif action_type == "inspect":
                machine_id = target
                # Check if robot is at the machine
                if np.array_equal(self.robot_locations[robot_id], self.machine_locations[machine_id]):
                    # Start inspection
                    self.robot_status[robot_id] = "inspecting"
                    self.robot_task_timers[robot_id] = self.task_durations["inspect"]
                    self.current_assignments[robot_id] = machine_id
                    
                    # If machine needs maintenance, reward for finding it
                    if self.machine_states[machine_id] > 0:
                        rewards[robot_id] += 30
                    else:
                        faults_detected = 0
                        for fault_group in self.fault_groups.values():
                            for fault in fault_group:
                                if fault in self.machine_faults and self.machine_faults[fault][machine_id] > 0.3:
                                    faults_detected += 1
                        rewards[robot_id] += faults_detected * 10  # Reward for each early fault detection
                    
                    # Battery consumption
                    self.battery_levels[robot_id] -= 2 * (1 + self.robot_faults["battery_degradation"][robot_id] * 0.5)
                else:
                    rewards[robot_id] -= 5  # Penalty for invalid action
            
            elif action_type.startswith("analyze_"):
                machine_id = target
                fault_type = action_type.split("_")[1]  # motor, sensor, power, comm
                
                # Check if robot is at the machine
                if np.array_equal(self.robot_locations[robot_id], self.machine_locations[machine_id]):
                    # Start analysis
                    self.robot_status[robot_id] = f"analyzing_{fault_type}"
                    self.robot_task_timers[robot_id] = self.task_durations[action_type]
                    self.current_assignments[robot_id] = machine_id
                    
                    # Fault detection reward
                    faults_detected = 0
                    
                    if fault_type in self.fault_groups:
                        for fault in self.fault_groups[fault_type]:
                            if fault in self.machine_faults and self.machine_faults[fault][machine_id] > self.fault_threshold * 0.7:
                                faults_detected += 1
                    
                    rewards[robot_id] += faults_detected * 50  # Reward for each fault detected
                    
                    # Battery consumption
                    self.battery_levels[robot_id] -= 3 * (1 + self.robot_faults["battery_degradation"][robot_id] * 0.5)
                else:
                    rewards[robot_id] -= 5  # Penalty for invalid action
            
            elif action_type.startswith("repair_"):
                machine_id = target
                fault_type = action_type.split("_")[1]  # motor, sensor, power, comm
                
                # Check if robot is at the machine
                if np.array_equal(self.robot_locations[robot_id], self.machine_locations[machine_id]):
                    # Start repair
                    self.robot_status[robot_id] = action_type + "ing"
                    self.robot_task_timers[robot_id] = self.task_durations[action_type]
                    self.current_assignments[robot_id] = machine_id
                    
                    # Battery consumption
                    self.battery_levels[robot_id] -= 8 * (1 + self.robot_faults["battery_degradation"][robot_id] * 0.5)
                else:
                    rewards[robot_id] -= 5  # Penalty for invalid action
            
            elif action_type == "replace_part":
                machine_id = target
                # Check if robot is at the machine
                if np.array_equal(self.robot_locations[robot_id], self.machine_locations[machine_id]):
                    # Start replacement
                    self.robot_status[robot_id] = "replacing"
                    self.robot_task_timers[robot_id] = self.task_durations["replace_part"]
                    self.current_assignments[robot_id] = machine_id
                    
                    # Battery consumption - highest for full replacement
                    self.battery_levels[robot_id] -= 15 * (1 + self.robot_faults["battery_degradation"][robot_id] * 0.5)
                else:
                    rewards[robot_id] -= 5  # Penalty for invalid action
            
            elif action_type == "delegate":
                other_robot, machine_id, task = target
                # Check if other robot is available
                if self.robot_status[other_robot] == "idle" and self.current_assignments[other_robot] == -1:
                    # Check if delegation succeeds (affected by communication faults)
                    comm_success_prob = 1 - self.robot_faults["packet_loss_prob"][robot_id] * 0.7
                    if random.random() < comm_success_prob:
                        self.current_assignments[other_robot] = machine_id
                        rewards[robot_id] += 10  # Reward for successful delegation
                    else:
                        rewards[robot_id] -= 5  # Penalty for failed delegation
                else:
                    rewards[robot_id] -= 5  # Penalty for invalid delegation
            
            # Common battery and fault effects
            # Low battery penalty
            if self.battery_levels[robot_id] < 10:
                rewards[robot_id] -= 10
            
            # Dead battery
            if self.battery_levels[robot_id] <= 0:
                self.battery_levels[robot_id] = 0
                self.robot_status[robot_id] = "out_of_battery"
                rewards[robot_id] -= 50  # Big penalty for running out of battery
        
        # Update machine states based on fault severities
        self._update_machine_states()
        
        # Check for newly broken machines
        for machine_id in range(self.num_machines):
            if self.machine_states[machine_id] == 2:  # broken
                # Penalty for all robots when a machine breaks
                for robot_id in range(self.num_robots):
                    rewards[robot_id] -= 100 / self.num_robots
        
        # Update history for visualization
        self.history["machine_states"].append(self.machine_states.copy())
        self.history["battery_levels"].append(self.battery_levels.copy())
        self.history["rewards"].append(rewards)
        
        # Track fault severities for visualization
        fault_snapshot = {}
        for fault_type, machines in self.machine_faults.items():
            fault_snapshot[fault_type] = machines.copy()
        self.history["fault_severities"].append(fault_snapshot)
        
        # Check if simulation should end
        done = self.time_step >= 100 or all(status == "out_of_battery" for status in self.robot_status)
        
        return self.get_state(), rewards, done
    
    def render(self, mode='console'):
        """
        Render the environment.
        """
        if mode == 'console':
            print(f"Time step: {self.time_step}")
            print("Robot locations:", self.robot_locations)
            print("Robot battery levels:", self.battery_levels)
            print("Robot status:", self.robot_status)
            print("Machine states:", self.machine_states)
            
            # Display fault information
            print("\nMachine Fault Severities:")
            for machine_id in range(self.num_machines):
                print(f"  Machine {machine_id}:")
                for fault_group, faults in self.fault_groups.items():
                    if fault_group in ["motor", "sensor"]:
                        print(f"    {fault_group.capitalize()} faults:")
                        for fault in faults:
                            if fault in self.machine_faults:
                                severity = self.machine_faults[fault][machine_id]
                                status = "OK" if severity < 0.3 else "Warning" if severity < 0.5 else "Critical" if severity < 0.8 else "Failure"
                                print(f"      {fault}: {severity:.2f} ({status})")
            
            print("\nRobot Fault States:")
            for robot_id in range(self.
