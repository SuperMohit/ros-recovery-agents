# IndustrialMind: Technical Specification

## LangGraph-Based Agent Architecture

### 1. Introduction

This document outlines the technical specifications for the IndustrialMind system's AI architecture, focusing on the LangGraph-based agent design. The system leverages our experience with drone AI surveillance to create an intelligent multi-agent maintenance solution for industrial environments.

### 2. System Architecture Overview

IndustrialMind uses a hierarchical architecture with two primary AI layers:

1. **Individual Maintenance Robot Agents**: Autonomous decision-making units deployed on each maintenance robot
2. **Central Coordinator Agent**: Strategic orchestration system that coordinates the fleet

Both layers are implemented using LangGraph, with specialized node designs for maintenance operations and multi-agent coordination.

## 3. Individual Robot Agent Architecture

### 3.1 State Management

```typescript
interface RobotState {
  // Identity and Status
  robotId: string;
  batteryLevel: number;  // 0-100%
  location: {x: number, y: number, z: number};
  currentStatus: 'idle' | 'moving' | 'diagnosing' | 'repairing' | 'charging';
  toolsAvailable: string[];
  specializations: string[];  // e.g., ['motor_expert', 'sensor_expert']
  
  // Task Management
  currentTask: Task | null;
  assignedMachines: string[];
  taskHistory: TaskResult[];
  
  // Environment Awareness
  nearbyMachines: Machine[];
  nearbyRobots: RobotInfo[];
  obstacleMap: ObstacleInfo[];
  
  // Internal State
  faultDetectionConfidence: number;  // 0-1
  repairSuccessProbability: number;  // 0-1
  internalDiagnostics: ComponentStatus[];
  
  // Communication
  lastCommunicationTime: number;
  messageQueue: Message[];
  delegationRequests: DelegationRequest[];
}

interface Machine {
  machineId: string;
  type: string;
  location: {x: number, y: number, z: number};
  currentState: 'operational' | 'maintenance_needed' | 'broken';
  detectedFaults: Fault[];
  maintenanceHistory: MaintenanceRecord[];
  priority: number;  // 0-100
}

interface Fault {
  faultId: string;
  type: 'motor' | 'sensor' | 'power' | 'communication' | 'structural' | 'fluid';
  severity: number;  // 0-1
  detectionConfidence: number;  // 0-1
  estimatedRepairTime: number;  // minutes
  repairToolsRequired: string[];
  timeDetected: number;
  description: string;
}
```

### 3.2 LangGraph Node Structure

```
┌───────────────────┐
│                   │
│  Perception       │───┐
│                   │   │
└───────────────────┘   │
                        │
┌───────────────────┐   │
│                   │   │
│  Status Update    │◄──┘
│                   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│                   │
│  Task Selection   │◄──────────┐
│                   │           │
└────────┬──────────┘           │
         │                      │
         ▼                      │
┌───────────────────┐           │
│                   │           │
│  Action Planning  │           │
│                   │           │
└────────┬──────────┘           │
         │                      │
         ▼                      │
┌───────────────────┐           │
│                   │           │
│ Execution Manager │           │
│                   │           │
└────────┬──────────┘           │
         │                      │
         ▼                      │
┌────────────────────┐          │
│                    │          │
│ Outcome Evaluation │──────────┘
│                    │
└────────┬───────────┘
         │
         ▼
┌───────────────────┐
│                   │
│ Knowledge Update  │
│                   │
└───────────────────┘
```

### 3.3 Node Descriptions

#### 3.3.1 Perception Node
- **Input**: Sensor data, environment state, machine telemetry
- **Processing**:
  - Sensor fusion for machine state evaluation
  - Anomaly detection in machine operation
  - Environmental awareness processing
  - Detection of nearby robots and machines
- **Output**: Processed environment state, detected anomalies

```python
def perception_node(state: AgentState) -> Dict:
    """Process sensor data and environment information."""
    # Process raw sensor data
    processed_sensor_data = process_sensors(state.raw_sensor_data)
    
    # Detect machine anomalies
    anomalies = detect_anomalies(processed_sensor_data, state.machine_models)
    
    # Update environment awareness
    environment_state = update_environment_map(
        state.current_map,
        processed_sensor_data.lidar,
        processed_sensor_data.camera
    )
    
    # Detect nearby robots and machines
    nearby_entities = detect_nearby_entities(
        environment_state,
        state.robot_position
    )
    
    return {
        "processed_sensor_data": processed_sensor_data,
        "detected_anomalies": anomalies,
        "environment_state": environment_state,
        "nearby_entities": nearby_entities
    }
```

#### 3.3.2 Status Update Node
- **Input**: Perception output, robot internal state
- **Processing**:
  - Battery level assessment
  - Tool and capability inventory
  - Self-diagnostic evaluation
  - Communication status check
- **Output**: Updated robot status

#### 3.3.3 Task Selection Node
- **Input**: Updated robot status, machine conditions, task queue
- **Processing**:
  - Priority calculation for machine maintenance
  - Task feasibility assessment based on robot capabilities
  - Resource requirement estimation
  - Coordination with other robots via central coordinator
- **Output**: Selected task with justification

```python
def task_selection_node(state: AgentState) -> Dict:
    """Select the highest priority task for the robot."""
    # Get current task queue
    available_tasks = get_available_tasks(
        state.machine_states,
        state.assigned_machines,
        state.task_queue
    )
    
    # Calculate priority for each task
    prioritized_tasks = []
    for task in available_tasks:
        priority_score = calculate_priority(
            task,
            state.robot_status,
            state.battery_level,
            state.machine_criticality
        )
        prioritized_tasks.append((task, priority_score))
    
    # Sort by priority
    prioritized_tasks.sort(key=lambda x: x[1], reverse=True)
    
    # Select highest priority feasible task
    selected_task = None
    task_justification = ""
    
    for task, score in prioritized_tasks:
        feasibility = assess_task_feasibility(
            task,
            state.robot_capabilities,
            state.battery_level,
            state.tool_inventory
        )
        
        if feasibility.feasible:
            selected_task = task
            task_justification = generate_justification(task, score, feasibility)
            break
    
    return {
        "selected_task": selected_task,
        "task_justification": task_justification,
        "task_priority_score": score if selected_task else 0,
        "alternative_tasks": prioritized_tasks[:3]  # Keep top 3 alternatives
    }
```

#### 3.3.4 Action Planning Node
- **Input**: Selected task, machine state, robot capabilities
- **Processing**:
  - Break down task into action sequence
  - Determine resource requirements
  - Calculate expected duration
  - Plan for contingencies
- **Output**: Detailed action plan

#### 3.3.5 Execution Manager Node
- **Input**: Action plan, robot state, real-time sensor data
- **Processing**:
  - Step-by-step action execution
  - Real-time monitoring of execution progress
  - Adaptation to unexpected conditions
  - Safety checking and constraint enforcement
- **Output**: Execution results, action outcomes

```python
def execution_manager_node(state: AgentState) -> Dict:
    """Execute the current action plan and monitor progress."""
    # Get current action plan
    action_plan = state.current_action_plan
    current_step = state.execution_step
    
    if not action_plan or current_step >= len(action_plan):
        return {
            "execution_complete": True,
            "execution_results": state.execution_results,
            "status": "completed"
        }
    
    # Get the current action to execute
    current_action = action_plan[current_step]
    
    # Execute the action
    try:
        action_result = execute_action(
            current_action,
            state.robot_controllers,
            state.sensor_data
        )
        
        # Monitor execution and check for unexpected conditions
        execution_status = monitor_execution(
            current_action,
            action_result,
            state.expected_outcomes,
            state.safety_thresholds
        )
        
        # Update execution results
        updated_results = state.execution_results.copy()
        updated_results.append({
            "action": current_action,
            "result": action_result,
            "status": execution_status.status
        })
        
        # Determine if we should proceed to the next step
        next_step = current_step + 1 if execution_status.continue_execution else current_step
        
        return {
            "execution_complete": False,
            "execution_results": updated_results,
            "execution_step": next_step,
            "status": execution_status.status,
            "adaptations": execution_status.adaptations
        }
    except Exception as e:
        # Handle execution errors
        return {
            "execution_complete": False,
            "execution_results": state.execution_results,
            "execution_error": str(e),
            "status": "error"
        }
```

#### 3.3.6 Outcome Evaluation Node
- **Input**: Execution results, expected outcomes, machine state
- **Processing**:
  - Compare results with expected outcomes
  - Evaluate repair/diagnosis success
  - Determine if additional actions are needed
  - Calculate rewards based on MDP reward function
- **Output**: Evaluation results, next steps recommendation

#### 3.3.7 Knowledge Update Node
- **Input**: Evaluation results, task history, environment changes
- **Processing**:
  - Update internal knowledge base
  - Refine fault detection models
  - Update repair success predictions
  - Share learnings with central coordinator
- **Output**: Updated knowledge base, shared insights

### 3.4 Decision Making Prompts

Each LangGraph node uses specialized prompts tailored to maintenance operations. Examples:

#### Task Selection Prompt

```
You are the task selection system for an autonomous maintenance robot. 
Your job is to select the most appropriate maintenance task based on:

1. Machine states and detected faults
2. Robot capabilities and current status
3. Overall maintenance priorities
4. Fleet-wide coordination considerations

Current Robot Status:
- ID: {{robot_id}}
- Battery Level: {{battery_level}}%
- Location: {{robot_location}}
- Available Tools: {{available_tools}}
- Specializations: {{specializations}}

Detected Machine Issues:
{{machine_issues_table}}

Fleet Status:
{{fleet_status_summary}}

Task Queue:
{{task_queue}}

Based on this information, select the most appropriate task for this robot. 
Consider:
- Task urgency and machine criticality
- Robot's proximity to machines
- Required tools and specializations
- Battery level sufficiency for task completion
- Ongoing work by other robots
- Potential for coordination

Provide your selection and detailed reasoning.
```

#### Diagnostic Planning Prompt

```
You are the diagnostic planning system for an autonomous maintenance robot.
Your job is to create a detailed diagnostic plan to investigate a potential machine fault.

Machine Information:
- ID: {{machine_id}}
- Type: {{machine_type}}
- Reported Symptoms: {{symptoms}}
- Maintenance History: {{maintenance_history}}

Detected Anomaly:
- Type: {{anomaly_type}}
- Confidence: {{detection_confidence}}%
- Sensor Readings: {{sensor_readings}}

Available Diagnostic Tools:
{{available_tools}}

Create a step-by-step diagnostic plan that:
1. Starts with non-invasive tests
2. Progressively narrows down the potential causes
3. Efficiently uses available diagnostic tools
4. Minimizes unnecessary steps
5. Includes decision points based on interim findings
6. Estimates the time required for each step

Your plan should be detailed enough for the robot to execute without further clarification.
```

### 3.5 Action Space Implementation

The action space from the MDP model is implemented as a set of executable functions:

```python
class ActionExecutionSystem:
    def __init__(self, robot_controllers, sensors, tools):
        self.robot_controllers = robot_controllers
        self.sensors = sensors
        self.tools = tools
        self.action_history = []
    
    def move_to(self, target_location, speed_factor=1.0):
        """Move robot to the specified location."""
        path = self.plan_path(self.robot_controllers.get_position(), target_location)
        return self.robot_controllers.follow_path(path, speed_factor)
    
    def inspect_machine(self, machine_id, inspection_level='standard'):
        """Perform visual and sensor-based inspection of a machine."""
        machine_location = self.get_machine_location(machine_id)
        
        # Position robot optimally for inspection
        self.position_for_inspection(machine_location)
        
        # Activate sensors based on inspection level
        sensor_data = self.sensors.collect_multi_sensor_data(
            inspection_level,
            machine_id
        )
        
        # Process and analyze the data
        inspection_results = self.analyze_inspection_data(sensor_data, machine_id)
        
        return inspection_results
    
    def analyze_component(self, machine_id, component_type, component_id):
        """Perform detailed analysis of a specific machine component."""
        # Select appropriate tools for analysis
        analysis_tools = self.select_analysis_tools(component_type)
        
        # Position robot for component access
        self.position_for_component_access(machine_id, component_id)
        
        # Deploy tools and collect data
        analysis_data = self.tools.collect_component_data(
            machine_id,
            component_id,
            analysis_tools
        )
        
        # Analyze the collected data
        analysis_results = self.process_component_analysis(analysis_data, component_type)
        
        return analysis_results
    
    def repair_component(self, machine_id, component_id, repair_procedure):
        """Execute repair procedure on a machine component."""
        # Prepare repair tools
        repair_tools = self.prepare_repair_tools(repair_procedure)
        
        # Position robot for repair access
        self.position_for_repair_access(machine_id, component_id)
        
        # Execute the repair procedure
        repair_result = self.tools.execute_repair_procedure(
            machine_id,
            component_id,
            repair_procedure,
            repair_tools
        )
        
        # Verify repair success
        verification_result = self.verify_repair(machine_id, component_id)
        
        return {
            "repair_result": repair_result,
            "verification": verification_result
        }
    
    def charge_battery(self):
        """Navigate to charging station and recharge battery."""
        # Find nearest charging station
        charging_station = self.find_nearest_charging_station()
        
        # Navigate to charging station
        self.move_to(charging_station.location)
        
        # Initiate charging
        charging_result = self.robot_controllers.initiate_charging()
        
        return charging_result
    
    def communicate(self, target_robot_id, message_type, message_content):
        """Send communication to another robot or central system."""
        # Format message
        message = self.format_message(target_robot_id, message_type, message_content)
        
        # Send message through communication system
        send_result = self.robot_controllers.send_message(message)
        
        return send_result
```

## 4. Central Coordinator Architecture

### 4.1 State Management

```typescript
interface CoordinatorState {
  // Fleet Overview
  robots: RobotStatus[];
  machineStates: MachineStatus[];
  maintenanceQueue: MaintenanceTask[];
  globalMap: EnvironmentMap;
  
  // System Status
  systemUptime:
