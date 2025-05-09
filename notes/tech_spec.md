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


interface CoordinatorState {
  // Fleet Overview
  robots: RobotStatus[];
  machineStates: MachineStatus[];
  maintenanceQueue: MaintenanceTask[];
  globalMap: EnvironmentMap;
  
  // System Status
  systemUptime: number;
  currentPerformanceMetrics: PerformanceMetrics;
  alertsLog: SystemAlert[];
  
  // Scheduling and Planning
  maintenanceSchedule: ScheduledTask[];
  robotAssignments: {robotId: string, assignedMachines: string[]}[];
  taskPriorities: {taskId: string, priority: number}[];
  
  // Resource Management
  sparePartsInventory: SparePart[];
  chargingStations: {stationId: string, status: string, robotId?: string}[];
  toolInventory: {toolId: string, toolType: string, availability: boolean}[];
  
  // Learning and Optimization
  performanceHistory: HistoricalPerformance[];
  learningModels: ModelState[];
  optimizationParameters: OptimizationConfig;
}

interface RobotStatus {
  robotId: string;
  status: 'active' | 'charging' | 'maintenance' | 'offline';
  batteryLevel: number;
  location: {x: number, y: number, z: number};
  currentTask?: string;
  specializations: string[];
  healthStatus: {
    overallHealth: number;
    componentHealth: {component: string, health: number}[];
  };
}

interface MachineStatus {
  machineId: string;
  status: 'operational' | 'degraded' | 'maintenance' | 'failed';
  healthScore: number;
  faults: Fault[];
  lastMaintenance: number;
  criticality: number;
  productionImpact: number;
}
```

### 4.2 LangGraph Node Structure

```
┌────────────────────┐
│                    │
│  System Monitor    │───┐
│                    │   │
└────────────────────┘   │
                         │
┌────────────────────┐   │
│                    │   │
│  Status Aggregator │◄──┘
│                    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│                    │
│ Priority Optimizer │
│                    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│                    │
│  Task Allocator    │
│                    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│                    │
│ Resource Scheduler │
│                    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│                    │
│ Conflict Resolver  │
│                    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│                    │
│ Performance Analyzer│
│                    │
└────────────────────┘
```

### 4.3 Node Descriptions

#### 4.3.1 System Monitor Node
- **Input**: Raw status data from all robots, machines, and environment
- **Processing**:
  - Collection of real-time telemetry
  - Detection of system-wide anomalies
  - Monitoring of network health and communication quality
  - Tracking of resource availability
- **Output**: System health snapshot, anomaly alerts

```python
def system_monitor_node(state: CoordinatorState) -> Dict:
    """Monitor overall system health and collect status updates."""
    # Collect telemetry from all robots
    robot_telemetry = collect_robot_telemetry(state.robots)
    
    # Monitor machine status
    machine_status = monitor_machines(state.machineStates)
    
    # Check network health
    network_health = check_network_health(state.communication_logs)
    
    # Monitor resource availability
    resource_status = check_resources(
        state.sparePartsInventory,
        state.chargingStations,
        state.toolInventory
    )
    
    # Detect system-wide anomalies
    anomalies = detect_system_anomalies(
        robot_telemetry,
        machine_status,
        network_health,
        resource_status,
        state.performance_baselines
    )
    
    # Generate alerts for critical issues
    alerts = generate_alerts(anomalies, state.alert_thresholds)
    
    return {
        "robot_telemetry": robot_telemetry,
        "machine_status": machine_status,
        "network_health": network_health,
        "resource_status": resource_status,
        "detected_anomalies": anomalies,
        "alerts": alerts
    }
```

#### 4.3.2 Status Aggregator Node
- **Input**: System monitor output, historical data
- **Processing**:
  - Compilation of system-wide status
  - Trend analysis and pattern recognition
  - Machine learning model updates
  - Correlation of related issues
- **Output**: Comprehensive system status report, trend analysis

#### 4.3.3 Priority Optimizer Node
- **Input**: System status, machine criticality, fault information
- **Processing**:
  - Calculation of task priorities based on multiple factors
  - Evaluation of production impact
  - Consideration of maintenance windows
  - Assessment of fault progression risk
- **Output**: Optimized task priority queue

```python
def priority_optimizer_node(state: CoordinatorState) -> Dict:
    """Optimize maintenance task priorities based on multiple factors."""
    # Get all pending maintenance tasks
    pending_tasks = get_pending_tasks(
        state.maintenanceQueue,
        state.machineStates
    )
    
    # Calculate priority scores for each task
    priority_scores = []
    for task in pending_tasks:
        # Get associated machine
        machine = get_machine_by_id(state.machineStates, task.machineId)
        
        # Calculate base priority from multiple factors
        priority = calculate_base_priority(
            machine.criticality,
            task.estimatedRepairTime,
            machine.productionImpact,
            task.faultSeverity
        )
        
        # Adjust for fault progression risk
        progression_risk = estimate_fault_progression(
            task.faultType,
            task.faultSeverity,
            state.fault_progression_models
        )
        
        # Adjust for maintenance windows
        window_factor = evaluate_maintenance_window(
            machine.scheduledDowntime,
            state.current_time
        )
        
        # Adjust for resource availability
        resource_factor = check_resource_availability(
            task.requiredResources,
            state.available_resources
        )
        
        # Calculate final priority score
        final_priority = priority * progression_risk * window_factor * resource_factor
        
        priority_scores.append({
            "taskId": task.taskId,
            "priority": final_priority,
            "factors": {
                "basePriority": priority,
                "progressionRisk": progression_risk,
                "windowFactor": window_factor,
                "resourceFactor": resource_factor
            }
        })
    
    # Sort tasks by priority
    prioritized_tasks = sort_by_priority(priority_scores)
    
    return {
        "prioritized_tasks": prioritized_tasks,
        "priority_explanations": generate_priority_explanations(priority_scores)
    }
```

#### 4.3.4 Task Allocator Node
- **Input**: Prioritized task queue, robot status, specializations
- **Processing**:
  - Matching robots to tasks based on capability
  - Consideration of robot location and transit time
  - Evaluation of robot workload balance
  - Optimization for battery efficiency
- **Output**: Robot-task assignments with justification

#### 4.3.5 Resource Scheduler Node
- **Input**: Task assignments, resource inventory, time constraints
- **Processing**:
  - Scheduling of charging station access
  - Allocation of spare parts and tools
  - Planning for resource contention
  - Optimization of resource utilization
- **Output**: Resource allocation schedule, resource constraints

```python
def resource_scheduler_node(state: CoordinatorState) -> Dict:
    """Schedule and allocate resources for assigned maintenance tasks."""
    # Get all assigned tasks
    assigned_tasks = get_assigned_tasks(state.robotAssignments)
    
    # Identify required resources
    required_resources = identify_required_resources(
        assigned_tasks,
        state.task_resource_requirements
    )
    
    # Schedule charging station access
    charging_schedule = schedule_charging_stations(
        state.robots,
        state.chargingStations,
        assigned_tasks,
        state.battery_consumption_models
    )
    
    # Allocate spare parts
    parts_allocation = allocate_spare_parts(
        required_resources.parts,
        state.sparePartsInventory,
        assigned_tasks
    )
    
    # Allocate tools
    tools_allocation = allocate_tools(
        required_resources.tools,
        state.toolInventory,
        assigned_tasks
    )
    
    # Identify resource constraints
    resource_constraints = identify_constraints(
        parts_allocation.constraints,
        tools_allocation.constraints,
        charging_schedule.constraints
    )
    
    # Generate resource utilization timeline
    resource_timeline = generate_resource_timeline(
        parts_allocation.timeline,
        tools_allocation.timeline,
        charging_schedule.timeline
    )
    
    return {
        "charging_schedule": charging_schedule.schedule,
        "parts_allocation": parts_allocation.allocations,
        "tools_allocation": tools_allocation.allocations,
        "resource_constraints": resource_constraints,
        "resource_timeline": resource_timeline
    }
```

#### 4.3.6 Conflict Resolver Node
- **Input**: Task assignments, resource schedules, constraints
- **Processing**:
  - Identification of scheduling conflicts
  - Resolution of resource contention
  - Negotiation of task priorities
  - Coordination of multi-robot activities
- **Output**: Conflict-free schedule, coordination directives

#### 4.3.7 Performance Analyzer Node
- **Input**: Historical performance data, current metrics, goals
- **Processing**:
  - Analysis of system efficiency
  - Identification of performance bottlenecks
  - Comparison with performance targets
  - Recommendation of optimization opportunities
- **Output**: Performance analysis, optimization recommendations

### 4.4 Multi-Agent Coordination Mechanisms

#### 4.4.1 Task Delegation Protocol

```python
class TaskDelegationProtocol:
    def __init__(self, coordinator):
        self.coordinator = coordinator
    
    def delegate_task(self, task, target_robot):
        """Delegate a maintenance task to a specific robot."""
        # Check if robot is available
        if not self.is_robot_available(target_robot):
            return {
                "success": False,
                "reason": "Robot unavailable",
                "alternatives": self.find_alternative_robots(task)
            }
        
        # Check if robot has necessary capabilities
        if not self.has_required_capabilities(target_robot, task):
            return {
                "success": False,
                "reason": "Insufficient capabilities",
                "missing_capabilities": self.identify_missing_capabilities(target_robot, task)
            }
        
        # Create delegation message
        delegation = {
            "messageType": "TASK_DELEGATION",
            "taskId": task.id,
            "priority": task.priority,
            "machineId": task.machineId,
            "faultDetails": task.faultDetails,
            "requiredTools": task.requiredTools,
            "estimatedDuration": task.estimatedDuration,
            "delegatedBy": self.coordinator.id,
            "delegationTime": current_time()
        }
        
        # Send delegation to robot
        response = self.coordinator.communication.send_message(
            target_robot.id,
            "TASK_DELEGATION",
            delegation
        )
        
        # Update assignment records
        if response.status == "ACCEPTED":
            self.coordinator.update_task_assignment(task.id, target_robot.id)
            return {
                "success": True,
                "responseTime": response.responseTime,
                "estimatedStartTime": response.estimatedStartTime
            }
        else:
            return {
                "success": False,
                "reason": response.reason,
                "alternatives": self.find_alternative_robots(task)
            }
```

#### 4.4.2 Conflict Resolution Strategy

```python
class ConflictResolutionStrategy:
    def __init__(self, coordinator):
        self.coordinator = coordinator
    
    def resolve_resource_conflict(self, resource, competing_tasks):
        """Resolve conflicts between tasks competing for the same resource."""
        # Sort competing tasks by priority
        sorted_tasks = sorted(
            competing_tasks,
            key=lambda t: t.priority,
            reverse=True
        )
        
        # Allocate resource to highest priority task
        highest_priority = sorted_tasks[0]
        
        # For remaining tasks, find alternatives
        alternative_plans = []
        for task in sorted_tasks[1:]:
            alternatives = self.find_alternative_resources(resource, task)
            if alternatives:
                # Use alternative resource
                alternative_plans.append({
                    "taskId": task.id,
                    "originalResource": resource.id,
                    "alternativeResource": alternatives[0].id,
                    "impact": "minimal"
                })
            else:
                # Reschedule task
                new_time = self.find_next_availability(resource, highest_priority)
                alternative_plans.append({
                    "taskId": task.id,
                    "originalResource": resource.id,
                    "rescheduledTime": new_time,
                    "impact": "moderate"
                })
        
        return {
            "resourceId": resource.id,
            "allocatedTask": highest_priority.id,
            "alternativePlans": alternative_plans
        }
    
    def resolve_location_conflict(self, location, competing_robots):
        """Resolve conflicts between robots trying to access the same location."""
        # Determine which robot has highest priority task
        sorted_robots = sorted(
            competing_robots,
            key=lambda r: self.get_robot_task_priority(r),
            reverse=True
        )
        
        primary_robot = sorted_robots[0]
        
        # Create access schedule for location
        access_schedule = [{
            "robotId": primary_robot.id,
            "accessTime": current_time(),
            "estimatedDuration": primary_robot.current_task.estimatedDuration
        }]
        
        current_end_time = access_schedule[0]["accessTime"] + access_schedule[0]["estimatedDuration"]
        
        # Schedule other robots sequentially
        for robot in sorted_robots[1:]:
            access_schedule.append({
                "robotId": robot.id,
                "accessTime": current_end_time,
                "estimatedDuration": robot.current_task.estimatedDuration
            })
            current_end_time += robot.current_task.estimatedDuration
        
        return {
            "locationId": location.id,
            "accessSchedule": access_schedule,
            "conflictResolutionMethod": "sequential_access"
        }
```

## 5. Reward Function Implementation

The MDP reward function is implemented to guide agent decision-making:

```python
class RewardSystem:
    def __init__(self, config):
        self.config = config
        self.reward_history = []
    
    def calculate_repair_reward(self, repair_action, outcome):
        """Calculate reward for repair actions."""
        base_reward = 0
        
        # Determine base reward by repair type
        if repair_action.type == "repair_motor" and outcome.success:
            base_reward = 120
        elif repair_action.type == "repair_sensor" and outcome.success:
            base_reward = 100
        elif repair_action.type == "repair_power" and outcome.success:
            base_reward = 110
        elif repair_action.type == "repair_comm" and outcome.success:
            base_reward = 90
        elif repair_action.type == "replace_part" and outcome.success:
            base_reward = 150
        
        # Failed repair penalty
        if not outcome.success:
            base_reward = -20
        
        # Adjust reward based on machine criticality
        criticality_factor = self.get_machine_criticality(repair_action.machineId)
        adjusted_reward = base_reward * (1 + 0.5 * criticality_factor)
        
        return adjusted_reward
    
    def calculate_diagnosis_reward(self, diagnosis_action, outcome):
        """Calculate reward for diagnostic actions."""
        reward = 0
        
        # Reward for detected faults
        if outcome.detected_faults:
            reward += 50 * len(outcome.detected_faults)
        
        # Reward for inspecting machines needing maintenance
        if diagnosis_action.type == "inspect":
            machine_status = self.get_machine_status(diagnosis_action.machineId)
            if machine_status == "maintenance_needed":
                reward += 30
            
            # Reward for early detection
            early_detections = len([f for f in outcome.detected_faults if f.severity < 0.3])
            reward += 10 * early_detections
        
        return reward
    
    def calculate_movement_reward(self, movement_action):
        """Calculate reward for movement actions."""
        # Calculate distance
        distance = self.calculate_distance(
            movement_action.start_position,
            movement_action.end_position
        )
        
        # Base movement penalty (-1 to -5)
        movement_penalty = -min(5, max(1, distance))
        
        return movement_penalty
    
    def calculate_system_penalties(self, state):
        """Calculate system-wide penalties."""
        penalties = 0
        
        # Machine breakdown penalty (shared)
        broken_machines = [m for m in state.machines if m.status == 2]
        penalties -= 100 * len(broken_machines)
        
        # Fault escalation penalty
        escalated_faults = [f for f in state.new_fault_states 
                          if f.current_severity > 0.6 and f.previous_severity <= 0.3]
        penalties -= 40 * len(escalated_faults)
        
        return penalties
    
    def calculate_battery_penalties(self, robot_state):
        """Calculate battery-related penalties."""
        penalties = 0
        
        # Low battery penalty
        if robot_state.battery_level < 10:
            penalties -= 10
        
        # Battery depletion penalty
        if robot_state.battery_level <= 0:
            penalties -= 50
        
        return penalties
    
    def calculate_communication_penalties(self, state):
        """Calculate communication-related penalties."""
        penalties = 0
        
        # Packet loss penalty
        packet_losses = state.communication_stats.packet_losses
        penalties -= 15 * packet_losses
        
        # Failed delegation penalty
        failed_delegations = len(state.failed_delegations)
        penalties -= 5 * failed_delegations
        
        # Successful delegation reward
        successful_delegations = len(state.successful_delegations)
        penalties += 10 * successful_delegations
        
        return penalties
    
    def calculate_total_reward(self, action, outcome, state):
        """Calculate the total reward for an action."""
        reward = 0
        
        # Action-specific rewards
        if action.category == "repair":
            reward += self.calculate_repair_reward(action, outcome)
        elif action.category == "diagnose":
            reward += self.calculate_diagnosis_reward(action, outcome)
        elif action.category == "move":
            reward += self.calculate_movement_reward(action)
        
        # System penalties
        reward += self.calculate_system_penalties(state)
        
        # Battery penalties
        reward += self.calculate_battery_penalties(state.robot)
        
        # Communication penalties
        reward += self.calculate_communication_penalties(state)
        
        # Record reward
        self.reward_history.append({
            "action": action,
            "reward": reward,
            "timestamp": current_time()
        })
        
        return reward
```

## 6. Machine Learning Components

### 6.1 Fault Classification Model

```python
class FaultClassificationModel:
    def __init__(self, model_config):
        self.model_path = model_config.model_path
        self.feature_columns = model_config.feature_columns
        self.label_column = model_config.label_column
        self.model = self.load_model()
        self.confidence_threshold = model_config.confidence_threshold
    
    def load_model(self):
        """Load the trained classification model."""
        try:
            # Load model from file
            if self.model_path.endswith('.pkl'):
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = tf.keras.models.load_model(self.model_path)
            
            return model
        except Exception as e:
            # Fallback to default model if loading fails
            logger.error(f"Failed to load model: {e}")
            return self.create_default_model()
    
    def create_default_model(self):
        """Create a default model if loading fails."""
        if hasattr(self, 'model_type') and self.model_type == 'neural_network':
            # Create a simple neural network
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(len(self.feature_columns),)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(len(self.fault_classes), activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            # Create a random forest classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        return model
    
    def preprocess_features(self, sensor_data):
        """Preprocess sensor data into model features."""
        features = []
        
        for column in self.feature_columns:
            if column in sensor_data:
                features.append(sensor_data[column])
            else:
                # Handle missing features
                features.append(0.0)
        
        # Apply any scaling or normalization required by the model
        features = self.apply_feature_scaling(features)
        
        return np.array(features).reshape(1, -1)
    
    def predict_fault(self, sensor_data):
        """Predict fault type and confidence from sensor data."""
        # Preprocess features
        features = self.preprocess_features(sensor_data)
        
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
        else:
            # For neural networks
            probabilities = self.model.predict(features)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class_idx])
        
        # Get predicted fault type
        fault_type = self.fault_classes[predicted_class_idx]
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return {
                "fault_type": "unknown",
                "confidence": confidence,
                "needs_further_analysis": True,
                "possible_types": self.get_possible_types(probabilities)
            }
        
        return {
            "fault_type": fault_type,
            "confidence": confidence,
            "needs_further_analysis": False,
            "probabilities": {self.fault_classes[i]: float(p) for i, p in enumerate(probabilities)}
        }
```

### 6.2 Repair Success Prediction Model

```python
class RepairSuccessPredictionModel:
    def __init__(self, model_config):
        self.model_path = model_config.model_path
        self.model = self.load_model()
        self.feature_extractors = self.initialize_feature_extractors()
    
    def load_model(self):
        """Load the trained repair success prediction model."""
        # Similar model loading logic as above
        pass
    
    def extract_features(self, repair_context):
        """Extract features from repair context for prediction."""
        features = {}
        
        # Extract robot-related features
        features.update(self.feature_extractors.robot_features(repair_context.robot))
        
        # Extract fault-related features
        features.update(self.feature_extractors.fault_features(repair_context.fault))
        
        # Extract machine-related features
        features.update(self.feature_extractors.machine_features(repair_context.machine))
        
        # Extract repair history features
        features.update(self.feature_extractors.history_features(
            repair_context.machine.maintenance_history,
            repair_context.fault.type
        ))
        
        # Extract environment features
        features.update(self.feature_extractors.environment_features(repair_context.environment))
        
        return np.array(list(features.values())).reshape(1, -1)
    
    def predict_success_probability(self, repair_context):
        """Predict the probability of repair success."""
        # Extract features
        features = self.extract_features(repair_context)
        
        # Make prediction
        success_probability = self.model.predict_proba(features)[0][1]  # Probability of success class
        
        # Generate explanation for prediction
        explanation = self.generate_explanation(success_probability, repair_context)
        
        return {
            "success_probability": success_probability,
            "explanation": explanation,
            "influencing_factors": self.identify_key_factors(features, repair_context)
        }
    
    def generate_explanation(self, probability, context):
        """Generate human-readable explanation for the prediction."""
        # Base explanation
        if probability > 0.8:
            base = "High chance of successful repair"
        elif probability > 0.5:
            base = "Moderate chance of successful repair"
        else:
            base = "Low chance of successful repair"
        
        # Add factor-specific explanations
        factors = []
        
        if context.robot.battery_level < 30:
            factors.append("low battery level may affect precision")
        
        if context.fault.severity > 0.7:
            factors.append("high fault severity increases complexity")
        
        if len(context.robot.successful_repairs) > 10:
            factors.append("robot has experience with similar repairs")
        
        # Combine explanations
        if factors:
            explanation = f"{base} because {', '.join(factors)}"
        else:
            explanation = base
        
        return explanation
```

## 7. Integration with ROS Environment

### 7.1 ROS Node Structure

```
/industrial_mind
  /agents
    /robot_{id}
      /perception
      /decision_making
      /action_execution
      /communication
    /central_coordinator
      /system_monitoring
      /task_allocation
      /resource_management
  /environment
    /machine_simulation
    /robot_simulation
    /physics
  /fault_simulation
    /motor_faults
    /sensor_faults
    /power_faults
    /comm_faults
  /visualization
    /dashboard
    /robot_status
    /machine_status
```

### 7.2 Topic Structure

```
# Robot-related topics
/industrial_mind/robots/{robot_id}/status
/industrial_mind/robots/{robot_id}/command
/industrial_mind/robots/{robot_id}/sensor_data
/industrial_mind/robots/{robot_id}/battery
/industrial_mind/robots/{robot_id}/position

# Machine-related topics
/industrial_mind/machines/{machine_id}/status
/industrial_mind/machines/{machine_id}/telemetry
/industrial_mind/machines/{machine_id}/faults

# Task-related topics
/industrial_mind/tasks/queue
/industrial_mind/tasks/assignments
/industrial_mind/tasks/completed

# System-level topics
/industrial_mind/system/status
/industrial_mind/system/performance
/industrial_mind/system/alerts

# Fault simulation topics
/industrial_mind/fault_simulation/control
/industrial_mind/fault_simulation/status
```

## 8. Testing and Evaluation Framework

### 8.1 Evaluation Scenarios

The system will be tested with scenarios derived from our fault simulation framework:

```python
class IndustrialEvaluationScenario:
    def __init__(self, scenario_config):
        self.name = scenario_config.name
        self.description = scenario_config.description
        self.duration = scenario_config.duration
        self.fault_sequence = scenario_config.fault_sequence
        self.success_criteria = scenario_config.success_criteria
        self.environment_config = scenario_config.environment
    
    def setup(self, simulation_manager):
        """Set up the evaluation scenario."""
        # Configure environment
        simulation_manager.configure_environment(self.environment_config)
        
        # Set up initial machine states
        for machine_config in self.environment_config.machines:
            simulation_manager.configure_machine(machine_config)
        
        # Set up robot configurations
        for robot_config in self.environment_config.robots:
            simulation_manager.configure_robot(robot_config)
        
        # Schedule fault injections
        for fault in self.fault_sequence:
            simulation_manager.schedule_fault(
                fault.time,
                fault.target,
                fault.type,
                fault.parameters
            )
    
    def evaluate_results(self, results):
        """Evaluate scenario results against success criteria."""
        evaluation = {}
        
        # Evaluate each success criterion
        for criterion in self.success_criteria:
            criterion_result = self.evaluate_criterion(criterion, results)
            evaluation[criterion.name] = criterion_result
        
        # Calculate overall score
        total_score = sum(result.score * criterion.weight 
                         for criterion, result in zip(self.success_criteria, evaluation.values()))
        max_score = sum(criterion.weight for criterion in self.success_criteria)
        normalized_score = total_score / max_score if max_score > 0 else 0
        
        return {
            "scenario_name": self.name,
            "criterion_results": evaluation,
            "overall_score": normalized_score,
            "passed": normalized_score >= 0.7,  # 70% is passing threshold
            "insights": self.generate_insights(evaluation, results)
        }
```

### 8.2 Test Scenarios

We will implement the following test scenarios:

1. **Progressive Motor Degradation**
   - Multiple machines develop increasing motor wear
   - Robots must detect, diagnose and prioritize maintenance

2. **Sensor Failure Cascade**
   - Sensor faults lead to misleading machine data
   - Robots must identify unreliable sensors and use alternative diagnostics

3. **Power System Instability**
   - Power brownouts affect both machines and robots
   - System must adapt to unreliable power conditions

4. **Communication Disruption**
   - Network degradation affects robot coordination
   - System must maintain operations with limited communication

5. **Resource Constraint Challenge**
   - Limited spare parts and tools for multiple failures
   - System must optimize resource allocation and maintenance scheduling

## 9. Implementation Timeline

| Phase | Timeframe | Key Deliverables |
|-------|-----------|------------------|
| Infrastructure Setup | Weeks 1-2 | ROS/Gazebo environment, basic simulation |
| Fault System Adaptation | Weeks 3-4 | Industrial machine fault models |
| Robot Agent Core | Weeks 5-6 | Basic LangGraph agent implementation |# IndustrialMind: Technical Specification

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
