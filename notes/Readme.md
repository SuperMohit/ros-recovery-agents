# Robotic Agents - Fault Scenarios Documentation

This document provides detailed information about the fault scenarios implemented to test the capabilities of our drone surveillance AI agent. These scenarios are designed to evaluate the agent's ability to detect, diagnose, and respond to various types of failures that could occur during real-world operation.

## Overview

The fault simulation system allows us to inject various types of faults into the drone simulation environment, creating realistic failure conditions that the AI agent must handle. These simulated faults progressively degrade the drone's capabilities, creating increasingly challenging situations that test the agent's decision-making abilities.

## Available Fault Types

### 1. Motor/Actuator Faults

| Fault Type | Description | Parameters | Observable Effects |
|------------|-------------|------------|-------------------|
| **Motor Wear** | Simulates gradual degradation of motor efficiency | - `efficiency`: Starting efficiency (0.0-1.0)<br>- `degradation_rate`: Efficiency loss per minute | - Reduced thrust<br>- Increased current draw<br>- Poor response to movement commands |
| **Control Instability** | Introduces oscillations in control signals | - `magnitude`: Oscillation size<br>- `frequency`: Oscillation frequency (Hz)<br>- `progression_rate`: Rate of increase | - Jittery movement<br>- Position overshoots<br>- Difficulty maintaining position |
| **Joint Backlash** | Adds deadband to position controller | - `deadband`: Size of deadband in radians | - Delayed response to direction changes<br>- Poor position repeatability |

### 2. Sensor Faults

| Fault Type | Description | Parameters | Observable Effects |
|------------|-------------|------------|-------------------|
| **IMU Drift** | Adds slowly increasing bias to IMU readings | - `magnitude`: Base drift magnitude<br>- `drift_x/y/z`: Relative drift in each axis<br>- `progression_rate`: Rate of increase | - Gradual deviation from actual orientation<br>- Increasing navigation errors |
| **Sensor Noise** | Adds variable Gaussian noise to sensor readings | - `magnitude`: Standard deviation of noise<br>- `progression_rate`: Rate of increase | - Jittery position estimates<br>- Erratic movement<br>- Decreased stability |
| **Position Jump** | Causes sudden jumps in position estimation | - `magnitude`: Jump size in meters<br>- `probability`: Chance of jump per update<br>- `progression_rate`: Rate of increase | - Sudden perceived position changes<br>- Drone overcorrection<br>- Navigation errors |

### 3. Battery/Power Faults

| Fault Type | Description | Parameters | Observable Effects |
|------------|-------------|------------|-------------------|
| **Battery Degradation** | Simulates accelerated battery depletion | - `initial_capacity`: Starting capacity (%)<br>- `degradation_rate`: Capacity loss per minute | - Faster than expected battery drain<br>- Reduced flight time<br>- Potential emergency situations |

### 4. Communication Faults

| Fault Type | Description | Parameters | Observable Effects |
|------------|-------------|------------|-------------------|
| **Communication Delay** | Introduces latency in command execution | - `delay`: Delay in seconds<br>- `progression_rate`: Rate of increase | - Delayed response to commands<br>- Control difficulties<br>- Poor synchronization |
| **Packet Loss** | Randomly drops command messages | - `loss_rate`: Probability of message loss<br>- `progression_rate`: Rate of increase | - Missing commands<br>- Incomplete execution sequences<br>- Unpredictable behavior |

## Predefined Scenarios

The system includes several predefined scenarios that combine multiple faults to create complex test situations:

### 1. Motor Degradation Scenario

Simulates the gradual wearing out of drone motors and control systems.

**Progression:**
1. Initial motor efficiency degradation (mild)
2. Addition of control instability after 30 seconds (moderate)
3. Both faults worsen progressively over time

**Expected Agent Response:**
- Detection of motor efficiency loss
- Recognition of unstable flight patterns
- Decision to reduce flight speed/aggressiveness
- Potential return to base if severe enough

### 2. Sensor Failure Scenario

Simulates the degradation and eventual failure of the drone's sensors.

**Progression:**
1. Initial IMU drift (subtle)
2. Addition of sensor noise after 45 seconds
3. Position jump glitches after 90 seconds
4. All faults worsen progressively

**Expected Agent Response:**
- Detection of navigation inconsistencies
- Adjustment of reliance on affected sensors
- Position re-calibration attempts
- Potential switch to alternate navigation modes

### 3. Communication Problem Scenario

Simulates degrading communications between the drone and control systems.

**Progression:**
1. Initial command delay (mild)
2. Addition of packet loss after 60 seconds
3. Both issues worsen progressively

**Expected Agent Response:**
- Detection of communication issues
- Reduced dependence on frequent commands
- More autonomous decision-making
- Potential return to communication range

### 4. Power System Failure Scenario

Simulates a failing battery and power distribution system.

**Progression:**
1. Initial battery degradation (moderate rate)
2. Addition of motor efficiency loss after 30 seconds
3. Both issues worsen progressively

**Expected Agent Response:**
- Detection of abnormal power consumption
- Battery life recalculation
- Prioritization of critical functions
- Emergency landing or return to charger

### 5. Catastrophic Failure Scenario

Combines multiple serious faults to test the agent's emergency management capabilities.

**Progression:**
1. Initial motor wear (moderate)
2. Addition of sensor drift after 20 seconds
3. Battery degradation after 40 seconds
4. Communication delay after 50 seconds
5. Control instability after 60 seconds
6. All faults worsen rapidly

**Expected Agent Response:**
- Correct prioritization of multiple issues
- Recognition of cascading failure pattern
- Emergency procedures initiation
- Safe landing or return decision making

## Multiple Faults Test Scenario

The `MultipleFaultsScenario` class implements a comprehensive test that progresses through several phases to evaluate the agent's full range of capabilities.

### Phase Structure

1. **Phase 0: Normal Operation (60s)**
   - Establishes baseline behavior
   - No active faults

2. **Phase 1: Sensor Drift (120s)**
   - Tests navigation challenge
   - Introduces progressive IMU drift

3. **Phase 2: Motor Wear + Control Instability (120s)**
   - Tests flight control challenge
   - Introduces motor wear
   - Adds control instability after 30 seconds

4. **Phase 3: Communication Problems (90s)**
   - Tests command & control challenge
   - Introduces communication delays
   - Adds packet loss after 30 seconds

5. **Phase 4: Battery Degradation (120s)**
   - Tests emergency management
   - Introduces rapid battery depletion

### Evaluation Metrics

The test collects and analyzes multiple metrics to evaluate agent performance:

1. **Issue Detection**
   - Time to detect each issue type
   - Percentage of injected faults detected
   - Accuracy of issue classification

2. **Response Time**
   - Time between issue detection and first action
   - Variation in response times across issue types
   - Prioritization of critical issues

3. **Action Appropriateness**
   - Correctness of actions taken for each issue
   - Effectiveness of actions in resolving issues
   - Adaptability when actions don't resolve issues

4. **Overall Mission Management**
   - Ability to balance safety with mission objectives
   - Management of multiple simultaneous issues
   - Decision quality under degraded conditions

### Test Output

The test generates comprehensive output for analysis:

1. **Detailed Log File**
   - Timestamped events
   - Issue detections
   - Agent actions
   - System state changes

2. **Summary Report**
   - Overall performance scores
   - Issue detection statistics
   - Response time analysis
   - Test verdict with recommendations

3. **Visualizations**
   - Battery level over time
   - Issue count timeline
   - Agent action frequency
   - Response time analysis
   - Event timeline
   - Performance by fault type

## Using the Fault Scenario System

### Basic Usage

The fault scenario system can be used in two ways:

1. **Individual Fault Injection**

```python
# Create the fault generator
fault_generator = FaultScenarioGenerator()

# Activate specific faults
fault_generator.activate_battery_degradation()
fault_generator.activate_motor_wear()
fault_generator.activate_sensor_drift("imu")

# Later, deactivate faults
fault_generator.deactivate_fault("battery_degradation")
fault_generator.deactivate_all_faults()
```

2. **Predefined Scenario Execution**

```python
# Create the fault generator
fault_generator = FaultScenarioGenerator()

# Run a predefined scenario
fault_generator.run_combined_scenario("motor_degradation")
fault_generator.run_combined_scenario("catastrophic_failure")
```

### Running the Comprehensive Test

To run the full multiple faults test scenario:

```bash
roslaunch drone_ai_agent test_multiple_faults.launch
```

This will:
1. Launch the drone simulation environment
2. Start the AI agent
3. Run the multiple faults test scenario
4. Generate detailed reports and visualizations

### Customizing Fault Parameters

Fault parameters can be customized when activating individual faults:

```python
fault_generator.activate_battery_degradation({
    "initial_capacity": 80.0,    # Start at 80% capacity
    "degradation_rate": 7.5      # Lose 7.5% per minute
})

fault_generator.activate_sensor_drift("imu", {
    "magnitude": 0.02,           # Stronger initial drift
    "drift_x": 0.3,              # More drift in x axis
    "drift_y": 0.1,              # Less drift in y axis
    "drift_z": 0.2,              # Moderate drift in z axis
    "progression_rate": 0.4      # Faster progression
})
```

## Creating Custom Test Scenarios

Custom test scenarios can be created by extending the `MultipleFaultsScenario` class or by writing custom scripts that use the `FaultScenarioGenerator`.

### Example: Custom Scenario

```python
class WeatherChallengeScenario:
    def __init__(self):
        rospy.init_node('weather_challenge_scenario')
        self.fault_generator = FaultScenarioGenerator()
        
    def run_test(self):
        # Start with wind gusts
        self.fault_generator.activate_control_instability({
            "magnitude": 0.15,
            "frequency": 1.5,    # Slower oscillation like wind gusts
            "progression_rate": 0.1
        })
        
        # After 2 minutes, add sensor issues (like rain on sensors)
        rospy.Timer(rospy.Duration(120), 
            lambda event: self.fault_generator.activate_sensor_noise("imu"), 
            oneshot=True
        )
        
        # After 4 minutes, add battery issues (like cold weather effects)
        rospy.Timer(rospy.Duration(240), 
            lambda event: self.fault_generator.activate_battery_degradation(), 
            oneshot=True
        )
        
        # Run for 10 minutes total
        rospy.sleep(rospy.Duration(600))
        self.fault_generator.deactivate_all_faults()
```

## Integration with the AI Agent

The fault simulation system integrates with the AI agent through standard ROS topics, requiring no special modifications to the agent itself. This black-box testing approach ensures realistic evaluation of the agent's capabilities.

Key integration points:

1. **Sensor Data Modification**: The fault system intercepts and modifies sensor readings
2. **Command Interception**: Control commands can be delayed, dropped, or modified
3. **System State Publication**: System states (battery, mode) are published to standard topics

## Extending the System

The fault simulation system can be extended in several ways:

1. **Adding New Fault Types**
   - Create new fault implementation methods in FaultScenarioGenerator
   - Follow the pattern of existing faults
   - Register fault parameters and progression

2. **Creating New Combined Scenarios**
   - Define new scenarios in run_combined_scenario()
   - Sequence multiple faults with appropriate timing
   - Consider interactions between different fault types

3. **Adding Environmental Challenges**
   - Implement weather effects (wind, rain, fog)
   - Create obstacle scenarios
   - Simulate RF interference or GPS denial

## Conclusion

The fault scenario system provides a comprehensive framework for testing the AI agent's ability to handle a wide range of failure conditions. By systematically injecting and progressing faults, we can evaluate the agent's detection, diagnosis, and response capabilities in a controlled environment before deployment to real-world systems.
