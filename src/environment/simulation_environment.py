import numpy as np

class SimulationEnvironment:
    def __init__(self, route_data):
        self.route_data = route_data['route']
        self.points = {point['id']: point['coordinates'] for point in self.route_data['points']}
        self.connections = self.route_data['connections']
        self.start = self.route_data['points'][0]['id']
        self.goal = self.route_data['points'][-1]['id']
        self.current_state = None
        self.done = False
        self.sensors = [0] * 8  # 8 sensors around the vehicle
        self.vehicle_position = np.array([0.0, 0.0], dtype=float)  # Ensure float type
        self.vehicle_angle = 0.0  # Initial angle (in degrees)
        self.velocity = 0.0  # Initial velocity
        self.max_speed = 2.0  # Maximum speed
        self.acceleration = 0.1  # Acceleration rate
        self.braking_force = 0.2  # Deceleration rate during braking
        self.time_elapsed = 0  # Time elapsed since the last checkpoint
        self.time_limit = 15  # Allow more steps before timing out

    def reset(self):
        self.current_state = self.start
        self.done = False
        self.vehicle_position = np.array(
            [self.points[self.start]['latitude'], self.points[self.start]['longitude']], dtype=float
        )  # Ensure float type
        self.vehicle_angle = 0.0
        self.velocity = 0.0  # Reset velocity
        self.time_elapsed = 0  # Reset the timer
        self.update_sensors()
        return self.get_state_features()

    def step(self, action):
        # Actions: 0 = accelerate, 1 = brake, 2 = maintain speed, 3 = turn left, 4 = turn right
        if action == 0:  # Accelerate
            self.velocity = min(self.velocity + self.acceleration, self.max_speed)
        elif action == 1:  # Brake
            self.velocity = max(self.velocity - self.braking_force, 0)
        elif action == 2:  # Maintain speed
            pass  # No change in velocity
        elif action == 3:  # Turn left
            if self.velocity > 0:  # Turning is only possible if the vehicle is moving
                self.vehicle_angle -= 15 * (self.velocity / self.max_speed)  # Turn angle depends on speed
        elif action == 4:  # Turn right
            if self.velocity > 0:  # Turning is only possible if the vehicle is moving
                self.vehicle_angle += 15 * (self.velocity / self.max_speed)  # Turn angle depends on speed

        # Update position based on velocity and angle
        movement_vector = self.get_movement_vector(self.velocity)
        self.vehicle_position += movement_vector

        self.update_sensors()
        self.time_elapsed += 1  # Increment the timer (assuming 1 step = 1 second)

        # Display remaining steps
        remaining_steps = max(0, self.time_limit - self.time_elapsed)
        print(f"Remaining Steps: {remaining_steps}/{self.time_limit}")

        # Check if the time limit is exceeded
        if self.time_elapsed >= self.time_limit:
            self.done = True

        reward = self.calculate_reward()
        self.done = self.done or self.check_done()
        return self.get_state_features(), reward, self.done, {}

    def get_movement_vector(self, speed):
        # Calculate movement vector based on the current angle
        rad_angle = np.radians(self.vehicle_angle)
        movement_vector = np.array([np.cos(rad_angle), np.sin(rad_angle)]) * speed
        return movement_vector

    def update_sensors(self):
        # Simulate sensor readings (e.g., distances to walls or checkpoints)
        self.sensors = [np.random.uniform(0, 10) for _ in range(8)]  # Placeholder for actual sensor logic

    def calculate_reward(self):
        # Reward based on proximity to the goal or checkpoints
        if self.current_state == self.goal:
            return 100  # Large reward for reaching the goal

        # Penalize for each step
        reward = -1

        # Encourage moving closer to the goal
        goal_position = np.array([self.points[self.goal]['latitude'], self.points[self.goal]['longitude']])
        distance_to_goal = np.linalg.norm(self.vehicle_position - goal_position)
        reward += -0.1 * distance_to_goal  # Small penalty for being far from the goal

        # Encourage acceleration
        if self.velocity > 0:
            reward += 1  # Reward for moving

        # Penalize for staying still
        if self.velocity == 0:
            reward -= 2  # Additional penalty for not moving

        return reward

    def check_done(self):
        # Check if the vehicle has reached the goal or collided with a wall
        if self.current_state == self.goal:
            return True
        # Add collision detection logic here
        return False

    def get_state_features(self):
        goal_position = np.array([self.points[self.goal]['latitude'], self.points[self.goal]['longitude']])
        distance_to_goal = np.linalg.norm(self.vehicle_position - goal_position)
        state_vector = np.concatenate([
            self.vehicle_position,  # x, y position
            [self.vehicle_angle],  # angle
            [self.velocity],  # velocity
            [distance_to_goal],  # distance to goal
            self.sensors  # sensor readings
        ])
        return state_vector

    def render(self):
        # Visualize the environment (e.g., vehicle position, sensors, walls)
        print("=== Simulation Environment ===")
        print(f"Vehicle Position: {self.vehicle_position}, Angle: {self.vehicle_angle}°, Velocity: {self.velocity}")
        print(f"Sensor Readings: {self.sensors}")
        print(f"Current State: {self.current_state}, Goal: {self.goal}")
        print(f"Time Elapsed: {self.time_elapsed}s / {self.time_limit}s")
        print(f"Done: {self.done}")
        print("==============================")