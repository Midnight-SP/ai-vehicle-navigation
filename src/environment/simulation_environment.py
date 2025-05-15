import numpy as np

class SimulationEnvironment:
    def __init__(self, route_data, genes):
        self.route_data = route_data['route']
        self.checkpoints = self.load_checkpoints()
        self.walls = self.load_walls()
        self.start_position = np.array([
            self.route_data['start_position']['latitude'],
            self.route_data['start_position']['longitude']
        ], dtype=float)
        self.start_angle = self.route_data['start_angle']
        self.current_checkpoint = 0
        self.done = False
        self.sensors = [0] * 8
        self.vehicle_position = self.start_position.copy()
        self.vehicle_angle = self.start_angle
        self.velocity = 0.0
        self.path = []  # Śledzenie ścieżki pojazdu

        # Parametry genetyczne
        self.max_speed = genes["speed"]
        self.acceleration = genes["acceleration"]
        self.braking_force = genes["braking_force"]
        self.turn_angle = genes["turn_angle"]

        self.time_elapsed = 0
        self.time_limit = 50

    def load_checkpoints(self):
        checkpoints = []
        for checkpoint in self.route_data['checkpoints']:
            start = checkpoint['start']
            end = checkpoint['end']
            checkpoints.append(((start['latitude'], start['longitude']), (end['latitude'], end['longitude'])))
        return checkpoints

    def load_walls(self):
        walls = []
        for wall in self.route_data['walls']:
            try:
                start = wall['start']
                end = wall['end']
                walls.append(((start['latitude'], start['longitude']), (end['latitude'], end['longitude'])))
            except KeyError as e:
                print(f"Error loading wall: {e}")
        return walls

    def reset(self):
        self.current_checkpoint = 0
        self.done = False
        self.vehicle_position = self.start_position.copy()
        self.vehicle_angle = self.start_angle
        self.velocity = 0.0
        self.time_elapsed = 0
        self.path = [self.vehicle_position.copy()]  # Resetuj ścieżkę
        self.update_sensors()
        return self.get_state_features()

    def step(self, action):
        if action == 0:  # Accelerate
            self.velocity = min(self.velocity + self.acceleration, self.max_speed)
        elif action == 1:  # Brake
            self.velocity = max(self.velocity - self.braking_force, 0)
        elif action == 2:  # Maintain speed
            pass
        elif action == 3:  # Turn left
            if self.velocity > 0:
                self.vehicle_angle -= self.turn_angle * (self.velocity / self.max_speed)
        elif action == 4:  # Turn right
            if self.velocity > 0:
                self.vehicle_angle += self.turn_angle * (self.velocity / self.max_speed)

        # Aktualizuj pozycję pojazdu
        movement_vector = self.get_movement_vector(self.velocity)
        self.vehicle_position += movement_vector
        self.path.append(self.vehicle_position.copy())  # Dodaj pozycję do ścieżki
        print(f"Updated vehicle position: {self.vehicle_position}")  # Log pozycji pojazdu

        # Aktualizuj sensory
        self.update_sensors()

        # Sprawdź, czy pojazd przekroczył punkt kontrolny
        checkpoint_crossed = self.check_checkpoint_crossed()

        # Zwiększ czas symulacji
        self.time_elapsed += 1
        if self.time_elapsed >= self.time_limit:
            self.done = True

        # Oblicz nagrodę
        reward = self.calculate_reward(checkpoint_crossed)
        return self.get_state_features(), reward, self.done, {}

    def get_movement_vector(self, speed):
        # Calculate movement vector based on the current angle
        rad_angle = np.radians(self.vehicle_angle)
        movement_vector = np.array([np.cos(rad_angle), np.sin(rad_angle)]) * speed
        return movement_vector
    
    def ray_intersects_wall(self, ray_origin, ray_direction, wall_start, wall_end):
        # Parametryczne równanie linii
        x1, y1 = wall_start
        x2, y2 = wall_end
        x3, y3 = ray_origin
        x4, y4 = ray_origin + ray_direction

        # Obliczanie determinanty
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Linie są równoległe

        # Obliczanie punktu przecięcia
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 0 <= t <= 1 and u >= 0:
            intersection = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
            return intersection
        return None

    def update_sensors(self):
        # Sensor kierunki w stopniach (0°, 45°, ..., 315°)
        sensor_angles = np.linspace(0, 360, len(self.sensors), endpoint=False)
        sensor_distances = []

        for angle in sensor_angles:
            # Przekształć kąt na radiany i oblicz kierunek promienia
            rad_angle = np.radians(self.vehicle_angle + angle)
            ray_direction = np.array([np.cos(rad_angle), np.sin(rad_angle)])

            # Znajdź najbliższy punkt przecięcia z dowolną ścianą
            min_distance = float('inf')
            for wall in self.walls:
                if len(wall) != 2 or len(wall[0]) != 2 or len(wall[1]) != 2:
                    print(f"Invalid wall format: {wall}")
                    continue
                intersection = self.ray_intersects_wall(self.vehicle_position, ray_direction, np.array(wall[0]), np.array(wall[1]))
                if intersection is not None:
                    distance = np.linalg.norm(intersection - self.vehicle_position)
                    min_distance = min(min_distance, distance)

            # Jeśli brak przecięcia, ustaw maksymalny zasięg (np. 10 jednostek)
            sensor_distances.append(min_distance if min_distance != float('inf') else 10.0)

        self.sensors = sensor_distances

    def calculate_reward(self, checkpoint_crossed):
        reward = -1
        if checkpoint_crossed:
            reward += 50
            print("Checkpoint crossed! Reward: 50")
        else:
            print("No checkpoint crossed. Penalty: -1")

        # Encourage acceleration
        if self.velocity > 0:
            reward += 1  # Reward for moving

        # Penalize for staying still
        if self.velocity == 0:
            reward -= 2  # Additional penalty for not moving

        # Penalize for time elapsed
        if self.time_elapsed >= self.time_limit:
            reward -= 10
        
        # Penalize for hitting walls
        for wall in self.walls:
            if self.ray_intersects_wall(self.vehicle_position, np.array([0, 0]), np.array(wall[0]), np.array(wall[1])) is not None:
                reward -= 20
                break  # Stop checking after the first hit

        return reward
        
    def check_checkpoint_crossed(self):
        if self.current_checkpoint >= len(self.checkpoints):
            print("No more checkpoints to cross.")
            return False

        checkpoint_start, checkpoint_end = self.checkpoints[self.current_checkpoint]
        vehicle_pos = self.vehicle_position

        checkpoint_vector = np.array(checkpoint_end) - np.array(checkpoint_start)
        vehicle_vector = vehicle_pos - np.array(checkpoint_start)
        cross_product = np.cross(checkpoint_vector, vehicle_vector)

        print(f"Vehicle position: {vehicle_pos}, Checkpoint: {checkpoint_start} -> {checkpoint_end}, Cross product: {cross_product}")

        if cross_product > 0:
            print(f"Checkpoint {self.current_checkpoint} crossed.")
            self.current_checkpoint += 1
            self.time_elapsed = 0
            if self.current_checkpoint == len(self.checkpoints):
                self.done = True
            return True
        return False

    def check_done(self):
        # Check if the vehicle has crossed all checkpoints
        if self.current_checkpoint >= len(self.checkpoints):
            return True  # Done when all checkpoints are crossed
        return False

    def get_state_features(self):
        # Calculate the distance to the next checkpoint
        if self.current_checkpoint < len(self.checkpoints):
            checkpoint_position = np.array(self.checkpoints[self.current_checkpoint][0])
            distance_to_checkpoint = np.linalg.norm(self.vehicle_position - checkpoint_position)
        else:
            distance_to_checkpoint = 0.0  # No more checkpoints

        # Construct the state vector
        state_vector = np.concatenate([
            self.vehicle_position,  # x, y position
            [self.vehicle_angle],  # angle
            [self.velocity],  # velocity
            [distance_to_checkpoint],  # distance to next checkpoint
            self.sensors  # sensor readings
        ])
        return state_vector