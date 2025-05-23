import numpy as np
import json
from copy import deepcopy
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, population_size, input_shape, output_shape, mutation_rate=0.1):
        self.population_size = population_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
        self.generation_stats = []  # Store fitness stats for each generation

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                "speed": np.random.uniform(3.0, 30.0),  # Maksymalna prędkość
                "acceleration": np.random.uniform(0.05, 0.35),  # Przyspieszenie
                "braking_force": np.random.uniform(0.1, 0.5),  # Siła hamowania
                "turn_angle": np.random.uniform(10, 45),  # Maksymalny kąt skrętu
                "model": self.create_random_model()  # Model decyzyjny
            }
            population.append(individual)
        return population
    
    def save_population(self, file_path="population.json"):
        # Serialize population to JSON
        serializable_population = []
        for individual in self.population:
            serializable_population.append({
                "speed": individual["speed"],
                "turn_angle": individual["turn_angle"],
                "model_weights": [w.tolist() for w in individual["model"].get_weights()]
            })
        with open(file_path, "w") as file:
            json.dump(serializable_population, file)
        print(f"Population saved to {file_path}")

    def load_population(self, file_path="population.json"):
        # Deserialize population from JSON
        with open(file_path, "r") as file:
            serialized_population = json.load(file)
        self.population = []
        for individual_data in serialized_population:
            model = self.create_random_model()
            model.set_weights([np.array(w) for w in individual_data["model_weights"]])
            self.population.append({
                "speed": individual_data.get("speed", np.random.uniform(3.0, 30.0)),
                "turn_angle": individual_data.get("turn_angle", np.random.uniform(10, 45)),
                "acceleration": individual_data.get("acceleration", np.random.uniform(0.05, 0.35)),
                "braking_force": individual_data.get("braking_force", np.random.uniform(0.1, 0.5)),
                "model": model
            })
        print(f"Population loaded from {file_path}")

    def create_random_model(self):
        from tensorflow.keras.models import Sequential # type: ignore
        from tensorflow.keras.layers import Dense, Input # type: ignore

        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.output_shape, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def evaluate_population(self, environment, episodes=10):
        fitness_scores = []
        for i, individual in enumerate(self.population):
            total_reward = 0
            for _ in range(episodes):
                environment.reset()
                environment.vehicle_speed = individual["speed"]
                environment.vehicle_turn_angle = individual["turn_angle"]
                state = environment.get_state_features()
                done = False
                while not done:
                    action = np.argmax(individual["model"].predict(np.expand_dims(state, axis=0)))
                    state, reward, done, _ = environment.step(action)
                    total_reward += reward
            average_reward = total_reward / episodes
            fitness_scores.append(average_reward)
            print(f"Individual {i}: Average Reward: {average_reward}")
        return fitness_scores

    def select_parents(self, fitness_scores):
        # Normalize fitness scores to probabilities
        fitness_sum = sum(fitness_scores)
        probabilities = [score / fitness_sum for score in fitness_scores]
        
        # Select two parents based on probabilities
        parents_indices = np.random.choice(len(self.population), size=2, p=probabilities, replace=False)
        return [self.population[parents_indices[0]], self.population[parents_indices[1]]]

    def crossover(self, parent1, parent2):
        child = {
            "speed": np.mean([parent1["speed"], parent2["speed"]]),
            "acceleration": np.mean([parent1["acceleration"], parent2["acceleration"]]),
            "braking_force": np.mean([parent1["braking_force"], parent2["braking_force"]]),
            "turn_angle": np.mean([parent1["turn_angle"], parent2["turn_angle"]]),
            "model": self.crossover_models(parent1["model"], parent2["model"])
        }
        return child

    def crossover_models(self, model1, model2):
        # Mix weights of two models
        child_model = self.create_random_model()
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        new_weights = [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]
        child_model.set_weights(new_weights)
        return child_model

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            individual["speed"] += np.random.uniform(-1, 1)
            individual["speed"] = np.clip(individual["speed"], 3.0, 30.0)
        if np.random.rand() < self.mutation_rate:
            individual["acceleration"] += np.random.uniform(-0.02, 0.02)
            individual["acceleration"] = np.clip(individual["acceleration"], 0.05, 0.35)
        if np.random.rand() < self.mutation_rate:
            individual["braking_force"] += np.random.uniform(-0.03, 0.03)
            individual["braking_force"] = np.clip(individual["braking_force"], 0.1, 0.5)
        if np.random.rand() < self.mutation_rate:
            individual["turn_angle"] += np.random.uniform(-2, 2)
            individual["turn_angle"] = np.clip(individual["turn_angle"], 10, 45)

    def evolve(self, fitness_scores):
        # Sortuj populację według wyników fitness (malejąco)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        best_individual = self.population[sorted_indices[0]]  # Najlepszy osobnik

        # Zachowaj statystyki generacji
        best_fitness = fitness_scores[sorted_indices[0]]
        worst_fitness = fitness_scores[sorted_indices[-1]]
        avg_fitness = np.mean(fitness_scores)
        self.generation_stats.append({
            "best": best_fitness,
            "worst": worst_fitness,
            "average": avg_fitness
        })

        # Tworzenie nowej populacji
        new_population = [best_individual]  # Dodaj najlepszego osobnika do nowej populacji (elitism)
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(fitness_scores)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)

        self.population = new_population

    def plot_generation_stats(self, output_file="generation_stats.png"):
        generations = range(1, len(self.generation_stats) + 1)
        best_fitness = [stat["best"] for stat in self.generation_stats]
        worst_fitness = [stat["worst"] for stat in self.generation_stats]
        avg_fitness = [stat["average"] for stat in self.generation_stats]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, label="Best Fitness", color="green")
        plt.plot(generations, worst_fitness, label="Worst Fitness", color="red")
        plt.plot(generations, avg_fitness, label="Average Fitness", color="blue")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Progress Across Generations")
        plt.legend()
        plt.grid()
        plt.savefig(output_file)
        plt.close()