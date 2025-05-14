# ai-vehicle-navigation/src/main.py

import os
import json
from agents.vehicle_agent import VehicleAgent
from environment.simulation_environment import SimulationEnvironment
from models.reinforcement_learning_model import ReinforcementLearningModel
from utils.genetic_algorithm import GeneticAlgorithm

def load_route(route_file):
    with open(route_file, 'r') as file:
        return json.load(file)

def main():
    route_data = load_route('data/routes/sample_route.json')
    environment = SimulationEnvironment(route_data)

    # Genetic Algorithm parameters
    population_size = 10
    input_shape = (13,)
    output_shape = 5
    mutation_rate = 0.1

    ga = GeneticAlgorithm(population_size, input_shape, output_shape, mutation_rate)

    # Check if a saved population exists
    population_file = "population.json"
    if os.path.exists(population_file):
        ga.load_population(population_file)
    else:
        print("No saved population found. Starting from scratch.")

    generations = 10
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        fitness_scores = ga.evaluate_population(environment)
        print(f"Population {generation + 1}: Fitness Scores: {fitness_scores}")
        ga.evolve(fitness_scores)
        print(f"Completed Generation {generation + 1}/{generations}\n")

        # Save the population after each generation
        ga.save_population(population_file)

    # Plot fitness progress
    ga.plot_generation_stats(output_file="generation_stats.png")
    print("Fitness progress plot saved as 'generation_stats.png'")

if __name__ == "__main__":
    main()