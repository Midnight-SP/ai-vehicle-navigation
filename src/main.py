# ai-vehicle-navigation/src/main.py

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from agents.vehicle_agent import VehicleAgent
from environment.simulation_environment import SimulationEnvironment
from models.reinforcement_learning_model import ReinforcementLearningModel
from utils.genetic_algorithm import GeneticAlgorithm

def load_route(route_file):
    with open(route_file, 'r') as file:
        return json.load(file)
    
def save_best_vehicle_path(path, generation):
    # Konwersja ścieżki z ndarray na listy
    path_as_list = [position.tolist() for position in path]
    file_name = f"best_vehicle_path_gen_{generation}.json"
    with open(file_name, "w") as file:
        json.dump({"path": path_as_list}, file, indent=4)
    print(f"Best vehicle path saved to {file_name}")

def visualize_route(route_data):
    # Rysowanie ścian
    for wall in route_data['route']['walls']:
        start = wall['start']
        end = wall['end']
        plt.plot(
            [start['longitude'], end['longitude']],
            [start['latitude'], end['latitude']],
            color='black', linewidth=2, label='Wall' if 'Wall' not in plt.gca().get_legend_handles_labels()[1] else ""
        )

    # Rysowanie punktów kontrolnych
    for checkpoint in route_data['route']['checkpoints']:
        start = checkpoint['start']
        end = checkpoint['end']
        plt.plot(
            [start['longitude'], end['longitude']],
            [start['latitude'], end['latitude']],
            color='blue', linestyle='--', linewidth=1.5, label='Checkpoint' if 'Checkpoint' not in plt.gca().get_legend_handles_labels()[1] else ""
        )

    # Rysowanie pozycji początkowej pojazdu
    start_position = route_data['route']['start_position']
    plt.scatter(
        start_position['longitude'], start_position['latitude'],
        color='red', label='Start Position'
    )

    # Ustawienia wykresu
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Route Visualization')
    plt.legend()
    plt.grid()
    plt.axis('equal')  # Zachowanie proporcji osi

    # Zapisz wykres do pliku
    plt.savefig("route.png")
    plt.close()  # Zamknij wykres, aby zwolnić pamięć

def visualize_best_vehicle_path(route_data, best_vehicle, environment):
    # Resetuj środowisko do początkowego stanu
    state = environment.reset()
    done = False
    positions = [environment.vehicle_position.copy()]  # Śledzenie pozycji pojazdu

    # Symuluj ruch najlepszego pojazdu
    while not done:
        action = best_vehicle["model"].predict(np.expand_dims(state, axis=0)).argmax()
        state, _, done, _ = environment.step(action)
        positions.append(environment.vehicle_position.copy())
        print(f"Vehicle position: {environment.vehicle_position}")  # Log pozycji pojazdu

    # Rysowanie trasy
    plt.figure(figsize=(10, 6))

    # Rysowanie ścian
    for wall in route_data['route']['walls']:
        start = wall['start']
        end = wall['end']
        plt.plot(
            [start['longitude'], end['longitude']],
            [start['latitude'], end['latitude']],
            color='black', linewidth=2, label='Wall' if 'Wall' not in plt.gca().get_legend_handles_labels()[1] else ""
        )

    # Rysowanie punktów kontrolnych
    for checkpoint in route_data['route']['checkpoints']:
        start = checkpoint['start']
        end = checkpoint['end']
        plt.plot(
            [start['longitude'], end['longitude']],
            [start['latitude'], end['latitude']],
            color='blue', linestyle='--', linewidth=1.5, label='Checkpoint' if 'Checkpoint' not in plt.gca().get_legend_handles_labels()[1] else ""
        )

    # Rysowanie pozycji początkowej pojazdu
    start_position = route_data['route']['start_position']
    plt.scatter(
        start_position['longitude'], start_position['latitude'],
        color='red', label='Start Position'
    )

    # Rysowanie trasy pojazdu
    positions = np.array(positions)
    plt.plot(positions[:, 1], positions[:, 0], color='green', label='Best Vehicle Path')

    # Ustawienia wykresu
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Best Vehicle Path')
    plt.legend()
    plt.grid()
    plt.axis('equal')  # Zachowanie proporcji osi

    # Zapisz wykres do pliku
    plt.savefig("best_vehicle_path.png")
    plt.close()  # Zamknij wykres, aby zwolnić pamięć

def main():
    route_data = load_route('data/routes/sample_route.json')

    # Wizualizacja trasy
    visualize_route(route_data)

    environment = SimulationEnvironment(route_data, genes={
        "speed": 2.0,
        "acceleration": 0.1,
        "braking_force": 0.2,
        "turn_angle": 30
    })

    # Genetic Algorithm parameters
    population_size = 10
    input_shape = (13,)
    output_shape = 5
    mutation_rate = 0.2

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

        # Znajdź najlepszego osobnika
        best_index = np.argmax(fitness_scores)
        best_vehicle = ga.population[best_index]

        # Zapisz ścieżkę najlepszego pojazdu
        save_best_vehicle_path(environment.path, generation + 1)

        # Wizualizacja trasy najlepszego pojazdu
        visualize_best_vehicle_path(route_data, best_vehicle, environment)

        # Ewolucja populacji
        ga.evolve(fitness_scores)
        print(f"Completed Generation {generation + 1}/{generations}\n")

        # Save the population after each generation
        ga.save_population(population_file)

if __name__ == "__main__":
    main()