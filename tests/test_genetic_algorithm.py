import unittest
import numpy as np
from utils.genetic_algorithm import GeneticAlgorithm

class TestGeneticAlgorithm(unittest.TestCase):
    def test_elitism(self):
        ga = GeneticAlgorithm(population_size=5, input_shape=(13,), output_shape=5, mutation_rate=0.1)

        # Przykładowa populacja
        ga.population = [
            {"speed": 2.0, "acceleration": 0.1, "braking_force": 0.2, "turn_angle": 30, "model": None},
            {"speed": 1.5, "acceleration": 0.15, "braking_force": 0.25, "turn_angle": 25, "model": None},
            {"speed": 2.5, "acceleration": 0.2, "braking_force": 0.3, "turn_angle": 35, "model": None},
            {"speed": 1.0, "acceleration": 0.05, "braking_force": 0.1, "turn_angle": 20, "model": None},
            {"speed": 3.0, "acceleration": 0.2, "braking_force": 0.3, "turn_angle": 40, "model": None}
        ]

        # Fitness scores (najlepszy osobnik ma najwyższy wynik)
        fitness_scores = [10, 20, 30, 40, 50]

        # Ewolucja
        ga.evolve(fitness_scores)

        # Sprawdź, czy najlepszy osobnik przeszedł do nowej populacji
        best_individual = ga.population[0]
        self.assertEqual(best_individual["speed"], 3.0)
        self.assertEqual(best_individual["acceleration"], 0.2)
        self.assertEqual(best_individual["braking_force"], 0.3)
        self.assertEqual(best_individual["turn_angle"], 40)

if __name__ == "__main__":
    unittest.main()