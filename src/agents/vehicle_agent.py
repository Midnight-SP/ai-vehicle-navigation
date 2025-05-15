import numpy as np 
import matplotlib.pyplot as plt

class VehicleAgent:
    def __init__(self, model, environment):
        self.model = model
        self.environment = environment
        self.state = None
        self.reward = 0
        self.done = False

    def train(self, episodes):
        rewards_per_episode = []

        for episode in range(episodes):
            print(f"Episode {episode + 1}/{episodes}")
            self.state = self.environment.reset()
            self.done = False
            total_reward = 0

            while not self.done:
                epsilon = max(0.1, 1 - episode / episodes)  # Decay epsilon over time
                action = self.act(self.state, epsilon=epsilon)
                next_state, reward, self.done, _ = self.environment.step(action)
                self.state = next_state
                total_reward += reward

            rewards_per_episode.append(total_reward)
            print(f"Episode {episode + 1}/{episodes}: Total Reward: {total_reward}")

        self.plot_training_progress(rewards_per_episode)

    def plot_training_progress(self, rewards_per_episode):
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_per_episode, label="Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress")
        plt.legend()
        plt.grid()
        plt.savefig("training_progress.png")  # Save the plot as an image file
        plt.close()

    def act(self, state, epsilon=0.1):
        # Jeśli eksploracja, wybierz losową akcję
        if np.random.rand() < epsilon:
            return np.random.choice(self.environment.action_space.n)

        # Wykorzystaj model sieci neuronowej do przewidywania akcji
        q_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])  # Wybierz akcję z najwyższą wartością Q

    def evaluate(self, test_episodes):
        total_reward = 0
        for episode in range(test_episodes):
            self.state = self.environment.reset()
            self.done = False
            
            while not self.done:
                action = self.act(self.state)
                self.state, self.reward, self.done, _ = self.environment.step(action)
                total_reward += self.reward
        
        average_reward = total_reward / test_episodes
        print(f"Average Reward over {test_episodes} episodes: {average_reward}")