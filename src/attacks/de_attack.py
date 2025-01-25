import numpy as np
import torch
import torch.nn.functional as F
from utils.utils import extract_mel_spectrogram
from models.models import Autoencoder, UNet

class DEAttack:
    def __init__(self, model, model_name, max_iter=20, population_size=10, mutation_factor=0.5, crossover_prob=0.5, epsilon=0.3, l2_weight=0, device='cuda'):
        self.model = model.to(device)
        self.model_name = model_name
        self.max_iter = max_iter
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.epsilon = epsilon
        self.device = device
        self.l2_weight = l2_weight

    def fitness_score(self, audio, original_audio, original_label):
        self.model.eval()
        with torch.no_grad():
            if self.model_name == "Baseline":
                features = extract_mel_spectrogram(audio, device=self.device)
            elif self.model_name == "AudioCLIP":
                features = audio / np.max(np.abs(audio))
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            autoencoder = Autoencoder()
            autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/best_autoencoder_model.pth'))
            autoencoder.to("cuda")
            features = autoencoder(features.unsqueeze(0))

            outputs = self.model(features)
            logits = F.softmax(outputs, dim=1)

            original_confidence = logits[0, original_label].item()
            other_confidence = logits[0].max().item() if logits[0].argmax() != original_label else logits[0].topk(2).values[1].item()

            # Compute L2 norm penalty (distance between adversarial and original audio)
            l2_penalty = np.linalg.norm(audio - original_audio)
            
            # Compute Q1 regularization term
            perturbation = audio - original_audio # compute perturbation
            epsilon = 1e-6  # Small constant to prevent division by zero
            q1_regularization = np.mean(np.abs(perturbation) / (np.abs(original_audio) + epsilon))

            # Combine the classification fitness with the L2 penalty
            fitness = other_confidence - original_confidence - l2_penalty * q1_regularization
            
            return fitness

    def initialize_population(self, original_audio):
        """
        Initializes the population for DE attack with noise proportional to the original audio.

        Args:
            original_audio (numpy.ndarray): The original audio waveform.

        Returns:
            numpy.ndarray: Initial population for DE.
        """
        # Generate noise proportional to the original audio
        noise = np.random.uniform(
            -np.abs(original_audio),
            np.abs(original_audio),
            size=(self.population_size, len(original_audio))
        ) * self.epsilon

        # Add noise to original audio to create initial population
        population = np.tile(original_audio, (self.population_size, 1)) + noise

        return np.clip(population, -1.0, 1.0)  # Ensure audio values are within valid range


    def mutate(self, population):
        mutated = []
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)

            mutant_vector = population[a] + self.mutation_factor * (population[b] - population[c])
            mutated.append(mutant_vector)
        return np.array(mutated)

    def crossover(self, target, mutant):
        trial = np.copy(target)
        for i in range(len(target)):
            if np.random.rand() < self.crossover_prob:
                trial[i] = mutant[i]
        return trial

    def clip_audio(self, audio, original_audio, epsilon):
        return np.clip(audio, original_audio - epsilon, original_audio + epsilon)

    def attack(self, original_audio, original_label):
        population = self.initialize_population(original_audio)
        best_individual = None
        best_score = -np.inf

        for iteration in range(self.max_iter):
            mutated_population = self.mutate(population)
            for i in range(self.population_size):
                trial_vector = self.crossover(population[i], mutated_population[i])
                trial_vector = self.clip_audio(trial_vector, original_audio, self.epsilon)

                trial_score = self.fitness_score(trial_vector, original_audio, original_label)
                target_score = self.fitness_score(population[i], original_audio, original_label)

                if trial_score > target_score:
                    population[i] = trial_vector

                if trial_score > best_score:
                    best_individual = trial_vector
                    best_score = trial_score

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness Score: {best_score:.4f}")

            if best_score > 0:
                print("Adversarial example found!")
                return best_individual, iteration + 1, best_score

        print("Failed to find an adversarial example within the maximum iterations.")
        return None, self.max_iter, best_score
