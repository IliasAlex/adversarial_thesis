import numpy as np
import torch
import torch.nn.functional as F
from utils.utils import extract_mel_spectrogram, add_normalized_noise, calculate_snr
from models.models import Autoencoder, UNet, Autoencoder_AudioCLIP_default, PasstAutoencoder

# autoencoder = Autoencoder_AudioCLIP_default()
# autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/best_audioclipautoencoder_model_default.pth'))
# autoencoder.to("cuda")


# autoencoder = Autoencoder()
# autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/best_autoencoder_model.pth'))
# autoencoder.to("cuda")


autoencoder = PasstAutoencoder()
autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/Passt_autoencoder.pth'))
autoencoder.to('cuda:1')


class DEAttack:
    def __init__(self, model, model_name, max_iter=20, population_size=10, mutation_factor=1.2, crossover_prob=0.6, epsilon=0.3, l2_weight=0, device='cuda', target_snr=5):
        self.model = model.to(device)
        self.model_name = model_name
        self.max_iter = max_iter
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.epsilon = epsilon
        self.device = device
        self.l2_weight = l2_weight
        self.target_snr = target_snr

    def fitness_score(self, audio, original_audio, original_label):
        self.model.eval()
        with torch.no_grad():
            if self.model_name == "Baseline" or self.model_name == "BaselineAvgPooling":
                features = extract_mel_spectrogram(audio, device=self.device).unsqueeze(0)
                #features = autoencoder(features)
                outputs = self.model(features)
            elif self.model_name == "AudioCLIP":
                features = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
                x = self.model.audioclip.audio._forward_pre_processing(features)
                x = 2 * (x - x.min()) / (x.max() - x.min() + 1e-6) - 1  # Normalize to [-1,1]
                x = autoencoder(x)
                x = self.model.audioclip.audio._forward_features(x)
                x = self.model.audioclip.audio._forward_reduction(x)
                outputs = self.model.classification_head(x)
            elif self.model_name == 'Passt':
                features = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
                data = self.model.mel(features)
                data = data.unsqueeze(1)
                data = autoencoder(data)
                outputs = self.model.net(data)[0]
            
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
            fitness = other_confidence - original_confidence #- l2_penalty * q1_regularization
            
            return fitness

    def initialize_population(self, original_audio):
        """
        Initializes the population for DE attack with noise proportional to the original audio,
        ensuring the noise is normalized to achieve a target SNR.

        Args:
            original_audio (numpy.ndarray): The original audio waveform.

        Returns:
            numpy.ndarray: Initial population for DE.
        """
        population = []
        noises = []
        clean_audios = []
        for _ in range(self.population_size):
            # Generate noise proportional to the original audio
            noise = np.random.uniform(
                -np.abs(original_audio),
                np.abs(original_audio)
            ) * self.epsilon

            # Add noise to original audio
            perturbed_audio = original_audio + noise

            # Normalize the noise to achieve the target SNR
            results = add_normalized_noise(original_audio, noise, self.target_snr)
            perturbed_audio = results['adversary']
            clean_audios.append(results['clean_audio'])
            
            # Collect the noise and perturbed audio
            noises.append(noise)
            population.append(perturbed_audio)

        return np.array(population), noises, clean_audios

    def mutate(self, population, original_audio):
        mutated = []
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)

            # Revert back to DE/rand/1 for better exploration
            mutant_vector = population[a] + self.mutation_factor * (population[b] - population[c])

            # Clip perturbations to stay within epsilon limits
            mutant_vector = np.clip(mutant_vector, original_audio - self.epsilon, original_audio + self.epsilon)
            
            mutated.append(mutant_vector)

        return np.array(mutated)

    def crossover(self, target, mutant):
        trial = np.copy(target)
        for i in range(len(target)):
            if np.random.rand() < self.crossover_prob:
                trial[i] = mutant[i]
        return trial

    def attack(self, original_audio, original_label):
        population, noises, clean_audios = self.initialize_population(original_audio)
        best_individual = None
        best_score = -np.inf
        
        for iteration in range(self.max_iter):
            mutated_population = self.mutate(population, original_audio)
            for i in range(self.population_size):
                trial_vector = self.crossover(population[i], mutated_population[i])
                noises[i] = trial_vector - original_audio
                results = add_normalized_noise(original_audio, noises[i], self.target_snr)
                trial_vector = results['adversary']
                clean_audios[i] = results['clean_audio']

                trial_score = self.fitness_score(trial_vector, original_audio, original_label)
                target_score = self.fitness_score(population[i], original_audio, original_label)

                if trial_score > target_score:
                    population[i] = trial_vector

                if trial_score > best_score:
                    best_individual = trial_vector
                    best_score = trial_score
                    best_clean_audio = clean_audios[i]

            if best_score > 0.99:  # Early stopping if strong adversarial example is found
                print("High-confidence adversarial example found!")
                break

        if best_score > 0:
            print(f"Final SNR: {calculate_snr(best_clean_audio, best_individual - best_clean_audio)}")
            return best_individual, iteration + 1, best_score, best_clean_audio

        print("Failed to find an adversarial example.")
        return None, self.max_iter, best_score, None
