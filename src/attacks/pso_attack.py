import numpy as np
import torch
import torch.nn.functional as F
import random
from utils.utils import extract_mel_spectrogram


class PSOAttack:
    def __init__(self, model, max_iter=20, swarm_size=10, epsilon=0.3, c1=0.7, c2=0.7, w_max=0.9, w_min=0.1, l2_weight=0, device='cuda'):
        self.model = model.to(device)
        self.max_iter = max_iter
        self.swarm_size = swarm_size
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.device = device
        self.l2_weight = l2_weight

    def fitness_score(self, audio, original_audio, original_label):
        """
        Compute the fitness score for a non-targeted attack with L2 norm penalty.
        The score is higher when the model predicts any class other than the original label,
        and penalized by the L2 norm of the perturbation.

        Args:
            audio (numpy.ndarray): The adversarial audio waveform.
            original_audio (numpy.ndarray): The original audio waveform.
            original_label (int): The true label of the original audio.
            l2_weight (float): The weight applied to the L2 norm penalty.

        Returns:
            float: The computed fitness score.
        """
        self.model.eval()
        with torch.no_grad():
            # Convert waveform to mel-spectrogram
            mel_tensor = extract_mel_spectrogram(audio, device=self.device)

            # Pass the mel-spectrogram to the model
            outputs = self.model(mel_tensor)
            logits = F.softmax(outputs, dim=1)

            # Confidence for the original class
            original_confidence = logits[0, original_label].item()

            # Maximum confidence for any class other than the original class
            other_confidence = logits[0].max().item() if logits[0].argmax() != original_label else logits[0].topk(2).values[1].item()

            # Compute L2 norm penalty (distance between adversarial and original audio)
            l2_penalty = np.linalg.norm(audio - original_audio)

            # Combine the classification fitness with the L2 penalty
            fitness = other_confidence - original_confidence - self.l2_weight * l2_penalty

            return fitness


    def fitness_score_targeted(self, audio, target_class):
        """
        Compute the fitness score for a targeted attack.
        The score is higher when the model predicts the target class with higher confidence.
        """
        self.model.eval()
        with torch.no_grad():
            # Convert waveform to mel-spectrogram
            mel_tensor = extract_mel_spectrogram(audio, device=self.device)

            # Pass the mel-spectrogram to the model
            outputs = self.model(mel_tensor)
            logits = F.softmax(outputs, dim=1)

            # Confidence for the target class
            target_confidence = logits[0, target_class].item()

            # Maximum confidence for any other class
            other_confidence = logits[0].max().item() if logits[0].argmax() != target_class else logits[0].topk(2).values[1].item()

            # Fitness score: maximize confidence in the target class
            return target_confidence - other_confidence

    def initialize_particles(self, original_audio):
        particles = []
        velocities = []
        for _ in range(self.swarm_size):
            noise = np.random.uniform(
                -np.abs(original_audio),  
                np.abs(original_audio)   
            ) * self.epsilon  
            particle = np.clip(original_audio + noise, -1.0, 1.0)
            velocity = np.zeros_like(original_audio)
            particles.append(particle)
            velocities.append(velocity)

        return np.array(particles), np.array(velocities)

    def update_velocity(self, velocity, particle, personal_best, global_best, w, c1, c2):
        r1, r2 = random.random(), random.random()
        inertia = w * velocity
        cognitive = c1 * r1 * (personal_best - particle)
        social = c2 * r2 * (global_best - particle)
        return inertia + cognitive + social

    def clip_audio(self, audio, original_audio, epsilon):
        return np.clip(audio, original_audio - epsilon, original_audio + epsilon)

    def attack(self, original_audio, original_label, target_class=None):
        """
        Perform the PSO attack to generate an adversarial example.
        If target_class is provided, it performs a targeted attack.
        Otherwise, it performs a non-targeted attack.
        """
        # Initialize particles and velocities
        particles, velocities = self.initialize_particles(original_audio)
        personal_best = np.copy(particles)

        # Determine the initial global best based on the type of attack
        if target_class is None:
            global_best = np.copy(particles[np.argmax([self.fitness_score(p, p, original_label) for p in particles])])
            personal_best_scores = [self.fitness_score(p, p, original_label) for p in particles]
            global_best_score = max(personal_best_scores)
        else:
            global_best = np.copy(particles[np.argmax([self.fitness_score_targeted(p, target_class) for p in particles])])
            personal_best_scores = [self.fitness_score_targeted(p, target_class) for p in particles]
            global_best_score = max(personal_best_scores)

        # Main optimization loop
        for iteration in range(self.max_iter):
            w = self.w_max - (iteration / self.max_iter) * (self.w_max - self.w_min)

            for i in range(self.swarm_size):
                # Update velocity and position of each particle
                velocities[i] = self.update_velocity(velocities[i], particles[i], personal_best[i], global_best, w, self.c1, self.c2)
                particles[i] = self.clip_audio(particles[i] + velocities[i], original_audio, self.epsilon)

                # Evaluate fitness based on attack type
                if target_class is None:
                    score = self.fitness_score(audio=particles[i], original_audio=original_audio, original_label= original_label)
                else:
                    score = self.fitness_score_targeted(particles[i], target_class)

                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best[i] = np.copy(particles[i])
                    personal_best_scores[i] = score

                # Update global best
                if score > global_best_score:
                    global_best = np.copy(particles[i])
                    global_best_score = score

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness Score: {global_best_score:.4f}")

            # Check if an adversarial example is found
            if target_class is None and global_best_score > 0:
                print("Non-targeted adversarial example found!")
                final_confidence = global_best_score
                return global_best, iteration + 1, final_confidence
            elif target_class is not None and global_best_score > 0:
                print("Targeted adversarial example found!")
                final_confidence = global_best_score
                return global_best, iteration + 1, final_confidence

        print("Failed to find an adversarial example within the maximum iterations.")
        return None, self.max_iter, global_best_score
