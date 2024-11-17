import numpy as np
import torch
import torch.nn.functional as F
import random

class PSOAttack:
    def __init__(self, model, max_iter=20, swarm_size=10, epsilon=0.3, c1=2.0, c2=2.0, w_max=0.9, w_min=0.1, device='cuda'):
        self.model = model.to(device)
        self.max_iter = max_iter
        self.swarm_size = swarm_size
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.device = device

    def fitness_score(self, audio, target_label):
        """
        Compute the fitness score based on the model's prediction.
        """
        self.model.eval()
        with torch.no_grad():
            audio_tensor = torch.tensor(audio, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            outputs = self.model(audio_tensor)
            logits = F.softmax(outputs, dim=1)
            target_confidence = logits[0, target_label].item()
            other_confidence = logits[0].max().item() if logits[0].argmax() != target_label else 0
            return target_confidence - other_confidence

    def initialize_particles(self, original_audio):
        """
        Initialize particles with random noise around the original audio.
        """
        particles = []
        velocities = []
        for _ in range(self.swarm_size):
            noise = np.random.uniform(-self.epsilon, self.epsilon, size=original_audio.shape)
            particle = np.clip(original_audio + noise, -1.0, 1.0)
            velocity = np.zeros_like(original_audio)
            particles.append(particle)
            velocities.append(velocity)
        return np.array(particles), np.array(velocities)

    def update_velocity(self, velocity, particle, personal_best, global_best, w, c1, c2):
        """
        Update the velocity of a particle.
        """
        r1, r2 = random.random(), random.random()
        inertia = w * velocity
        cognitive = c1 * r1 * (personal_best - particle)
        social = c2 * r2 * (global_best - particle)
        return inertia + cognitive + social

    def clip_audio(self, audio, original_audio, epsilon):
        """
        Clip the audio to ensure perturbation is within bounds.
        """
        return np.clip(audio, original_audio - epsilon, original_audio + epsilon)

    def attack(self, original_audio, target_label):
        """
        Perform the PSO attack to generate an adversarial example.
        """
        # Initialize particles and velocities
        particles, velocities = self.initialize_particles(original_audio)
        personal_best = np.copy(particles)
        global_best = np.copy(particles[np.argmax([self.fitness_score(p, target_label) for p in particles])])

        personal_best_scores = [self.fitness_score(p, target_label) for p in particles]
        global_best_score = max(personal_best_scores)

        # Main optimization loop
        for iteration in range(self.max_iter):
            w = self.w_max - (iteration / self.max_iter) * (self.w_max - self.w_min)

            for i in range(self.swarm_size):
                # Update velocity and position of each particle
                velocities[i] = self.update_velocity(velocities[i], particles[i], personal_best[i], global_best, w, self.c1, self.c2)
                particles[i] = self.clip_audio(particles[i] + velocities[i], original_audio, self.epsilon)

                # Evaluate fitness
                score = self.fitness_score(particles[i], target_label)
                if score > personal_best_scores[i]:
                    personal_best[i] = np.copy(particles[i])
                    personal_best_scores[i] = score

                if score > global_best_score:
                    global_best = np.copy(particles[i])
                    global_best_score = score

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness Score: {global_best_score:.4f}")

            # Check if an adversarial example is found
            if global_best_score > 0.1:
                print("Adversarial example found!")
                return global_best

        print("Failed to find an adversarial example within the maximum iterations.")
        return None
