import numpy as np
import torch
import torch.nn.functional as F
import random

class PSOAttack:
    def __init__(self, model, max_iter=100, swarm_size=150, epsilon=0.1, c1=2.0, c2=2.0, w_max=1.0, w_min=0.3, patience=20, mutation_rate=0.2, device='cuda'):
        self.model = model.to(device)
        self.max_iter = max_iter
        self.swarm_size = swarm_size
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.patience = patience
        self.mutation_rate = mutation_rate
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
        r1, r2 = random.random(), random.random()
        inertia = w * velocity
        cognitive = c1 * r1 * (personal_best - particle)
        social = c2 * r2 * (global_best - particle)
        return inertia + cognitive + social

    def clip_audio(self, audio, original_audio, epsilon):
        return np.clip(audio, original_audio - epsilon, original_audio + epsilon)

    def mutate_particle(self, particle):
        mutation = np.random.uniform(-self.mutation_rate, self.mutation_rate, size=particle.shape)
        return np.clip(particle + mutation, -1.0, 1.0)

    def attack(self, original_audio, target_label):
        particles, velocities = self.initialize_particles(original_audio)
        personal_best = np.copy(particles)
        global_best = np.copy(particles[np.argmax([self.fitness_score(p, target_label) for p in particles])])

        personal_best_scores = [self.fitness_score(p, target_label) for p in particles]
        global_best_score = max(personal_best_scores)

        no_improvement = 0

        for iteration in range(self.max_iter):
            w = self.w_max - (iteration / self.max_iter) * (self.w_max - self.w_min)

            for i in range(self.swarm_size):
                velocities[i] = self.update_velocity(velocities[i], particles[i], personal_best[i], global_best, w, self.c1, self.c2)
                particles[i] = self.clip_audio(particles[i] + velocities[i], original_audio, self.epsilon)

                score = self.fitness_score(particles[i], target_label)
                if score > personal_best_scores[i]:
                    personal_best[i] = np.copy(particles[i])
                    personal_best_scores[i] = score

                if score > global_best_score:
                    global_best = np.copy(particles[i])
                    global_best_score = score
                    no_improvement = 0  # Reset if improvement found

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness Score: {global_best_score:.4f}")
            print(f"Sample Particle Scores: {[round(score, 4) for score in personal_best_scores[:5]]}")

            # Apply mutation if no improvement over 3/4 of the patience period
            if no_improvement >= 3 * self.patience // 4:
                print("Applying mutation to escape local minima.")
                temp_particles = [self.mutate_particle(p) for p in personal_best[:self.swarm_size // 4]]
                particles[:len(temp_particles)] = temp_particles

            # Early stopping if no improvement for the patience period
            if no_improvement >= self.patience:
                print("Early stopping due to no improvement.")
                break

            # Check for adversarial example with a significant improvement threshold
            if global_best_score > 0.1:  # Adjust threshold as needed
                print("Adversarial example found!")
                return global_best

            no_improvement += 1

        print("Failed to find an adversarial example within the maximum iterations.")
        return None
