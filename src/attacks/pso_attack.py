import numpy as np
import torch
import torch.nn.functional as F
import random
from utils.utils import extract_mel_spectrogram, calculate_snr, add_normalized_noise
from models.models import Autoencoder,UNet, Autoencoder_AudioCLIP_default, PasstAutoencoder

# autoencoder = Autoencoder_AudioCLIP_default()
# autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/best_audioclipautoencoder_model_default.pth'))
# autoencoder.to("cuda")

# autoencoder = Autoencoder()
# autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/best_autoencoder_model.pth'))
# autoencoder.to("cuda")

# autoencoder = PasstAutoencoder()
# autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/Passt_autoencoder.pth'))
# autoencoder.to('cuda:1')

class PSOAttack:
    def __init__(self, model, model_name, max_iter=20, swarm_size=10, epsilon=0.3, c1=0.7, c2=0.7, w_max=0.9, w_min=0.1, l2_weight=0, device='cuda', target_snr=5):
        self.model = model.to(device)
        self.model_name = model_name
        self.max_iter = max_iter
        self.swarm_size = swarm_size
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.device = device
        self.l2_weight = l2_weight
        self.target_snr = target_snr
        
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
            if self.model_name == "Baseline" or self.model_name == 'BaselineAvgPooling':
                # Convert waveform to mel-spectrogram
                features = extract_mel_spectrogram(audio, device=self.device).unsqueeze(0)
                #features = autoencoder(features)
                # Pass the mel-spectrogram to the model            
                outputs = self.model(features)
            elif self.model_name == "AudioCLIP":
                features = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                x = self.model.audioclip.audio._forward_pre_processing(features)
                print(f"Min {x.min()}, Max {x.max()}")
                x = 2 * (x - x.min()) / (x.max() - x.min() + 1e-6) - 1  # Normalize to [-1,1]
                #x = autoencoder(x)
                x = self.model.audioclip.audio._forward_features(x)
                x = self.model.audioclip.audio._forward_reduction(x)
                #emb = self.model.audioclip.audio._forward_classifier(x)
                outputs = self.model.classification_head(x)
                #outputs = self.model(features)
            elif self.model_name == 'Passt':
                features = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
                data = self.model.mel(features)
                data = data.unsqueeze(1)
                #data = autoencoder(data)
                outputs = self.model.net(data)[0]
                
            logits = F.softmax(outputs, dim=1)

            # Confidence for the original class
            original_confidence = logits[0, original_label].item()

            # Maximum confidence for any class other than the original class
            other_confidence = logits[0].max().item() if logits[0].argmax() != original_label else logits[0].topk(2).values[1].item()
                        
            # # Compute L2 norm penalty (distance between adversarial and original audio)
            # l2_penalty = np.linalg.norm(audio - original_audio)
            
            # # Compute Q1 regularization term
            # perturbation = audio - original_audio # compute perturbation
            # epsilon = 1e-6  # Small constant to prevent division by zero
            # q1_regularization = np.mean(np.abs(perturbation) / (np.abs(original_audio) + epsilon))


            # Combine the classification fitness with the L2 penalty
            fitness = other_confidence - original_confidence #- l2_penalty * self.l2_weight

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
        original_audios = []
        noises = []
        
        for _ in range(self.swarm_size):
            noise = np.random.uniform(
                -np.abs(original_audio),  
                np.abs(original_audio)   
            ) * self.epsilon  
            
            particle = original_audio + noise
            
            results = add_normalized_noise(original_audio, particle-original_audio, self.target_snr)
            particle = results['adversary']
            original_audios.append(results['clean_audio'])
            
            noises.append(noise)            
            velocity = np.zeros_like(original_audio)
            particles.append(particle)
            velocities.append(velocity)

        return np.array(particles), np.array(velocities), original_audios, noises

    def update_velocity(self, velocity, particle, personal_best, global_best, w, c1, c2):
        r1, r2 = random.random(), random.random()
        inertia = w * velocity
        cognitive = c1 * r1 * (personal_best - particle)
        social = c2 * r2 * (global_best - particle)
        return inertia + cognitive + social

    def attack(self, original_audio, original_label, target_class=None):
        """
        Perform the PSO attack to generate an adversarial example.
        If target_class is provided, it performs a targeted attack.
        Otherwise, it performs a non-targeted attack.
        """
        # Initialize particles, velocities, and original audios
        particles, velocities, original_audios, noises = self.initialize_particles(original_audio)
        personal_best = np.copy(particles)

        # Determine the initial global best based on the type of attack
        if target_class is None:
            # Calculate fitness scores for each particle
            personal_best_scores = [self.fitness_score(p, p, original_label) for p in particles]
            # Find the index of the particle with the maximum fitness score
            global_best_index = np.argmax(personal_best_scores)
            # Set global_best and corresponding global original audio
            global_best = np.copy(particles[global_best_index])
            global_best_clean_audio = np.copy(original_audios[global_best_index])
            global_best_score = personal_best_scores[global_best_index]
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
                
                noises[i] += velocities[i]
                results = add_normalized_noise(original_audio, noises[i], self.target_snr)
                particles[i] = results['adversary']
                original_audios[i] = results['clean_audio']

                audio = np.copy(results['adversary'])
                
                # Evaluate fitness based on attack type
                if target_class is None:
                    score = self.fitness_score(audio=audio, original_audio=original_audio, original_label=original_label)
                else:
                    score = self.fitness_score_targeted(particles[i], target_class)

                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best[i] = audio
                    personal_best_scores[i] = score
                    
                # Update global best
                if score > global_best_score:
                    global_best = audio
                    global_best_score = score
                    global_best_clean_audio = original_audios[i]

            #print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness Score: {global_best_score:.4f}")

            # Check if an adversarial example is found
            if target_class is None and global_best_score > 0:
                print("Non-targeted adversarial example found!")
                final_confidence = global_best_score
                print(f"Calculated SNR: {calculate_snr(global_best_clean_audio, global_best-global_best_clean_audio)}")
                return global_best, iteration + 1, final_confidence, global_best_clean_audio
            elif target_class is not None and global_best_score > 0:
                print("Targeted adversarial example found!")
                final_confidence = global_best_score
                return global_best, iteration + 1, final_confidence

        print("Failed to find an adversarial example within the maximum iterations.")
        return None, self.max_iter, global_best_score, None
