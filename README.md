# Adversarial attacks and robustness in deep neural networks for sound event detection
Master's Thesis on Artificial Intelligence by Ilias Alexandropoulos

Thesis Supervisor: Theodoros Giannakopoulos

# Training

```
python3 main.py -c experiments/config.yaml -m train
```
- `experiments/config.yaml`: Path to the training config. 

# Attacks

## PSO Attack
```
python3 main.py -c experiments/pso_attack.yaml -m attack
```
- `experiments/pso_attack.yaml`: Path to the pso attack config file.

## DE Attack
```
python3 main.py -c experiments/de_attack.yaml -m attack
```
- `experiments/de_attack.yaml`: Path to the de attack config file.

