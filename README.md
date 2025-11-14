# 🤖 Reinforcement Learning CartPole Agent (PPO)

A solution to the classic CartPole environment using the Proximal Policy Optimization (PPO) algorithm. This project demonstrates foundational skills in Reinforcement Learning (RL) and optimal control systems.

## ✨ Project Achievement

The trained agent achieved a near-perfect score, balancing the pole for **499 steps** and obtaining the maximum possible reward (500.00), effectively solving the environment.

## 🌟 Features

- **Algorithm:** Implementation of **PPO** via the Stable-Baselines3 library.
- **Environment:** Solves the **Gymnasium CartPole-v1** control environment.
- **Training vs. Testing:** Clear separation of training and evaluation phases.
- **Optimal Control:** Demonstrates learning through trial-and-error to achieve optimal control in a dynamic system.

## 🛠️ Tech Stack

- **Language:** Python
- **RL Framework:** Stable-Baselines3
- **Environment:** Gymnasium
- **Deep Learning Backend:** PyTorch (required for Stable-Baselines3)

## 🚀 How to Run

### 1. Prerequisites
- The environment must be installed (see requirements.txt).
- The model file (`ppo_cartpole_trained.zip`) must be present in the root directory.

### 2. Install Dependencies
```bash
pip install -r requirements.txt

### 3. Train the Agent (Optional)
To reproduce the perfect agent (this takes a few minutes):
py train_ppo.py

### 4. Test the Agent (The Demo)
To visually observe the agent balancing the pole:
py test_ppo.py
