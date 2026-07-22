# Autonomous Highway Driving with Deep Q-Network (DQN)

**Course:** CMP4501 – Introduction to AI

**Author:** Okan Umut Özen

---

## 1. Project Overview
This project implements an autonomous highway driving agent using the **Deep Q-Network (DQN)** reinforcement learning algorithm. The agent is tasked with navigating through dense traffic, maintaining an optimal speed, and avoiding collisions within a `highway-v0` environment.

---

## 2. Methodology & Architecture

### Reward Function Strategy
To achieve a balance between speed and safety, the following reward structure was implemented:
- **Collision Penalty:** -1.0
- **Speed Reward:** +0.4 (Targeting 20-30 m/s)
- **Lane Efficiency:** Balanced with right-lane and lane-change rewards.

### DQN Configuration
The agent utilizes an `MlpPolicy` with a robust architecture:
- **Neural Network:** Two hidden layers, each with 256 neurons (`[256, 256]`).
- **Hyperparameters:**
  - **Learning Rate:** 5e-4
  - **Buffer Size:** 15,000 steps
  - **Gamma (Discount Factor):** 0.99
  - **Exploration:** Initial ε=1.0, decaying to 0.05 over 30% of training steps.

---

## 3. Training & Evaluation Pipeline

### Training (`train.py`)
Training was conducted over **50,000 timesteps**. The process included a 1,000-step exploration phase where the agent performed random actions to populate the replay buffer, ensuring a diverse learning experience. The model state was saved at start, halfway, and completion points to ensure reproducibility.

### Evaluation (`evaluate.py`)
Post-training evaluation was performed to validate the agent's generalization capabilities.
- **Method:** 200-step deterministic simulation.
- **Output:** The agent's decision-making process is captured and rendered into an MP4 file located in `assets/evaluation/final_drive.mp4`.

---

## 4. Challenges & Technical Hurdles

**1. Optimization of the Action Space:**
Initially, the agent struggled with sudden lane changes. By utilizing `DiscreteMetaAction`, we constrained the agent to meaningful maneuvers (change lane left/right, speed up/down), which drastically reduced the convergence time.

**2. Visualizing Performance:**
To bridge the gap between abstract reinforcement learning metrics and visual performance, a dedicated evaluation script was developed. This script bypasses the training overhead and captures raw frames directly from the environment to generate high-fidelity driving demonstrations.

---

## 5. Work Breakdown Structure

| Category | Task | Status |
| :--- | :--- | :--- |
| **Infrastructure** | Environment Configuration | Done |
| **Model Design** | DQN Network Architecture [256, 256] | Done |
| **Training** | 50k Timestep Execution | Done |
| **Evaluation** | MP4 Performance Capture | Done |
| **Documentation** | Technical README Write-up | Done |

---
**Conclusion:** The agent successfully learned a policy that prioritizes safe navigation in high-density traffic. This project serves as a practical implementation of deep reinforcement learning in a constrained dynamic environment.