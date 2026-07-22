import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

# Çevreyi oluştur
env = gym.make("highway-v0", render_mode=None)
# Eğitiminle aynı ayarları buraya da koymalısın
env.unwrapped.configure({"observation": {"type": "Kinematics", "vehicles_count": 15}, "duration": 40})

# Eğittiğin modeli yükle
model = DQN.load("checkpoints/model_final")

rewards = []
lengths = []

print("📊 Model test ediliyor...")
for _ in range(30):  # 30 farklı senaryoda test et
    obs, _ = env.reset()
    done = truncated = False
    total_reward = 0
    steps = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
    rewards.append(total_reward)
    lengths.append(steps)

# Grafiği Çiz
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(rewards, color='blue')
plt.title("Bölüm Başına Ödül")
plt.subplot(1, 2, 2)
plt.plot(lengths, color='green')
plt.title("Hayatta Kalma Süresi (Adım)")
plt.tight_layout()
plt.savefig("assets/final_grafik.png")
print("✅ Grafik 'assets/final_grafik.png' olarak kaydedildi!")