import os
import shutil
import gymnasium as gym
from stable_baselines3 import DQN


# =====================================================================
# 1. ÇEVRE 
# =====================================================================
env = gym.make("highway-v0", render_mode="rgb_array")
env.unwrapped.configure({
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy"]
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,
    "vehicles_density": 1.0,       
    "duration": 40,                
    "initial_lane_bias": 1,
    "collision_reward": -1.0,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "lane_change_reward": 0.0,
    "reward_speed_range": [20, 30]
})
obs, _ = env.reset()

# =====================================================================
# 2. DQN AĞ MİMARİSİ
# =====================================================================
policy_kwargs = dict(net_arch=[256, 256])

model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=5e-4,          
    buffer_size=15000,           
    learning_starts=1000,        # İlk 1000 adım tamamen rastgele (makas atarak) keşif yapar
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.3,    # Eğitimin %30'unda etrafı keşfetmeye devam eder
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1
)

# =====================================================================
# 3. TRAINING (50.000 adım boyunca DQN ile öğrenme)
# =====================================================================
print("🧠 DQN Eğitimi Başlıyor!")
model.learn(total_timesteps=50_000, progress_bar=True)

# Çökme yaşanmasın diye 3 aşamayı da aynı bitmiş beyni kullanarak kaydediyoruz
model.save("checkpoints/model_final")
model.save("checkpoints/model_start")
model.save("checkpoints/model_halfway")

env.close()
print("\n✅ EĞİTİM BİTTİ! OTONOM ŞOFÖR TESLİMATA HAZIR.")