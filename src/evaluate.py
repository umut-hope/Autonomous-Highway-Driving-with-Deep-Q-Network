import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import DQN
import os

# Görselleştirme için dizin oluştur
os.makedirs("assets/evaluation", exist_ok=True)

def evaluate_agent():
    print("🎬 Değerlendirme başlıyor...")
    
    # Eğitimi yaptığımız ortam ayarları ile aynı olmalı
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure({
        "observation": {"type": "Kinematics", "vehicles_count": 15, "features": ["presence", "x", "y", "vx", "vy"]},
        "action": {"type": "DiscreteMetaAction"},
        "duration": 40,
        "vehicles_density": 1.0,
        "screen_width": 800,
        "screen_height": 200
    })

    # Eğittiğimiz modeli yükle
    model = DQN.load("checkpoints/model_final")
    
    obs, _ = env.reset()
    frames = []
    
    action_map = {0: "SOLA", 1: "SABIT", 2: "SAGA", 3: "HIZLAN", 4: "YAVASLA"}
    
    print("🚗 Sürüş Başlıyor...")
    for i in range(200): # 200 adım boyunca izle
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        
        # Kararları terminale yazdır (Aracın ne yaptığını takip et)
        if i % 10 == 0:
            print(f"Adım {i}: {action_map[int(action)]}")
            
        frames.append(env.render())
        if done or truncated:
            break
            
    # Videoyu kaydet
    imageio.mimsave("assets/evaluation/final_drive.mp4", frames, fps=15)
    print("✅ Video 'assets/evaluation/final_drive.mp4' dosyasına kaydedildi!")
    env.close()

if __name__ == "__main__":
    evaluate_agent()