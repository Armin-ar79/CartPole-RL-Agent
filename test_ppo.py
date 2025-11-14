import gymnasium as gym
from stable_baselines3 import PPO
import time

# --- 1. بارگذاری محیط شبیه‌سازی ---
# render_mode="human" برای نمایش بصری عملکرد
env = gym.make("CartPole-v1", render_mode="human")

# --- 2. بارگذاری مدل آموزش دیده ---
print("در حال بارگذاری مدل آموزش دیده...")
try:
    model = PPO.load("ppo_cartpole_trained.zip")
except FileNotFoundError:
    print("خطا: مدل ppo_cartpole_trained.zip پیدا نشد.")
    env.close()
    exit()

# --- 3. اجرای عامل هوشمند ---
print("شروع تست عامل هوشمند (q را برای خروج بزنید)...")
# شروع محیط
observation, info = env.reset()
total_reward = 0
done = False

# اجرای عامل در محیط برای 500 گام یا تا زمانی که میله بیفتد
for step in range(500):
    # مدل بر اساس observation (وضعیت محیط)، بهترین حرکت (action) را انتخاب می‌کند
    action, _states = model.predict(observation, deterministic=True)
    
    # اجرای حرکت
    observation, reward, terminated, truncated, info = env.step(action)
    
    # نمایش فریم
    env.render()
    
    # جمع‌آوری پاداش
    total_reward += reward
    
    # 0.01 ثانیه صبر کن تا شبیه‌سازی قابل مشاهده باشد
    time.sleep(0.01)

    if terminated or truncated:
        print(f"تست تمام شد. میله برای {step} گام سرپا ماند.")
        break

# --- 4. بستن محیط ---
env.close()
print(f"پاداش نهایی کسب شده: {total_reward:.2f}")