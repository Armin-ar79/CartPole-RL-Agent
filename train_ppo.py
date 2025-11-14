import gymnasium as gym
from stable_baselines3 import PPO

# --- 1. تعریف محیط ---
# برای آموزش نیازی به نمایش گرافیکی نیست، پس render_mode را حذف می‌کنیم
env = gym.make("CartPole-v1") 

# --- 2. تعریف مدل PPO ---
# MlpPolicy: نوع شبکه عصبی عامل (یک شبکه ساده چند لایه)
# env: محیطی که عامل باید در آن یاد بگیرد
# tensorboard_log: برای ذخیره نمودارهای پیشرفت یادگیری (اختیاری)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./cartpole_ppo_tensorboard/"
)

print("شروع آموزش مدل PPO...")
# --- 3. آموزش مدل ---
# 10,000 گام برای یادگیری تعادل در محیط CartPole کافی است
model.learn(total_timesteps=10000)
print("آموزش کامل شد.")

# --- 4. ذخیره مدل آموزش دیده ---
# ذخیره مدل برای استفاده بعدی
model.save("ppo_cartpole_trained")
print("مدل آموزش دیده با نام 'ppo_cartpole_trained.zip' ذخیره شد.")

# --- 5. بستن محیط ---
env.close()