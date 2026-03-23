import gymnasium as gym
import os
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
#倒立摆（CartPole）控制
log_dir = "configs/"
os.makedirs(log_dir, exist_ok=True)
# 初始化环境
env = gym.make("CartPole-v1")
env = Monitor(env, log_dir)
#初始化模型
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    batch_size=32,
    gamma=0.99,
    verbose=0,
    tensorboard_log=log_dir,
)
#训练模型
model.learn(total_timesteps=10000)
#保存模型
model.save("models/problem4_DQN_pth")
#加载模型验证
model = DQN.load("models/problem4_DQN_pth",env=env)
#评估模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"评估结果 - 平均奖励: {mean_reward:.2f}, 标准差: {std_reward:.2f}")
#可视化
env = gym.make("CartPole-v1", render_mode="human")
for _ in range(5):
    obs, info = env.reset()  # 重置环境，获取初始观测
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # 模型预测动作
        obs, rewards, terminated, truncated, info = env.step(action)  # 执行动作
        done = terminated or truncated  # 判断回合是否结束
        env.render()  # 渲染环境画面
env.close()  # 关闭环境

