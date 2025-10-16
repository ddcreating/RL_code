#!/usr/bin/env python3
"""
全面的SAC Hopper模型评估脚本
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

def comprehensive_evaluation(model_path, n_episodes=100):
    """
    全面评估SAC Hopper模型
    """
    print("=== SAC Hopper 全面评估 ===\n")
    
    # 1. 加载模型
    print(f"加载模型: {model_path}")
    try:
        model = SAC.load(model_path)
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 2. 创建环境
    env = gym.make("Hopper-v4")
    
    # 3. 使用stable_baselines3的evaluate_policy进行评估
    print(f"\n开始评估 {n_episodes} 个episodes...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_episodes,
        deterministic=True,
        return_episode_rewards=False
    )
    
    print(f"📊 评估结果:")
    print(f"   平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   评估episodes: {n_episodes}")
    
    # 4. 详细分析每个episode
    print(f"\n详细分析每个episode...")
    episode_rewards = []
    episode_lengths = []
    
    for i in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (i + 1) % 20 == 0:
            print(f"   完成 {i + 1}/{n_episodes} episodes...")
    
    env.close()
    
    # 5. 统计分析
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    
    print(f"\n📈 详细统计:")
    print(f"   奖励统计:")
    print(f"     平均: {np.mean(episode_rewards):.2f}")
    print(f"     标准差: {np.std(episode_rewards):.2f}")
    print(f"     最小值: {np.min(episode_rewards):.2f}")
    print(f"     最大值: {np.max(episode_rewards):.2f}")
    print(f"     中位数: {np.median(episode_rewards):.2f}")
    
    print(f"   Episode长度统计:")
    print(f"     平均: {np.mean(episode_lengths):.2f}")
    print(f"     标准差: {np.std(episode_lengths):.2f}")
    print(f"     最小值: {np.min(episode_lengths):.2f}")
    print(f"     最大值: {np.max(episode_lengths):.2f}")
    
    # 6. 性能评估
    print(f"\n🎯 性能评估:")
    if np.mean(episode_rewards) > 3000:
        print("   🏆 优秀! 模型性能很好")
    elif np.mean(episode_rewards) > 2000:
        print("   ✅ 良好! 模型性能不错")
    elif np.mean(episode_rewards) > 1000:
        print("   ⚠️  一般! 模型需要更多训练")
    else:
        print("   ❌ 较差! 模型需要大量训练")
    
    # 7. 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 奖励分布
    ax1.hist(episode_rewards, bins=20, alpha=0.7, color='blue')
    ax1.axvline(np.mean(episode_rewards), color='red', linestyle='--', label=f'平均值: {np.mean(episode_rewards):.1f}')
    ax1.set_xlabel('Episode奖励')
    ax1.set_ylabel('频次')
    ax1.set_title('Episode奖励分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode长度分布
    ax2.hist(episode_lengths, bins=20, alpha=0.7, color='green')
    ax2.axvline(np.mean(episode_lengths), color='red', linestyle='--', label=f'平均值: {np.mean(episode_lengths):.1f}')
    ax2.set_xlabel('Episode长度')
    ax2.set_ylabel('频次')
    ax2.set_title('Episode长度分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hopper_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 图表已保存为 'hopper_evaluation.png'")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

if __name__ == "__main__":
    # 模型路径
    model_dir = "./sac_local/sac_models/sac_hopper"
    
    # 找到最佳模型
    best_model_path = os.path.join(model_dir, "best_model.zip")
    final_model_path = os.path.join(model_dir, "sac_hopper.zip")
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
    elif os.path.exists(final_model_path):
        model_path = final_model_path
    else:
        print("❌ 没有找到模型文件!")
        exit(1)
    
    # 进行全面评估
    results = comprehensive_evaluation(model_path, n_episodes=100)
    
    if results:
        print(f"\n🎉 评估完成!")
        print(f"建议: 如果平均奖励低于2000，考虑增加训练步数到500k-1M步")
