import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from simulated_k8s_env import SimulatedK8sEnv
from dqn_agent import DQN
import random
import platform

# 使用默认字体，确保图表不包含中文

def train_dqn(episodes=100, max_steps=100):
    """Training DQN agent using original parameters"""
    
    env = SimulatedK8sEnv()
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    pod_types = ['video', 'net', 'disk', 'default', 'cpu_intensive', 'memory_intensive']
    
    dqn_agent = DQN(n_states, n_actions)
    
    total_rewards = []
    losses = []
    
    print("Starting DQN agent training...")
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0  # Initialize loss_count for each episode
        
        for step in range(max_steps):
            # Randomly select Pod type
            pod_type = random.choice(pod_types)
            env.set_pod_type(pod_type)
            
            # Choose action using original epsilon-greedy policy
            action = dqn_agent.choose_action(state)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # Reward smoothing (reduce reward fluctuation)
            if reward < -1000:  # Handle extreme rewards
                reward = -1000
            elif reward > 1000:
                reward = 1000
            
            # Store experience
            dqn_agent.store_transition(state, action, reward, next_state)
            
            # Learn
            loss = dqn_agent.learn()
            if loss is not None:
                episode_loss += loss
                loss_count += 1  # Count valid loss updates
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Record results
        total_rewards.append(episode_reward)
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        losses.append(avg_loss)
        
        # Print training progress
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_reward:.2f}, Loss: {avg_loss:.4f}")
        
        # Save model every 100 episodes
        if (episode + 1) % 100 == 0:
            dqn_agent.save_model(f'dqn_model_ep{episode+1}.pth')
    
    # Save final model
    dqn_agent.save_model('dqn_model.pth')
    print("Training completed, model saved.")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards)
    # Add moving average to show trend
    if len(total_rewards) > 10:
        moving_avg = np.convolve(total_rewards, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(total_rewards)), moving_avg, 'r-', label='10-episode moving average')
    plt.title('DQN Training - Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('dqn_training.png')
    print("Training curve saved as 'dqn_training.png'")
    
    print(f'Final reward: {total_rewards[-1]:.2f}')
    print(f'Average reward: {np.mean(total_rewards[-10:]):.2f}')
    print(f'Reward standard deviation: {np.std(total_rewards[-10:]):.2f}')
    
    return dqn_agent, total_rewards, losses

def random_scheduling(env, episodes=10, max_steps=100):
    """Random scheduling strategy"""
    results = []
    resource_stats = []
    utilization_stats = []  # Detailed resource utilization tracking
    pod_types = ['video', 'net', 'disk', 'default', 'cpu_intensive', 'memory_intensive']
    
    for _ in range(episodes):
        env.reset()
        total_reward = 0
        episode_stats = {}
        episode_utilization = {}
        
        for step in range(max_steps):
            # Randomly select Pod type
            pod_type = random.choice(pod_types)
            env.set_pod_type(pod_type)
            
            # Randomly select node
            action = random.randint(0, env.action_space.n - 1)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            
            # Record resource usage and detailed utilization
            if step % 10 == 0:
                cpu_values = [node['cpu'] for node in env.nodes]
                mem_values = [node['mem'] for node in env.nodes]
                net_values = [node['recv'] + node['tran'] for node in env.nodes]
                io_values = [node['read'] + node['write'] for node in env.nodes]
                
                # 记录标准差
                episode_stats[step] = {
                    'cpu_std': np.std(cpu_values),
                    'mem_std': np.std(mem_values),
                    'net_std': np.std(net_values),
                    'io_std': np.std(io_values)
                }
                
                # 记录每个节点的详细利用率
                episode_utilization[step] = {
                    'cpu': cpu_values,
                    'mem': mem_values,
                    'recv': [node['recv'] for node in env.nodes],
                    'tran': [node['tran'] for node in env.nodes],
                    'read': [node['read'] for node in env.nodes],
                    'write': [node['write'] for node in env.nodes]
                }
            
            if done:
                break
        
        results.append(total_reward)
        resource_stats.append(episode_stats)
        utilization_stats.append(episode_utilization)
    
    return np.mean(results), resource_stats, utilization_stats

def round_robin_scheduling(env, episodes=10, max_steps=100):
    """Round-robin scheduling strategy"""
    results = []
    resource_stats = []
    utilization_stats = []  # Detailed resource utilization tracking
    pod_types = ['video', 'net', 'disk', 'default', 'cpu_intensive', 'memory_intensive']
    
    for _ in range(episodes):
        env.reset()
        total_reward = 0
        current_node = 0
        episode_stats = {}
        episode_utilization = {}
        
        for step in range(max_steps):
            # 随机选择Pod类型
            pod_type = random.choice(pod_types)
            env.set_pod_type(pod_type)
            
            # 轮询选择节点
            action = current_node % env.action_space.n
            current_node += 1
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            
            # 记录资源使用情况和详细利用率
            if step % 10 == 0:
                cpu_values = [node['cpu'] for node in env.nodes]
                mem_values = [node['mem'] for node in env.nodes]
                net_values = [node['recv'] + node['tran'] for node in env.nodes]
                io_values = [node['read'] + node['write'] for node in env.nodes]
                
                # 记录标准差
                episode_stats[step] = {
                    'cpu_std': np.std(cpu_values),
                    'mem_std': np.std(mem_values),
                    'net_std': np.std(net_values),
                    'io_std': np.std(io_values)
                }
                
                # 记录每个节点的详细利用率
                episode_utilization[step] = {
                    'cpu': cpu_values,
                    'mem': mem_values,
                    'recv': [node['recv'] for node in env.nodes],
                    'tran': [node['tran'] for node in env.nodes],
                    'read': [node['read'] for node in env.nodes],
                    'write': [node['write'] for node in env.nodes]
                }
            
            if done:
                break
        
        results.append(total_reward)
        resource_stats.append(episode_stats)
        utilization_stats.append(episode_utilization)
    
    return np.mean(results), resource_stats, utilization_stats

def least_utilization_scheduling(env, episodes=10, max_steps=100):
    """Least utilization scheduling strategy"""
    results = []
    resource_stats = []
    utilization_stats = []  # Detailed resource utilization tracking
    pod_types = ['video', 'net', 'disk', 'default', 'cpu_intensive', 'memory_intensive']
    
    for _ in range(episodes):
        env.reset()
        total_reward = 0
        episode_stats = {}
        episode_utilization = {}
        
        for step in range(max_steps):
            # 随机选择Pod类型
            pod_type = random.choice(pod_types)
            env.set_pod_type(pod_type)
            
            # 计算每个节点的平均资源利用率
            node_utilizations = []
            for node in env.nodes:
                # 计算平均资源利用率
                avg_util = (node['cpu'] + node['mem'] + node['recv'] + node['tran'] + node['read'] + node['write']) / 6
                node_utilizations.append(avg_util)
            
            # 选择利用率最低的节点
            action = np.argmin(node_utilizations)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            
            # 记录资源使用情况和详细利用率
            if step % 10 == 0:
                cpu_values = [node['cpu'] for node in env.nodes]
                mem_values = [node['mem'] for node in env.nodes]
                net_values = [node['recv'] + node['tran'] for node in env.nodes]
                io_values = [node['read'] + node['write'] for node in env.nodes]
                
                # 记录标准差
                episode_stats[step] = {
                    'cpu_std': np.std(cpu_values),
                    'mem_std': np.std(mem_values),
                    'net_std': np.std(net_values),
                    'io_std': np.std(io_values)
                }
                
                # 记录每个节点的详细利用率
                episode_utilization[step] = {
                    'cpu': cpu_values,
                    'mem': mem_values,
                    'recv': [node['recv'] for node in env.nodes],
                    'tran': [node['tran'] for node in env.nodes],
                    'read': [node['read'] for node in env.nodes],
                    'write': [node['write'] for node in env.nodes]
                }
            
            if done:
                break
        
        results.append(total_reward)
        resource_stats.append(episode_stats)
        utilization_stats.append(episode_utilization)
    
    return np.mean(results), resource_stats, utilization_stats

def dqn_scheduling(env, dqn_agent, episodes=10, max_steps=100):
    """DQN scheduling strategy"""
    results = []
    resource_stats = []
    utilization_stats = []
    pod_types = ['video', 'net', 'disk', 'default', 'cpu_intensive', 'memory_intensive']
    
    # Save original epsilon for testing
    old_epsilon = dqn_agent.EPSILON
    dqn_agent.EPSILON = 0.1  # Reduce exploration during testing
    
    for _ in range(episodes):
        env.reset()
        total_reward = 0
        episode_stats = {}
        episode_utilization = {}
        
        for step in range(max_steps):
            # Randomly select Pod type
            pod_type = random.choice(pod_types)
            env.set_pod_type(pod_type)
            
            # Get current state from environment
            state = env.state
            
            # Use original DQN epsilon-greedy policy
            action = dqn_agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            
            # 记录资源使用情况和详细利用率
            if step % 10 == 0:
                cpu_values = [node['cpu'] for node in env.nodes]
                mem_values = [node['mem'] for node in env.nodes]
                net_values = [node['recv'] + node['tran'] for node in env.nodes]
                io_values = [node['read'] + node['write'] for node in env.nodes]
                
                # 记录标准差
                episode_stats[step] = {
                    'cpu_std': np.std(cpu_values),
                    'mem_std': np.std(mem_values),
                    'net_std': np.std(net_values),
                    'io_std': np.std(io_values)
                }
                
                # 记录每个节点的详细利用率
                episode_utilization[step] = {
                    'cpu': cpu_values,
                    'mem': mem_values,
                    'recv': [node['recv'] for node in env.nodes],
                    'tran': [node['tran'] for node in env.nodes],
                    'read': [node['read'] for node in env.nodes],
                    'write': [node['write'] for node in env.nodes]
                }
            
            if done:
                break
        
        results.append(total_reward)
        resource_stats.append(episode_stats)
        utilization_stats.append(episode_utilization)
    
    # 恢复原始探索率
    dqn_agent.EPSILON = old_epsilon
    
    print(f'DQN average reward: {np.mean(results):.2f}, std dev: {np.std(results):.2f}')
    
    return np.mean(results), resource_stats, utilization_stats

def run_comparison(dqn_agent, test_episodes=10, max_steps=100):
    """Compare scheduling strategies performance"""
    env = SimulatedK8sEnv()
    
    print("\nComparing scheduling strategies...")
    
    # Run scheduling strategies
    start_time = time.time()
    random_reward, random_stats, random_util = random_scheduling(env, test_episodes, max_steps)
    random_time = time.time() - start_time
    
    start_time = time.time()
    dqn_reward, dqn_stats, dqn_util = dqn_scheduling(env, dqn_agent, test_episodes, max_steps)
    dqn_time = time.time() - start_time
    
    # Print comparison results
    print(f"\nScheduling Strategy Performance Comparison:")
    print(f"Random Scheduling - Average Reward: {random_reward:.2f}, Execution Time: {random_time:.2f}s")
    print(f"DQN Scheduling - Average Reward: {dqn_reward:.2f}, Execution Time: {dqn_time:.2f}s")
    
    # Collect resource balance and utilization data
    strategies = ['Random Scheduling', 'DQN Scheduling']
    rewards = [random_reward, dqn_reward]
    stats = [random_stats, dqn_stats]
    utilizations = [random_util, dqn_util]
    
    return strategies, rewards, stats, utilizations

def plot_utilization_trends(strategies, utilizations):
    """Plot utilization trends for different scheduling strategies"""
    # Get steps
    steps = list(utilizations[0][0].keys())
    num_nodes = len(utilizations[0][0][steps[0]]['cpu'])  # Get number of nodes
    
    # Resource types
    resource_types = {
        'cpu': 'CPU',
        'mem': 'Memory',
        'recv': 'Network Receive',
        'tran': 'Network Transmit',
        'read': 'Disk Read',
        'write': 'Disk Write'
    }
    
    # Create a comprehensive plot showing average resource utilization
    plt.figure(figsize=(20, 10))
    
    # Create subplot for each resource type
    for i, (resource, resource_name) in enumerate(resource_types.items()):
        plt.subplot(2, 3, i + 1)
        
        # Plot average utilization for each strategy
        for j, strategy in enumerate(strategies):
            avg_util = []
            for step in steps:
                # Calculate average across all nodes at this step
                all_node_values = []
                for ep in utilizations[j]:
                    all_node_values.extend(ep[step][resource])
                avg_util.append(np.mean(all_node_values))
            
            plt.plot(steps, avg_util, marker='o', label=strategy)
        
        plt.title(f'{resource_name} Utilization Comparison')
        plt.xlabel('Step')
        plt.ylabel(f'Average {resource_name} Utilization (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('utilization_comparison.png', dpi=300)
    print("Resource utilization comparison saved as 'utilization_comparison.png'")

def plot_comparison(strategies, rewards, stats, utilizations):
    """Plot performance comparison"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Average reward comparison
    plt.subplot(2, 3, 1)
    plt.bar(strategies, rewards)
    plt.title('Average Reward Comparison')
    plt.xlabel('Scheduling Strategy')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=15)
    
    # Plot 2: CPU balance comparison
    plt.subplot(2, 3, 2)
    steps = list(stats[0][0].keys())  # Get steps
    
    for i, strategy_stats in enumerate(stats):
        avg_cpu_std = []
        for step in steps:
            step_values = [ep[step]['cpu_std'] for ep in strategy_stats]
            avg_cpu_std.append(np.mean(step_values))
        plt.plot(steps, avg_cpu_std, marker='o', label=strategies[i])
    
    plt.title('CPU Balance Comparison')
    plt.xlabel('Step')
    plt.ylabel('CPU Utilization Std Dev')
    plt.legend()
    
    # Plot 3: Memory balance comparison
    plt.subplot(2, 3, 3)
    for i, strategy_stats in enumerate(stats):
        avg_mem_std = []
        for step in steps:
            step_values = [ep[step]['mem_std'] for ep in strategy_stats]
            avg_mem_std.append(np.mean(step_values))
        plt.plot(steps, avg_mem_std, marker='s', label=strategies[i])
    
    plt.title('Memory Balance Comparison')
    plt.xlabel('Step')
    plt.ylabel('Memory Utilization Std Dev')
    plt.legend()
    
    # Plot 4: Network balance comparison
    plt.subplot(2, 3, 4)
    for i, strategy_stats in enumerate(stats):
        avg_net_std = []
        for step in steps:
            step_values = [ep[step]['net_std'] for ep in strategy_stats]
            avg_net_std.append(np.mean(step_values))
        plt.plot(steps, avg_net_std, marker='^', label=strategies[i])
    
    plt.title('Network Balance Comparison')
    plt.xlabel('Step')
    plt.ylabel('Network Utilization Std Dev')
    plt.legend()
    
    # Plot 5: Disk IO balance comparison
    plt.subplot(2, 3, 5)
    for i, strategy_stats in enumerate(stats):
        avg_io_std = []
        for step in steps:
            step_values = [ep[step]['io_std'] for ep in strategy_stats]
            avg_io_std.append(np.mean(step_values))
        plt.plot(steps, avg_io_std, marker='D', label=strategies[i])
    
    plt.title('Disk IO Balance Comparison')
    plt.xlabel('Step')
    plt.ylabel('IO Utilization Std Dev')
    plt.legend()
    
    # Plot 6: Overall performance radar chart
    plt.subplot(2, 3, 6, polar=True)
    
    # Calculate average balance across resource dimensions
    categories = ['CPU Balance', 'Memory Balance', 'Network Balance', 'IO Balance']
    N = len(categories)
    
    # Calculate angles
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Normalize data (lower std dev = better balance)
    max_values = {}
    for cat_idx, cat in enumerate(['cpu_std', 'mem_std', 'net_std', 'io_std']):
        all_values = []
        for strategy_stats in stats:
            values = []
            for ep in strategy_stats:
                values.extend([ep[step][cat] for step in ep])
            all_values.extend(values)
        max_values[cat] = max(all_values)
    
    for i, strategy_stats in enumerate(stats):
        values = []
        for cat in ['cpu_std', 'mem_std', 'net_std', 'io_std']:
            cat_values = []
            for ep in strategy_stats:
                cat_values.extend([ep[step][cat] for step in ep])
            # Normalize: lower std dev = higher performance score
            normalized = 1 - (np.mean(cat_values) / max_values[cat])
            values.append(normalized)
        
        # Close radar chart curve
        values.append(values[0])
        closed_angles = angles + [angles[0]]
        plt.plot(closed_angles, values, linewidth=2, label=strategies[i])
        plt.fill(closed_angles, values, alpha=0.1)
    
    # Set tick labels
    plt.xticks(angles, categories)
    plt.title('Overall Resource Balance Performance')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('scheduling_comparison.png', dpi=300)
    print("\nPerformance comparison saved as 'scheduling_comparison.png'")
    
    # 绘制资源利用率趋势图
    plot_utilization_trends(strategies, utilizations)
    
    # 显示图形
    plt.show()

def main():
    # Train DQN agent using original parameters
    print('Starting DQN agent training...')
    dqn_agent, rewards, losses = train_dqn(episodes=100, max_steps=100)
    
    # Compare scheduling strategies
    print('Training completed, comparing scheduling strategies...')
    strategies, rewards, stats, utilizations = run_comparison(dqn_agent, test_episodes=10, max_steps=100)
    
    # Plot comparison charts
    plot_comparison(strategies, rewards, stats, utilizations)

if __name__ == "__main__":
    main()