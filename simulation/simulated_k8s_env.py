import numpy as np
import gym
from gym import spaces, core
import random
import uuid
from math import sin, cos

class SimulatedK8sEnv(core.Env):
    def __init__(self):
        self.count = 0
        self.maxCount = 100
        
        # 环境参数
        self.action_space = spaces.Discrete(4)  # 4个节点
        low = np.array([0]*30, dtype=np.float32)
        high = np.array([100]*30, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # 初始化节点状态（模拟4个节点），为每个节点添加个性特征
        self.nodes = []
        # 节点个性特征：不同节点有不同的基准负载和波动特性
        node_profiles = [
            # 节点1: 计算密集型服务器
            {'base_load': 25, 'volatility': 0.15, 'resource_bias': 'cpu'},
            # 节点2: 内存密集型服务器
            {'base_load': 30, 'volatility': 0.12, 'resource_bias': 'mem'},
            # 节点3: 网络密集型服务器
            {'base_load': 20, 'volatility': 0.20, 'resource_bias': 'net'},
            # 节点4: 存储密集型服务器
            {'base_load': 15, 'volatility': 0.18, 'resource_bias': 'io'}
        ]
        
        for i, profile in enumerate(node_profiles):
            # 根据节点个性特征初始化资源使用
            base_bias = {}
            if profile['resource_bias'] == 'cpu':
                base_bias = {'cpu': 40, 'mem': 20, 'recv': 10, 'tran': 10, 'read': 15, 'write': 15}
            elif profile['resource_bias'] == 'mem':
                base_bias = {'cpu': 20, 'mem': 40, 'recv': 10, 'tran': 10, 'read': 15, 'write': 15}
            elif profile['resource_bias'] == 'net':
                base_bias = {'cpu': 20, 'mem': 25, 'recv': 30, 'tran': 25, 'read': 15, 'write': 15}
            elif profile['resource_bias'] == 'io':
                base_bias = {'cpu': 20, 'mem': 20, 'recv': 10, 'tran': 10, 'read': 35, 'write': 30}
            
            # 初始化节点，添加个性特征
            node = {
                'cpu': base_bias['cpu'] * (1 + random.uniform(-0.3, 0.3)),
                'mem': base_bias['mem'] * (1 + random.uniform(-0.3, 0.3)),
                'recv': base_bias['recv'] * (1 + random.uniform(-0.3, 0.3)),
                'tran': base_bias['tran'] * (1 + random.uniform(-0.3, 0.3)),
                'read': base_bias['read'] * (1 + random.uniform(-0.3, 0.3)),
                'write': base_bias['write'] * (1 + random.uniform(-0.3, 0.3)),
                'pods': [],  # 在该节点运行的pods列表
                'profile': profile,
                # 添加资源趋势因子，用于模拟长时间的资源使用趋势
                'trend_cpu': random.uniform(-0.02, 0.02),
                'trend_mem': random.uniform(-0.02, 0.02),
                'trend_recv': random.uniform(-0.02, 0.02),
                'trend_tran': random.uniform(-0.02, 0.02),
                'trend_read': random.uniform(-0.02, 0.02),
                'trend_write': random.uniform(-0.02, 0.02),
                # 记录上次突发时间
                'last_burst': {'cpu': -100, 'mem': -100, 'net': -100, 'io': -100}
            }
            self.nodes.append(node)
        
        # 预设pod类型及其资源需求范围（min和max）
        self.pod_types = {
            'video': {
                'cpu': {'min': 20, 'max': 50},  # CPU消耗范围
                'mem': {'min': 15, 'max': 40},  # 内存消耗范围
                'recv': {'min': 10, 'max': 30}, # 网络接收范围
                'tran': {'min': 5, 'max': 15},  # 网络发送范围
                'read': {'min': 2, 'max': 10},  # 磁盘读取范围
                'write': {'min': 1, 'max': 8}   # 磁盘写入范围
            },
            'net': {
                'cpu': {'min': 5, 'max': 20},   
                'mem': {'min': 10, 'max': 30}, 
                'recv': {'min': 30, 'max': 80}, 
                'tran': {'min': 20, 'max': 60}, 
                'read': {'min': 1, 'max': 10}, 
                'write': {'min': 1, 'max': 8}   
            },
            'disk': {
                'cpu': {'min': 15, 'max': 40},  
                'mem': {'min': 10, 'max': 30}, 
                'recv': {'min': 5, 'max': 15},  
                'tran': {'min': 5, 'max': 15},  
                'read': {'min': 30, 'max': 80}, 
                'write': {'min': 30, 'max': 80} 
            },
            'default': {
                'cpu': {'min': 2, 'max': 10},   
                'mem': {'min': 2, 'max': 10},   
                'recv': {'min': 2, 'max': 10},  
                'tran': {'min': 2, 'max': 10},  
                'read': {'min': 2, 'max': 10},  
                'write': {'min': 2, 'max': 10}  
            },
            'cpu_intensive': {
                'cpu': {'min': 50, 'max': 90},  
                'mem': {'min': 5, 'max': 20},   
                'recv': {'min': 2, 'max': 10},  
                'tran': {'min': 2, 'max': 10},  
                'read': {'min': 2, 'max': 10},  
                'write': {'min': 2, 'max': 10}  
            },
            'memory_intensive': {
                'cpu': {'min': 5, 'max': 20},   
                'mem': {'min': 50, 'max': 90},  
                'recv': {'min': 2, 'max': 10},  
                'tran': {'min': 2, 'max': 10},  
                'read': {'min': 2, 'max': 10},  
                'write': {'min': 2, 'max': 10}  
            }
        }
        
        self.current_pod_type = None
        self.state = self._get_state()
        self.isDone = False
        
        # 用于模拟现实世界的资源使用模式
        self.resource_noise_params = {
            'cpu': {'mean': 0.0, 'std': 0.02, 'auto_corr': 0.8},
            'mem': {'mean': 0.0, 'std': 0.01, 'auto_corr': 0.9},
            'recv': {'mean': 0.0, 'std': 0.05, 'auto_corr': 0.7},
            'tran': {'mean': 0.0, 'std': 0.04, 'auto_corr': 0.75},
            'read': {'mean': 0.0, 'std': 0.03, 'auto_corr': 0.85},
            'write': {'mean': 0.0, 'std': 0.03, 'auto_corr': 0.82}
        }
        
        # 存储前一时刻的噪声值，用于生成自相关的波动
        self.previous_noise = {resource: [0.0] * 4 for resource in self.resource_noise_params.keys()}
        
    def _get_state(self):
        # 构建状态向量，包含4个节点的6个资源指标和当前pod的资源需求范围的平均值
        state = []
        for node in self.nodes:
            state.extend([node['cpu'], node['mem'], node['recv'], node['tran'], node['read'], node['write']])
        
        # 添加当前pod的资源需求范围的平均值
        if self.current_pod_type:
            pod_range = self.pod_types[self.current_pod_type]
            state.extend([
                (pod_range['cpu']['min'] + pod_range['cpu']['max']) / 2,
                (pod_range['mem']['min'] + pod_range['mem']['max']) / 2,
                (pod_range['recv']['min'] + pod_range['recv']['max']) / 2,
                (pod_range['tran']['min'] + pod_range['tran']['max']) / 2,
                (pod_range['read']['min'] + pod_range['read']['max']) / 2,
                (pod_range['write']['min'] + pod_range['write']['max']) / 2
            ])
        else:
            state.extend([0]*6)
            
        return np.array(state, dtype=np.float32)
    
    def reset(self):
        self.count = 0
        self.isDone = False
        
        # 重新初始化节点状态
        self.nodes = []
        # 节点个性特征：不同节点有不同的基准负载和波动特性
        node_profiles = [
            # 节点1: 计算密集型服务器
            {'base_load': 25, 'volatility': 0.15, 'resource_bias': 'cpu'},
            # 节点2: 内存密集型服务器
            {'base_load': 30, 'volatility': 0.12, 'resource_bias': 'mem'},
            # 节点3: 网络密集型服务器
            {'base_load': 20, 'volatility': 0.20, 'resource_bias': 'net'},
            # 节点4: 存储密集型服务器
            {'base_load': 15, 'volatility': 0.18, 'resource_bias': 'io'}
        ]
        
        for i, profile in enumerate(node_profiles):
            # 根据节点个性特征初始化资源使用
            base_bias = {}
            if profile['resource_bias'] == 'cpu':
                base_bias = {'cpu': 40, 'mem': 20, 'recv': 10, 'tran': 10, 'read': 15, 'write': 15}
            elif profile['resource_bias'] == 'mem':
                base_bias = {'cpu': 20, 'mem': 40, 'recv': 10, 'tran': 10, 'read': 15, 'write': 15}
            elif profile['resource_bias'] == 'net':
                base_bias = {'cpu': 20, 'mem': 25, 'recv': 30, 'tran': 25, 'read': 15, 'write': 15}
            elif profile['resource_bias'] == 'io':
                base_bias = {'cpu': 20, 'mem': 20, 'recv': 10, 'tran': 10, 'read': 35, 'write': 30}
            
            # 初始化节点，添加个性特征
            node = {
                'cpu': base_bias['cpu'] * (1 + random.uniform(-0.3, 0.3)),
                'mem': base_bias['mem'] * (1 + random.uniform(-0.3, 0.3)),
                'recv': base_bias['recv'] * (1 + random.uniform(-0.3, 0.3)),
                'tran': base_bias['tran'] * (1 + random.uniform(-0.3, 0.3)),
                'read': base_bias['read'] * (1 + random.uniform(-0.3, 0.3)),
                'write': base_bias['write'] * (1 + random.uniform(-0.3, 0.3)),
                'pods': [],  # 在该节点运行的pods列表
                'profile': profile,
                # 添加资源趋势因子，用于模拟长时间的资源使用趋势
                'trend_cpu': random.uniform(-0.02, 0.02),
                'trend_mem': random.uniform(-0.02, 0.02),
                'trend_recv': random.uniform(-0.02, 0.02),
                'trend_tran': random.uniform(-0.02, 0.02),
                'trend_read': random.uniform(-0.02, 0.02),
                'trend_write': random.uniform(-0.02, 0.02),
                # 记录上次突发时间
                'last_burst': {'cpu': -100, 'mem': -100, 'net': -100, 'io': -100}
            }
            self.nodes.append(node)
        
        # 增加pod类型的随机性
        self.current_pod_type = random.choice(['video', 'net', 'disk', 'default', 'cpu_intensive', 'memory_intensive'])
        self.state = self._get_state()
        return self.state
    
    def set_pod_type(self, pod_type):
        # 设置当前要调度的pod类型
        if pod_type in self.pod_types:
            self.current_pod_type = pod_type
            self.state = self._get_state()
    
    def _apply_time_pattern(self, base_value, resource_type, step):
        """应用时间相关的模式到资源使用，模拟一天中的不同时段"""
        # 基本周期模式（模拟24小时周期）
        daily_cycle = 0.1 * sin(2 * np.pi * step / 24) + 0.05 * cos(4 * np.pi * step / 24)
        
        # 资源类型特定的时间模式
        resource_patterns = {
            'cpu': 0.15 * sin(2 * np.pi * step / 36) + 0.08 * cos(2 * np.pi * step / 12),  # CPU使用模式
            'mem': 0.10 * sin(2 * np.pi * step / 48) + 0.05 * cos(2 * np.pi * step / 24),  # 内存使用模式
            'net': 0.20 * sin(2 * np.pi * step / 18) + 0.10 * cos(2 * np.pi * step / 9),   # 网络使用模式
            'io': 0.12 * sin(2 * np.pi * step / 30) + 0.06 * cos(2 * np.pi * step / 15)   # IO使用模式
        }
        
        pattern = daily_cycle + resource_patterns.get(resource_type, 0)
        return base_value * (1 + pattern)
    
    def _apply_burst(self, value, resource_type, node, step):
        """应用突发流量模式，模拟不可预测的流量峰值"""
        burst_types = {
            'cpu': {'probability': 0.05, 'factor': 1.5, 'duration': 3, 'cooldown': 15},
            'mem': {'probability': 0.03, 'factor': 1.4, 'duration': 5, 'cooldown': 20},
            'net': {'probability': 0.10, 'factor': 2.0, 'duration': 2, 'cooldown': 10},
            'io':  {'probability': 0.07, 'factor': 1.6, 'duration': 4, 'cooldown': 12}
        }
        
        burst_info = burst_types.get(resource_type, {})
        last_burst = node['last_burst'].get(resource_type, -100)
        
        # 检查是否可以触发新的突发
        if random.random() < burst_info.get('probability', 0) and step - last_burst > burst_info.get('cooldown', 10):
            node['last_burst'][resource_type] = step
            return value * burst_info.get('factor', 1.0)
        
        # 如果正在突发持续时间内，继续应用突发
        if step - last_burst < burst_info.get('duration', 0) and step - last_burst >= 0:
            return value * burst_info.get('factor', 1.0)
            
        return value
    
    def _apply_correlation(self, node):
        """应用资源之间的相关性，模拟真实系统中的资源使用关联"""
        # CPU和内存通常是相关的
        cpu_change = node.get('cpu_change', 0)
        node['mem'] += cpu_change * 0.3 * random.uniform(0.8, 1.2)
        
        # 网络和磁盘IO通常是相关的
        net_change = (node.get('recv_change', 0) + node.get('tran_change', 0)) / 2
        node['read'] += net_change * 0.2 * random.uniform(0.7, 1.3)
        node['write'] += net_change * 0.15 * random.uniform(0.7, 1.3)
        
        # 重置变化值
        for key in ['cpu_change', 'mem_change', 'recv_change', 'tran_change', 'read_change', 'write_change']:
            node[key] = 0
    
    def _update_pod_resources(self, pod, pod_range, step):
        """更新pod资源使用，添加更复杂的时间模式和波动"""
        # 基于pod类型的资源使用模式
        patterns = {
            'video': {'cpu': 0.15, 'mem': 0.10, 'recv': 0.25, 'tran': 0.20, 'read': 0.05, 'write': 0.05},
            'net': {'cpu': 0.10, 'mem': 0.15, 'recv': 0.30, 'tran': 0.25, 'read': 0.05, 'write': 0.05},
            'disk': {'cpu': 0.15, 'mem': 0.10, 'recv': 0.10, 'tran': 0.10, 'read': 0.30, 'write': 0.25},
            'default': {'cpu': 0.08, 'mem': 0.08, 'recv': 0.10, 'tran': 0.10, 'read': 0.10, 'write': 0.10},
            'cpu_intensive': {'cpu': 0.20, 'mem': 0.05, 'recv': 0.05, 'tran': 0.05, 'read': 0.05, 'write': 0.05},
            'memory_intensive': {'cpu': 0.05, 'mem': 0.20, 'recv': 0.05, 'tran': 0.05, 'read': 0.05, 'write': 0.05}
        }
        
        pod_pattern = patterns.get(pod['type'], patterns['default'])
        
        # 对每种资源应用特定模式
        for resource in ['cpu', 'mem', 'recv', 'tran', 'read', 'write']:
            # 基础波动
            base_fluctuation = pod_pattern[resource] * (random.random() - 0.5)
            
            # 时间相关模式
            time_pattern = 0.05 * sin(2 * np.pi * step / 10 + hash(pod['id']) % 10)
            
            # 缓慢趋势变化
            if f'trend_{resource}' not in pod:
                pod[f'trend_{resource}'] = random.uniform(-0.01, 0.01)
            
            # 趋势可能随时间改变
            if random.random() < 0.1:
                pod[f'trend_{resource}'] += random.uniform(-0.01, 0.01)
                pod[f'trend_{resource}'] = max(-0.03, min(0.03, pod[f'trend_{resource}']))
            
            # 应用变化
            new_value = pod[resource] * (1 + base_fluctuation + time_pattern + pod[f'trend_{resource}'])
            
            # 确保在范围内
            pod[resource] = max(pod_range[resource]['min'], min(pod_range[resource]['max'], new_value))
    
    def step(self, action):
        # 执行调度动作，将pod调度到指定节点
        self.count += 1
        
        # 重置每个节点的资源基础值（去除之前所有pod的消耗）
        for node in self.nodes:
            # 获取节点个性特征
            profile = node['profile']
            
            # 根据节点个性和时间模式设置基础资源使用
            node['cpu'] = profile['base_load'] * (2.0 if profile['resource_bias'] == 'cpu' else 1.0) * (1 + random.uniform(-0.1, 0.1))
            node['mem'] = profile['base_load'] * (2.0 if profile['resource_bias'] == 'mem' else 1.0) * (1 + random.uniform(-0.1, 0.1))
            node['recv'] = profile['base_load'] * (1.5 if profile['resource_bias'] == 'net' else 1.0) * (1 + random.uniform(-0.1, 0.1))
            node['tran'] = profile['base_load'] * (1.5 if profile['resource_bias'] == 'net' else 1.0) * (1 + random.uniform(-0.1, 0.1))
            node['read'] = profile['base_load'] * (2.0 if profile['resource_bias'] == 'io' else 1.0) * (1 + random.uniform(-0.1, 0.1))
            node['write'] = profile['base_load'] * (2.0 if profile['resource_bias'] == 'io' else 1.0) * (1 + random.uniform(-0.1, 0.1))
            
            # 应用时间模式到基础资源使用
            node['cpu'] = self._apply_time_pattern(node['cpu'], 'cpu', self.count)
            node['mem'] = self._apply_time_pattern(node['mem'], 'mem', self.count)
            node['recv'] = self._apply_time_pattern(node['recv'], 'net', self.count)
            node['tran'] = self._apply_time_pattern(node['tran'], 'net', self.count)
            node['read'] = self._apply_time_pattern(node['read'], 'io', self.count)
            node['write'] = self._apply_time_pattern(node['write'], 'io', self.count)
            
            # 应用节点特定的趋势
            for resource in ['cpu', 'mem', 'recv', 'tran', 'read', 'write']:
                node[resource] *= (1 + node[f'trend_{resource}'])
                # 趋势本身也会随机变化
                if random.random() < 0.2:
                    node[f'trend_{resource}'] += random.uniform(-0.01, 0.01)
                    node[f'trend_{resource}'] = max(-0.03, min(0.03, node[f'trend_{resource}']))
        
        # 将新的pod调度到指定节点
        if action in range(4) and self.current_pod_type:
            node = self.nodes[action]
            pod_range = self.pod_types.get(self.current_pod_type, self.pod_types['default'])
            
            # 创建新pod，设置初始资源使用在范围内随机值
            new_pod = {
                'id': str(uuid.uuid4())[:8],
                'type': self.current_pod_type,
                'cpu': random.uniform(pod_range['cpu']['min'], pod_range['cpu']['max']),
                'mem': random.uniform(pod_range['mem']['min'], pod_range['mem']['max']),
                'recv': random.uniform(pod_range['recv']['min'], pod_range['recv']['max']),
                'tran': random.uniform(pod_range['tran']['min'], pod_range['tran']['max']),
                'read': random.uniform(pod_range['read']['min'], pod_range['read']['max']),
                'write': random.uniform(pod_range['write']['min'], pod_range['write']['max']),
                'age': 0  # 记录pod的生命周期
            }
            node['pods'].append(new_pod)
        
        # 动态更新每个节点上所有pod的资源使用情况，并累加到节点资源使用中
        for node in self.nodes:
            # 记录原始值用于计算变化
            original_cpu = node['cpu']
            original_recv = node['recv']
            original_tran = node['tran']
            
            for pod in node['pods']:
                pod_range = self.pod_types[pod['type']]
                pod['age'] += 1  # 更新pod年龄
                
                # 更新pod资源使用
                self._update_pod_resources(pod, pod_range, self.count)
                
                # 将pod资源使用累加到节点
                node['cpu'] += pod['cpu']
                node['mem'] += pod['mem']
                node['recv'] += pod['recv']
                node['tran'] += pod['tran']
                node['read'] += pod['read']
                node['write'] += pod['write']
            
            # 计算资源变化用于相关性
            node['cpu_change'] = node['cpu'] - original_cpu
            node['recv_change'] = node['recv'] - original_recv
            node['tran_change'] = node['tran'] - original_tran
            
            # 应用资源之间的相关性
            self._apply_correlation(node)
            
            # 应用突发流量
            node['cpu'] = self._apply_burst(node['cpu'], 'cpu', node, self.count)
            node['mem'] = self._apply_burst(node['mem'], 'mem', node, self.count)
            node['recv'] = self._apply_burst(node['recv'], 'net', node, self.count)
            node['tran'] = self._apply_burst(node['tran'], 'net', node, self.count)
            node['read'] = self._apply_burst(node['read'], 'io', node, self.count)
            node['write'] = self._apply_burst(node['write'], 'io', node, self.count)
            
            # 确保节点资源在合理范围内
            for resource in ['cpu', 'mem', 'recv', 'tran', 'read', 'write']:
                node[resource] = min(95, node[resource])
        
        # 添加节点级别的随机波动（模拟背景负载变化）
        for node in self.nodes:
            volatility = node['profile']['volatility']
            for resource in ['cpu', 'mem', 'recv', 'tran', 'read', 'write']:
                node[resource] = max(5, min(95, node[resource] * (1 + random.uniform(-volatility/2, volatility/2))))
        
        # 更新状态
        self.state = self._get_state()
        
        # 计算奖励（基于资源均衡性和利用率）
        cpu_values = [node['cpu'] for node in self.nodes]
        mem_values = [node['mem'] for node in self.nodes]
        net_values = [node['recv'] + node['tran'] for node in self.nodes]
        io_values = [node['read'] + node['write'] for node in self.nodes]
        
        # 标准差惩罚
        std_cpu = np.std(cpu_values)
        std_mem = np.std(mem_values)
        std_net = np.std(net_values)
        std_io = np.std(io_values)
        
        # 平均利用率奖励（鼓励合理使用资源但不过载）
        avg_cpu = np.mean(cpu_values)
        avg_mem = np.mean(mem_values)
        
        # 非线性奖励函数：鼓励利用率在40-70%之间
        cpu_util_reward = -2 * (avg_cpu - 55)**2 / 100 + 55 if 30 <= avg_cpu <= 80 else -100
        mem_util_reward = -2 * (avg_mem - 55)**2 / 100 + 55 if 30 <= avg_mem <= 80 else -100
        
        # 综合奖励计算
        reward = -0.8 * (std_cpu + std_mem + std_net + std_io) + 0.3 * (cpu_util_reward + mem_util_reward)
        
        # 添加额外的惩罚
        for node in self.nodes:
            # 过载惩罚
            if node['cpu'] > 85 or node['mem'] > 85:
                reward -= 80
            if node['cpu'] > 90 or node['mem'] > 90:
                reward -= 150  # 更严重的惩罚
            
            # 资源不平衡惩罚（同一节点内不同资源使用差异大）
            cpu_mem_diff = abs(node['cpu'] - node['mem'])
            net_io_diff = abs((node['recv'] + node['tran']) - (node['read'] + node['write']))
            if cpu_mem_diff > 40:
                reward -= cpu_mem_diff / 2
            if net_io_diff > 50:
                reward -= net_io_diff / 2
        
        # 更真实的Pod生命周期管理
        for node in self.nodes:
            pods_to_remove = []
            for i, pod in enumerate(node['pods']):
                # Pod年龄相关的移除概率
                age_factor = min(pod['age'] / 100, 1.0)  # 年龄越大，移除概率越高
                base_removal_prob = 0.05
                
                # Pod类型相关的生命周期差异
                type_factors = {
                    'default': 1.0,
                    'cpu_intensive': 0.8,  # CPU密集型Pod生命周期较短
                    'memory_intensive': 1.2,  # 内存密集型Pod生命周期较长
                    'video': 0.9,
                    'net': 1.1,
                    'disk': 0.9
                }
                type_factor = type_factors.get(pod['type'], 1.0)
                
                # 计算最终移除概率
                removal_prob = base_removal_prob * type_factor * (1 + age_factor)
                
                # 资源使用过高的Pod更容易被移除（模拟资源压力导致的重启）
                resource_overload = any([pod[r] > 80 for r in ['cpu', 'mem']])
                if resource_overload:
                    removal_prob *= 1.5
                
                if random.random() < removal_prob:
                    pods_to_remove.append(i)
            
            # 从后往前移除，避免索引问题
            for i in reversed(pods_to_remove):
                node['pods'].pop(i)
        
        # 检查是否达到最大步数
        if self.count >= self.maxCount:
            self.isDone = True
        
        # 更智能的Pod类型选择，模拟真实的工作负载分布
        pod_weights = {
            'default': 0.35,  # 默认类型最常见
            'video': 0.15,
            'net': 0.15,
            'disk': 0.15,
            'cpu_intensive': 0.10,
            'memory_intensive': 0.10
        }
        
        # 时间相关的Pod类型分布
        time_factor = sin(2 * np.pi * self.count / 24)
        # 白天CPU密集型任务更多，晚上网络和视频任务更多
        if time_factor > 0:  # 模拟白天
            pod_weights['cpu_intensive'] *= 1.5
            pod_weights['video'] *= 0.8
            pod_weights['net'] *= 0.9
        else:  # 模拟晚上
            pod_weights['cpu_intensive'] *= 0.8
            pod_weights['video'] *= 1.5
            pod_weights['net'] *= 1.3
        
        # 归一化权重
        total_weight = sum(pod_weights.values())
        normalized_weights = [w/total_weight for w in pod_weights.values()]
        
        self.current_pod_type = random.choices(
            list(pod_weights.keys()), 
            weights=normalized_weights, 
            k=1
        )[0]
        
        return self.state, reward, self.isDone, {}
    
    def render(self, mode='human'):
        # 打印当前状态信息
        print(f"Step: {self.count}/{self.maxCount}")
        print(f"Current Pod Type: {self.current_pod_type}")
        for i, node in enumerate(self.nodes):
            print(f"Node {i+1}: CPU={node['cpu']:.2f}%, MEM={node['mem']:.2f}%, "
                  f"NET={node['recv']+node['tran']:.2f}%, IO={node['read']+node['write']:.2f}%")
    
    def close(self):
        pass