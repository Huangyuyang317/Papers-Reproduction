import numpy as np
from scipy.stats import poisson, binom
import time

def trim_distribution(dist, threshold=1e-9):
    """截断概率极小的尾部，保持计算效率"""
    return {k: v for k, v in dist.items() if v > threshold}

def convolve_distributions(dist_a, dist_b):
    """计算两个离散分布 A + B 的卷积"""
    new_dist = {}
    for val_a, prob_a in dist_a.items():
        for val_b, prob_b in dist_b.items():
            new_val = val_a + val_b
            new_dist[new_val] = new_dist.get(new_val, 0.0) + prob_a * prob_b
    return trim_distribution(new_dist)

def binomial_projection_of_distribution(dist_v, theta):
    """
    已知随机变量 V 的分布 dist_v 计算子变量 Y ~ Bin(V, theta) 的分布。
    """
    if theta <= 1e-9:
        return {0: 1.0}
    if theta >= 1.0 - 1e-9:
        return dist_v
        
    y_dist = {}
    for v_val, v_prob in dist_v.items():
        if v_val == 0:
            y_dist[0] = y_dist.get(0, 0.0) + v_prob
            continue
            
        # 计算 Bin(v_val, theta) 的分布
        k_list = np.arange(v_val + 1)
        probs = binom.pmf(k_list, v_val, theta)
        
        for k, p in zip(k_list, probs):
            if p > 1e-9:
                y_dist[k] = y_dist.get(k, 0.0) + v_prob * p
                
    return trim_distribution(y_dist)

# ==========================================
# RO 主函数
# ==========================================

def ro(node_lis):
    """
    RO Heuristic (Recursive Optimization) with Leaf Refinement.
    
    Args:
        node_lis: List of Stage objects (nodes).
        
    Returns:
        local_stocks: List of local base-stock levels (s_i).
        elapsed_time: Execution time.
    """
    start_time = time.time()
    
    # 建立索引映射
    nodes_by_id = {node.node: node for node in node_lis}
    
    # 缓存
    cost_memo = {} 
    S_r0 = {} # 中间梯级库存
    
    # ---------------------------------------------------------
    # Step 1: Compute Intermediate Echelon Base-Stock Levels (S_r0)
    # 自底向上 (Bottom-Up)
    # ---------------------------------------------------------
    
    # 按索引倒序遍历（确保先处理子节点）
    for i in range(len(node_lis) - 1, -1, -1):
        current_node = node_lis[i]
        cost_memo[current_node.node] = {} # 初始化缓存
        
        # 寻找最优 y (S_r0)
        # 凸函数搜索：从 0 开始增加，直到成本上升
        best_y = 0
        min_val = float('inf')
        
        y = 0
        while True:
            # 计算 C_i(y)
            val = calculate_Ci(y, current_node, S_r0, cost_memo, node_lis)
            
            # 存入 memo
            cost_memo[current_node.node][y] = val
            
            if val < min_val:
                min_val = val
                best_y = y
                y += 1
            else:
                # 成本开始上升，停止
                break
        
        S_r0[current_node.node] = best_y

    # ---------------------------------------------------------
    # Step 2: Calculate Initial Local Base-Stock Levels (s_i)
    # ---------------------------------------------------------
    local_stocks = [0] * len(node_lis)
    
    for node in node_lis:
        if node.is_leaf:
            # 暂时赋值，Step 3 会更新它
            local_stocks[node.node] = S_r0[node.node]
        else:
            # 非叶节点: s_i = S_i^r0 - sum(S_child^r0)
            sum_child_S = sum(S_r0[child.node] for child in node.successors)
            local_stocks[node.node] = max(0, S_r0[node.node] - sum_child_S)
    
    # ---------------------------------------------------------
    # Step 3: Refinement for Leaves (Eq 10)
    # 自顶向下 (Top-Down) 传播缺货并再优化叶节点
    # ---------------------------------------------------------
    
    # 初始化：根节点接收到的上游缺货分布为 {0: 1.0} (无缺货)
    incoming_shortage_dist = {node.node: {0: 1.0} for node in node_lis}
    
    # A. 传播过程
    for i in range(len(node_lis)): # 正序遍历 (Top-Down)
        node = node_lis[i]
        
        # 叶节点不产生给下游的缺货，跳过
        if node.is_leaf:
            continue
            
        # 1. 计算当前节点的总负载分布 (Incoming Shortage + Local Demand)
        # 泊松需求分布
        mu = node.demand_rate * node.l
        limit = int(poisson.ppf(0.99999, mu))
        probs = poisson.pmf(np.arange(limit + 1), mu)
        demand_dist = {k: probs[k] for k in range(limit + 1) if probs[k] > 1e-9}
        
        # 卷积：总负载
        total_load_dist = convolve_distributions(incoming_shortage_dist[node.node], demand_dist)
        
        # 2. 计算当前节点产生的缺货 V
        # 逻辑：V = [ Total_Load - Local_Available_Inventory ]^+
        # 其中 Local_Available_Inventory 即 Step 2 算出的 s_i
        # (因为 s_i = S_r0 - sum_child_S，正是留给本地消化波动的量)
        local_avail = local_stocks[node.node]
        
        generated_v_dist = {}
        for load, prob in total_load_dist.items():
            v = max(0, load - local_avail)
            generated_v_dist[v] = generated_v_dist.get(v, 0.0) + prob
        
        # 3. 将缺货 V 分配给子节点
        for child in node.successors:
            child_shortage = binomial_projection_of_distribution(generated_v_dist, child.theta)
            incoming_shortage_dist[child.node] = child_shortage

    # B. 再优化过程 (仅针对叶节点)
    for node in node_lis:
        if not node.is_leaf:
            continue
            
        # 1. 计算有效需求分布 (Local Demand + Incoming Shortage)
        mu = node.demand_rate * node.l
        limit = int(poisson.ppf(0.99999, mu))
        probs = poisson.pmf(np.arange(limit + 1), mu)
        my_demand_dist = {k: probs[k] for k in range(limit + 1) if probs[k] > 1e-9}
        
        effective_demand_dist = convolve_distributions(my_demand_dist, incoming_shortage_dist[node.node])
        
        # 2. Newsvendor 优化
        # 寻找最小的 s，使得 P(Effective_Demand <= s) >= b / (h + b)
        target_ratio = node.b / (node.h + node.b)
        
        sorted_demand = sorted(effective_demand_dist.items()) # 按需求值排序
        cumulative_prob = 0.0
        optimal_s = sorted_demand[-1][0] # 默认最大值
        
        for d_val, prob in sorted_demand:
            cumulative_prob += prob
            if cumulative_prob >= target_ratio:
                optimal_s = d_val
                break
        
        # 更新叶节点的本地库存
        local_stocks[node.node] = int(optimal_s)

    elapsed_time = time.time() - start_time
    return local_stocks, elapsed_time


# ==========================================
# 核心递归计算函数
# ==========================================

def calculate_Ci(y, node, S_r0, cost_memo, node_lis):
    """
    计算节点 i 在梯级库存 y 下的期望成本。
    """
    # 检查缓存
    if y in cost_memo[node.node]:
        return cost_memo[node.node][y]
    
    # 1. 提前期需求分布
    mu = node.demand_rate * node.l
    limit = int(poisson.ppf(0.99999, mu))
    probs = poisson.pmf(np.arange(limit + 1), mu)
    
    exp_cost = 0.0
    
    for k in range(limit + 1):
        if probs[k] < 1e-9: continue
        
        # x 是净梯级库存
        x = y - k
        hat_c = 0.0
        
        # Term 1: Echelon Holding Cost
        hat_c += node.H * x
        
        # Term 2: Penalty / Recursive Cost
        if node.is_leaf:
            # Leaf: Newsvendor cost
            shortage = max(0, -x)
            hat_c += (node.h + node.b) * shortage
        else:
            # Internal: Allocation of shortage
            sum_child_S = sum(S_r0[child.node] for child in node.successors)
            V = max(0, -(x - sum_child_S))
            
            if V == 0:
                for child in node.successors:
                    # 没有缺货，子节点处于其最优状态 S_r0
                    hat_c += calculate_Ci(S_r0[child.node], child, S_r0, cost_memo, node_lis)
            else:
                for child in node.successors:
                    theta = child.theta
                    # 优化：如果 theta 极小或极大，跳过二项分布计算
                    if theta <= 1e-9:
                         hat_c += calculate_Ci(S_r0[child.node], child, S_r0, cost_memo, node_lis)
                    elif theta >= 1.0 - 1e-9:
                         hat_c += calculate_Ci(S_r0[child.node] - V, child, S_r0, cost_memo, node_lis)
                    else:
                        # 二项分布卷积
                        bin_limit = V
                        bin_probs = binom.pmf(np.arange(bin_limit + 1), V, theta)
                        
                        child_exp_val = 0.0
                        for m in range(bin_limit + 1):
                            if bin_probs[m] < 1e-9: continue
                            
                            child_val = calculate_Ci(S_r0[child.node] - m, child, S_r0, cost_memo, node_lis)
                            child_exp_val += child_val * bin_probs[m]
                        
                        hat_c += child_exp_val
                        
        exp_cost += hat_c * probs[k]
        
    # 写入缓存
    cost_memo[node.node][y] = exp_cost
    return exp_cost