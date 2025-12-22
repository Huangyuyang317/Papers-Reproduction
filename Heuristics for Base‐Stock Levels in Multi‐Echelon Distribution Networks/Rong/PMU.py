import numpy as np
from scipy.stats import poisson, binom
import time
# 引入 evaluate_cost 中用到的相同数学函数，确保计算一致性
from calculate_cost import convolve_distributions, get_shortage_distribution, get_positive_negative_expectations, binomial_split

def pmu(node_lis):
    start_time = time.time()
    root = node_lis[0]
    
    # 初始缺货输入
    initial_shortage = {0: 1.0}
    
    # 开始递归优化
    _, best_S_map = optimize_subtree_unified(root, initial_shortage)
    
    # S -> s 转换
    local_stocks = [0] * len(node_lis)
    for node in node_lis:
        S_i = best_S_map[node.node]
        if node.is_leaf:
            local_stocks[node.node] = S_i
        else:
            sum_child_S = sum(best_S_map[child.node] for child in node.successors)
            local_stocks[node.node] = max(0, S_i - sum_child_S)
            
    return local_stocks, time.time() - start_time

def optimize_subtree_unified(node, incoming_shortage_dist):
    # 1. 确定搜索范围
    mu = node.demand_rate * node.l
    # PMU 的搜索范围必须足够大，否则会被 RO 反超
    high_bound = int(mu + 5 * np.sqrt(max(1, mu))) + 50
    if not node.is_leaf: high_bound += 50
    
    low = 0
    high = high_bound
    best_S = 0
    min_cost = float('inf')
    best_child_map = {}

    # 目标函数：计算子树总成本
    def evaluate_S(S_val):
        # A. 本地成本 (逻辑同 calculate_cost.py)
        # 本地需求
        limit = int(poisson.ppf(0.99999, mu))
        probs = poisson.pmf(np.arange(limit + 1), mu)
        demand_dist = {k: probs[k] for k in range(limit + 1) if probs[k] > 1e-9}
        
        # 卷积
        total_load_dist = convolve_distributions(demand_dist, incoming_shortage_dist)
        
        # 净库存
        net_dist = {}
        for load, prob in total_load_dist.items():
            net = S_val - load
            net_dist[net] = net_dist.get(net, 0.0) + prob
        
        exp_pos, exp_neg = get_positive_negative_expectations(net_dist)
        cost = node.H * exp_pos
        if node.is_leaf:
            cost += (node.h + node.b) * exp_neg
            
        # B. 子节点成本 (递归优化)
        child_maps = {}
        if not node.is_leaf:
            current_shortage = get_shortage_distribution(net_dist)
            
            # 分配逻辑 (同 calculate_cost，支持2个子节点)
            if len(node.successors) > 0:
                child_inputs = {child: {} for child in node.successors}
                
                # 处理有缺货情况
                for v, v_prob in current_shortage.items():
                    if len(node.successors) == 2:
                        c1, c2 = node.successors
                        k_list = np.arange(v + 1)
                        bin_probs = binom.pmf(k_list, v, c1.theta)
                        for k, p in zip(k_list, bin_probs):
                            jp = v_prob * p
                            if jp > 1e-10:
                                child_inputs[c1][k] = child_inputs[c1].get(k, 0.0) + jp
                                child_inputs[c2][v-k] = child_inputs[c2].get(v-k, 0.0) + jp
                    else: # 单子节点
                        c1 = node.successors[0]
                        child_inputs[c1][v] = child_inputs[c1].get(v, 0.0) + v_prob

                # 处理无缺货情况 (0)
                prob_zero = 1.0 - sum(current_shortage.values())
                if prob_zero > 1e-9:
                    for child in node.successors:
                        child_inputs[child][0] = child_inputs[child].get(0, 0.0) + prob_zero
                
                # 递归调用优化
                for child in node.successors:
                    c_min, c_map = optimize_subtree_unified(child, child_inputs[child])
                    cost += c_min
                    child_maps.update(c_map)
                    
        return cost, child_maps

    # 二分搜索
    while low <= high:
        mid = (low + high) // 2
        cost_mid, map_mid = evaluate_S(mid)
        cost_next, _ = evaluate_S(mid + 1)
        
        if cost_mid < cost_next:
            if cost_mid < min_cost:
                min_cost = cost_mid
                best_S = mid
                best_child_map = map_mid
            high = mid - 1
        else:
            low = mid + 1
            
    # 合并结果
    final_map = best_child_map.copy()
    final_map[node.node] = best_S
    return min_cost, final_map