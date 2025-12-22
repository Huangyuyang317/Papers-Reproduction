import numpy as np
from scipy.stats import poisson, binom

# ==========================================
# 辅助数学函数：离散分布运算
# ==========================================

def trim_distribution(dist, threshold=1e-9):
    """
    截断概率极小的尾部，防止字典无限膨胀，保证计算速度。
    """
    return {k: v for k, v in dist.items() if v > threshold}

def convolve_distributions(dist_a, dist_b):
    """
    计算两个独立离散分布 A + B 的和的分布 (卷积)。
    Used for: Effective Demand = Local Demand + Incoming Shortage
    """
    new_dist = {}
    for val_a, prob_a in dist_a.items():
        for val_b, prob_b in dist_b.items():
            new_val = val_a + val_b
            new_dist[new_val] = new_dist.get(new_val, 0.0) + prob_a * prob_b
    return trim_distribution(new_dist)

def get_positive_negative_expectations(net_inventory_dist):
    """
    给定净库存 (Net = OnHand - Backorder) 的分布，
    计算 E[On_Hand] 和 E[Backorder]。
    E[(x)+] and E[(x)-]
    """
    exp_pos = 0.0 # E[max(0, x)]
    exp_neg = 0.0 # E[max(0, -x)]
    
    for val, prob in net_inventory_dist.items():
        if val > 0:
            exp_pos += val * prob
        elif val < 0:
            exp_neg += (-val) * prob
            
    return exp_pos, exp_neg

def get_shortage_distribution(net_inventory_dist):
    """
    从净库存分布中提取缺货量的分布 B = max(0, -Net)。
    用于传递给下游。
    """
    shortage_dist = {}
    for val, prob in net_inventory_dist.items():
        b_val = max(0, -val)
        if b_val > 0: # 只有大于0的缺货才需要传递
            shortage_dist[b_val] = shortage_dist.get(b_val, 0.0) + prob
            
    # 注意：这里返回的分布不包含 0 的情况。
    # 如果父节点没缺货(B=0)，在逻辑中直接处理，不进入二项分布计算以节省时间。
    return shortage_dist

def binomial_split(shortage_amount, theta):
    """
    计算 Binomial(n=shortage_amount, p=theta) 的分布。
    """
    if shortage_amount == 0:
        return {0: 1.0}
    
    dist = {}
    k_list = np.arange(shortage_amount + 1)
    probs = binom.pmf(k_list, shortage_amount, theta)
    
    for k, p in zip(k_list, probs):
        if p > 1e-9:
            dist[k] = p
    return dist

# ==========================================
# 核心评估逻辑
# ==========================================

def evaluate_cost(node_lis, local_stocks):
    """
    计算给定本地库存策略下的系统精确期望成本。
    对应论文 Equation (2) 和 Equation (3)。
    
    Args:
        node_lis: 节点列表
        local_stocks: 对应的本地库存 s_i 列表 (顺序与 node_lis 一致)
        
    Returns:
        total_expected_cost: float
    """
    
    # 1. 建立索引和映射
    nodes_by_id = {node.node: node for node in node_lis}
    
    # 2. 将本地库存 s_i 转换为梯级库存 S_i
    # 公式: S_i = s_i + sum(S_child)
    # 需要自底向上计算
    echelon_stocks = {}
    
    # 假设 node_lis 是按层级或是广度优先排列的，我们需要逆序 (从叶子到根)
    for i in range(len(node_lis) - 1, -1, -1):
        node = node_lis[i]
        s_i = local_stocks[i] # 获取传入的决策变量
        
        sum_child_S = 0
        for child in node.successors:
            sum_child_S += echelon_stocks[child.node]
            
        echelon_stocks[node.node] = s_i + sum_child_S
        
    # 3. 自顶向下递归评估成本
    # 初始状态：根节点接收到的缺货为 0 ({0: 1.0})
    root = node_lis[0] # 假设第一个是根
    initial_shortage = {0: 1.0}
    
    return recursive_evaluation(root, initial_shortage, echelon_stocks)


def recursive_evaluation(node, incoming_shortage_dist, echelon_stocks):
    """
    递归计算以 node 为根的子树的总期望成本。
    
    Args:
        node: 当前节点对象
        incoming_shortage_dist: 来自父节点的缺货分布 (字典)
        echelon_stocks: 预计算好的梯级库存字典
        
    Returns:
        cost: 该子树的总期望成本
    """
    
    # --- A. 准备当前节点的参数 ---
    S_i = echelon_stocks[node.node]
    
    # 构建本地提前期需求分布 D_i
    mu = node.demand_rate * node.l
    limit = int(poisson.ppf(0.99999, mu))
    probs = poisson.pmf(np.arange(limit + 1), mu)
    demand_dist = {k: probs[k] for k in range(limit + 1) if probs[k] > 1e-9}
    
    # --- B. 计算有效净梯级库存分布 ---
    # Net_Inventory = S_i - (Demand + Incoming_Shortage)
    #               = S_i - Total_Load
    
    # 1. 卷积计算总负载 (Demand + Incoming Backorder)
    total_load_dist = convolve_distributions(demand_dist, incoming_shortage_dist)
    
    # 2. 计算净库存分布
    net_inventory_dist = {}
    for load, prob in total_load_dist.items():
        net_val = S_i - load
        net_inventory_dist[net_val] = net_inventory_dist.get(net_val, 0.0) + prob
        
    # --- C. 计算成本分量 ---
    # 计算 E[Net]^+ (On Hand) 和 E[Net]^- (Shortage)
    exp_on_hand, exp_shortage = get_positive_negative_expectations(net_inventory_dist)
    
    # 1. 梯级持有成本 (Echelon Holding Cost)
    # Cost += H_i * E[On_Hand] (注意：这里使用的是 Clark-Scarf 变换后的梯级持有成本)
    # 依据 Equation 3: hat_C = H_i * x + ...
    current_node_cost = node.H * exp_on_hand
    
    # 2. 叶节点惩罚成本 (仅叶节点)
    if node.is_leaf:
        # 依据 Equation 3 对于叶节点: (h_i + b_i) * [x]^-
        # 注意: 这里 [x]^- 就是实际的缺货量
        # 为什么是 h_i + b_i ? 因为前面 H_i = h_i - h_parent
        # 数学上 H_i * x + (h+b)*x- 等价于 h*I + b*B
        current_node_cost += (node.h + node.b) * exp_shortage
        
    # --- D. 递归处理子节点 (处理缺货分配) ---
    children_cost = 0.0
    
    if not node.is_leaf:
        # 提取当前节点的缺货分布 (用于分给子节点)
        # V = [S_i - Total_Load]^-
        current_shortage_dist = get_shortage_distribution(net_inventory_dist)
        
        # 如果有缺货，需要拆分
        if current_shortage_dist:
            # 这是一个复杂的混合分布，我们需要遍历每一个可能的缺货值 v
            # 然后对 v 进行二项分布拆分
            
            # 为了避免在这个嵌套层级做极其昂贵的“分布的分布”计算，
            # 我们先计算出传递给每个子节点的“总缺货分布”。
            
            # 初始化子节点的输入缺货分布
            child_incoming_dists = {child: {} for child in node.successors}
            
            # 遍历当前节点的每一个缺货可能值 v
            for v, v_prob in current_shortage_dist.items():
                
                # 对于这个特定的 v，根据 theta 分配给子节点
                # 这是一个多项分布拆分 (Multinomial Split)，或者连续二项拆分
                # RO 算法假设只有两个子节点 (Binary Tree)，所以可以用 Binomial
                
                # 假设 successors 是 [child1, child2]
                if len(node.successors) == 2:
                    c1 = node.successors[0]
                    c2 = node.successors[1]
                    
                    # 给 c1 分配 k (Binomial), 剩下的 v-k 给 c2
                    # Bin(v, theta1) ? 
                    # 注意：theta 是相对父节点流量的比例。
                    # 如果 theta1 + theta2 = 1，则直接用 theta1
                    
                    theta = c1.theta
                    
                    # 预计算 Binomial 分布
                    k_list = np.arange(v + 1)
                    bin_probs = binom.pmf(k_list, v, theta)
                    
                    for k, p in zip(k_list, bin_probs):
                        joint_prob = v_prob * p
                        if joint_prob < 1e-10: continue
                        
                        # Child 1 receives k
                        child_incoming_dists[c1][k] = child_incoming_dists[c1].get(k, 0.0) + joint_prob
                        # Child 2 receives v - k
                        child_incoming_dists[c2][v-k] = child_incoming_dists[c2].get(v-k, 0.0) + joint_prob
                        
                elif len(node.successors) == 1:
                    # 只有一个子节点，全部给它
                    c1 = node.successors[0]
                    child_incoming_dists[c1][v] = child_incoming_dists[c1].get(v, 0.0) + v_prob
                    
                else:
                    # 超过2个子节点 (论文主要是二叉树，但为了通用性)
                    # 这里实现起来极复杂(多项分布)。
                    # 简单起见，假设网络是二叉树。如果不是，需要级联 Binomial。
                    raise NotImplementedError("Current Exact Eval supports max 2 successors per node")

            # 递归调用
            for child in node.successors:
                # 加上没缺货的情况 (0)
                # 上面的循环只处理了 v > 0。
                # 所有的 sum(current_shortage_dist.values()) 是缺货总概率 P(Backorder > 0)
                # 没缺货的概率 P(Backorder=0) 应该加到子节点接收到 0 的概率里。
                
                prob_no_shortage = 1.0 - sum(current_shortage_dist.values())
                if prob_no_shortage > 1e-9:
                    child_incoming_dists[child][0] = child_incoming_dists[child].get(0, 0.0) + prob_no_shortage
                
                # Trim 并递归
                trimmed_dist = trim_distribution(child_incoming_dists[child])
                children_cost += recursive_evaluation(child, trimmed_dist, echelon_stocks)
                
        else:
            # 当前节点完全没有缺货 (概率极低，或者库存很高)
            # 子节点接收到的缺货全是 0
            for child in node.successors:
                children_cost += recursive_evaluation(child, {0: 1.0}, echelon_stocks)

    return current_node_cost + children_cost