import numpy as np
from scipy.stats import poisson
import time

# ==========================================
# 辅助数学工具：Shang & Song 专用函数
# ==========================================

def get_G_inverse(target_prob, mu):
    """
    计算 Shang & Song 启发式所需的线性插值泊松逆函数 G^(-1)(y)。
    对应论文 Eq (6) 上方的描述。
    G_D(x) 是泊松 CDF F_D(x) 的线性插值版本。
    """
    if target_prob <= 0: return 0.0
    if target_prob >= 1: return float('inf')
    
    # 寻找整数 k 使得 F(k-1) < target_prob <= F(k)
    # 使用 scipy 的 ppf 找到这个 k
    k = int(poisson.ppf(target_prob, mu))
    
    # 获取 F(k) and F(k-1)
    cdf_k = poisson.cdf(k, mu)
    cdf_k_minus_1 = poisson.cdf(k - 1, mu)
    
    # 论文中定义的 G_D(x) 是分段线性的。
    # 在区间 (k-0.5, k+0.5) 附近进行插值。
    # 实际上，我们需要求解 x 使得 G(x) = target_prob
    # G(x) 在整数点 k 附近的值等于 F(k)。
    # 我们可以简化为在 [k-1, k] 区间内进行线性插值求解。
    
    # 简单线性插值逻辑 (Linear Interpolation):
    # prob = cdf_k_minus_1 + (x - (k-1)) * (cdf_k - cdf_k_minus_1)
    # x - (k-1) = (prob - cdf_k_minus_1) / (cdf_k - cdf_k_minus_1)
    # x = (k-1) + ...
    
    # 注意：论文中 G(x) 的定义在 x >= 0.5 时略有偏移 (x-0.5)，
    # 但核心思想是利用概率差进行插值得到连续的 S 值。
    
    if cdf_k == cdf_k_minus_1: # 概率密度极小
        return float(k)
        
    fraction = (target_prob - cdf_k_minus_1) / (cdf_k - cdf_k_minus_1)
    # 修正到论文的坐标系 (x 取值对应库存)
    # 通常 Shang & Song 的 G^-1 返回的是连续库存值
    return (k - 1) + fraction


def poisson_loss(mu, s):
    """泊松损失函数 E[(D-s)+]"""
    if mu <= 0: return 0.0
    # 利用 scipy 的 survival function (sf = 1 - cdf)
    # 公式: mu * P(D >= s) - s * P(D >= s+1)
    return mu * poisson.sf(s-1, mu) - s * poisson.sf(s, mu)

def inverse_poisson_loss(mu, target_loss):
    """寻找最小整数 s 使得 E[(D-s)+] <= target_loss"""
    if target_loss >= mu: return 0
    if target_loss <= 1e-9: return int(mu + 6 * np.sqrt(mu)) + 1
    
    low = 0
    high = int(mu + 6 * np.sqrt(mu)) + 10
    best_s = high
    
    while low <= high:
        mid = (low + high) // 2
        loss = poisson_loss(mu, mid)
        if loss <= target_loss:
            best_s = mid
            high = mid - 1
        else:
            low = mid + 1
    return best_s

# ==========================================
# DA 主函数
# ==========================================

def da(node_lis):
    start_time = time.time()
    
    # 1. 识别叶节点
    leaves = [n for n in node_lis if n.is_leaf]
    
    # 存储全局聚合的缺货量 E[B_i]
    aggregated_backorders = {n.node: 0.0 for n in node_lis}
    
    # =========================================================
    # Step 1 & 2: Decomposition & Serial System Optimization
    # =========================================================
    
    for leaf in leaves:
        # A. 构建串行链 (Leaf -> ... -> Root)
        # 注意：为了计算方便，我们构建 list: [Root, ..., Leaf]
        chain = []
        curr = leaf
        while curr:
            chain.append(curr)
            curr = curr.predecessor
        chain.reverse() # 现在是 [Root, ..., Leaf]
        
        # B. 准备参数
        # 串行系统各节点的需求率都等于叶节点需求率
        lam_w = leaf.demand_rate
        
        # 计算该链上的累积参数
        # H_cum_from_root[i]: 从根节点到节点 i 的 H 之和
        H_cum_from_root = {}
        curr_H_sum = 0.0
        for node in chain:
            curr_H_sum += node.H
            H_cum_from_root[node.node] = curr_H_sum
            
        # L_cum_to_leaf[i]: 从节点 i 到叶节点的提前期之和 (用于计算提前期需求)
        L_cum_to_leaf = {}
        curr_L_sum = 0.0
        # 倒序遍历链计算下游 L 之和
        for node in reversed(chain):
            curr_L_sum += node.l
            L_cum_to_leaf[node.node] = curr_L_sum
            
        # C. 计算串行系统梯级库存 S^SS (Pass 1)
        serial_S = {} # 存储 S_iw^SS
        
        for i, node in enumerate(chain):
            # 1. 有效提前期需求参数
            mu_D = lam_w * L_cum_to_leaf[node.node]
            
            # 2. 准备持有成本参数 (Shang & Song Heuristic)
            # b: 串行系统末端缺货成本
            b = leaf.b
            
            # H_upstream: 根节点到父节点的 H 之和 (如果是根节点则为0)
            if i == 0:
                H_upstream = 0.0
            else:
                H_upstream = H_cum_from_root[chain[i-1].node]
            
            # H_current: 根节点到当前的 H 之和
            H_current_cum = H_cum_from_root[node.node]
            
            # 3. 计算 S^SS
            if node.is_leaf:
                # 叶节点：直接用 Newsvendor 公式 (Eq 11 上半部分)
                # Ratio = (b + H_upstream) / (b + H_current)
                ratio = (b + H_upstream) / (b + H_current_cum)
                
                # 叶节点使用标准逆函数 F^-1 (因为是最终库存)
                S_val = poisson.ppf(ratio, mu_D)
            else:
                # 非叶节点：使用 Shang & Song 平均近似 (Eq 11 下半部分)
                # Lower Bound Ratio (假设下游持有成本极高/库存为0)
                ratio_low = (b + H_upstream) / (b + H_current_cum)
                
                # Upper Bound Ratio (假设下游持有成本为0/库存无限)
                # 分母只包含到 Root 的路径持有成本 + Root 自身的调整?
                # 依据 Shang & Song (2003) Eq 17:
                # Lower index: h_k (local) + ...
                # 简化理解：
                # Ratio 1: 考虑本地 H 的影响
                # Ratio 2: 忽略本地 H，假设由整个系统分担 (通常近似为只加 Root 的 H 或极小 H)
                # 根据论文 Eq 6 (OWMR): Ratio2 = b / (b + h0) -> h0 是根节点本地持有
                
                # 在 General Network 中，对应 Upper Bound 的通常是:
                # (b + H_upstream) / (b + H_upstream + H_root_echelon) ?
                # 或者更标准的 S&S: (b + H_upstream) / (b + H_upstream) -> 1 (Infinite stock)
                
                # 我们采用最稳健的 Shang & Song 实现：
                # 两个极值点基于“当前节点增加库存的边际成本” vs “根节点增加库存的边际成本”
                
                h_current_echelon = node.H
                h_root_echelon = chain[0].H 
                
                # Ratio 1 (Cost if held here):
                r1 = (b + H_upstream) / (b + H_upstream + h_current_echelon)
                
                # Ratio 2 (Cost if held at root):
                # 对应论文 Eq 6 的第二项 G^-1( b / (b+h0) )
                r2 = (b + H_upstream) / (b + H_upstream + h_root_echelon)
                
                # 使用线性插值逆函数 G^-1
                val1 = get_G_inverse(r1, mu_D)
                val2 = get_G_inverse(r2, mu_D)
                
                S_val = 0.5 * (val1 + val2)
            
            serial_S[node.node] = S_val
            
        # D. 计算串行系统本地库存 s^d 和 缺货 E[B] (Pass 2)
        # 必须先算出所有 S 才能算 s
        
        for i, node in enumerate(chain):
            S_curr = serial_S[node.node]
            
            if node.is_leaf:
                s_d = S_curr
            else:
                # S_next 是下游节点在这个串行系统中的 S
                S_next = serial_S[chain[i+1].node]
                s_d = S_curr - S_next
            
            # 计算该节点在这个串行链中面临的缺货
            # 需求 ~ Poisson(lambda_w * L_node)
            # 注意：这里是用节点的本地提前期 L_i，不是累积提前期
            mu_local = lam_w * node.l
            
            # E[B_iw]
            eb = poisson_loss(mu_local, s_d)
            
            # 累加到全局
            aggregated_backorders[node.node] += eb

    # =========================================================
    # Step 3: Aggregation (Backorder Matching)
    # =========================================================
    
    local_stocks = [0] * len(node_lis)
    
    for node in node_lis:
        target_B = aggregated_backorders[node.node]
        
        # 节点的真实总需求分布参数
        # lambda_i 是所有流经该节点的需求之和
        mu_total = node.demand_rate * node.l
        
        if mu_total <= 1e-9:
            local_stocks[node.node] = 0
            continue
            
        # 求解 s_a
        s_a = inverse_poisson_loss(mu_total, target_B)
        local_stocks[node.node] = int(s_a)
        
    elapsed_time = time.time() - start_time
    return local_stocks, elapsed_time