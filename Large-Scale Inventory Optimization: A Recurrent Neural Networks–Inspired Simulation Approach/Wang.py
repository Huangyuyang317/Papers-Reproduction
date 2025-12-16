import math
from collections import deque

outer_service_time = 8
safety_coefficient = 2.1
mean = 40
market_std = 10 

P_VALUE = 1

class Stage:
    def __init__(self, name, lead_time, holding_cost, is_leaf):
        self.name = name
        self.lead_time = lead_time
        self.holding_cost = holding_cost
        self.is_leaf = is_leaf

        self.predecessors = []
        self.successors = []

        self.cumulative_time = self.lead_time

        self.temp_degree = 0
        self.p = None
        self.rank = 0

        self.f = {}
        self.g = {}
        self.best_ans = {}

        self.final_S = None
        self.final_SI = None
        self.final_base_stock = None
        
        self.effective_sigma = 0 

    def add_successor(self, stage):
        self.successors.append(stage)
        self.temp_degree += 1
        stage.predecessors.append(self)
        stage.temp_degree += 1

    def calculate_cumulative_time(self):
        if self.predecessors:
            l = [i.cumulative_time for i in self.predecessors]
            self.cumulative_time += max(l)


def label(lis):
    ans = []
    cur = deque()
    for i in lis:
        if i.temp_degree == 1:
            cur.append(i)

    processed = set(cur)
    k = 1
    while cur:
        s = cur.popleft()
        s.rank = k
        k += 1
        for j in s.successors + s.predecessors:
            if j.rank > 0:
                continue
            j.temp_degree -= 1
            s.p = j
            if j.temp_degree == 1:
                if j not in processed:
                    cur.append(j)
                    processed.add(j)
        ans.append(s)
    return ans

def calculate_sigma_propagation(all_nodes, p_val):
    for node in all_nodes:
        get_node_sigma(node, p_val)

def get_node_sigma(node, p_val):
    if node.effective_sigma > 0:
        return node.effective_sigma
    
    if not node.successors: 
        node.effective_sigma = market_std
        return node.effective_sigma
    
    sum_val = 0
    for succ in node.successors:
        s = get_node_sigma(succ, p_val)
        sum_val += s ** p_val
    
    node.effective_sigma = sum_val ** (1.0 / p_val)
    return node.effective_sigma


def ck(stage, S, SI):
    Tk = stage.lead_time
    tau = SI + Tk - S

    if tau < 0:
        return float("inf")

    cost = 0
    cost += stage.holding_cost * safety_coefficient * stage.effective_sigma * math.sqrt(tau)

    for i in stage.predecessors:
        if i.rank < stage.rank:
            actual_lookup_SI = min(SI, i.cumulative_time)
            cost += i.f.get(actual_lookup_SI, float("inf"))

    for i in stage.successors:
        if i.rank < stage.rank:
            cost += i.g.get(S, float("inf"))

    return cost


def dp(ans):
    for i in range(len(ans) - 1):
        k = ans[i]
        p = ans[i].p
        Mk = k.cumulative_time
        Tk = k.lead_time

        if p in k.successors:  
            for S in range(Mk + 1):
                cost = float("inf")
                best_SI = -1
                lower_SI = max(0, S - Tk)
                upper_SI = Mk - Tk
                for SI in range(lower_SI, upper_SI + 1):
                    val = ck(k, S, SI)
                    if val < cost:
                        cost = val
                        best_SI = SI
                k.f[S] = cost
                k.best_ans[('f', S)] = best_SI
        else:  
            for SI in range(Mk - Tk + 1):
                cost = float("inf")
                best_S = -1
                lower_S = 0
                upper_S = (min(SI + Tk, outer_service_time)
                           if k.is_leaf else SI + Tk)
                for S in range(lower_S, upper_S + 1):
                    val = ck(k, S, SI)
                    if val < cost:
                        cost = val
                        best_S = S
                k.g[SI] = cost
                k.best_ans[('g', SI)] = best_S

    k = ans[-1]
    Mk = k.cumulative_time
    Tk = k.lead_time
    for SI in range(Mk - Tk + 1):
        cost = float("inf")
        best_S = -1
        lower_S = 0
        upper_S = (min(SI + Tk, outer_service_time)
                   if k.is_leaf else SI + Tk)
        for S in range(lower_S, upper_S + 1):
            val = ck(k, S, SI)
            if val < cost:
                cost = val
                best_S = S
        k.g[SI] = cost
        k.best_ans[('g', SI)] = best_S

    return k

def get_optimal_solution(root_node, all_nodes_ordered):
    min_total_cost = float("inf")
    best_root_SI = -1
    for SI, cost in root_node.g.items():
        if cost < min_total_cost:
            min_total_cost = cost
            best_root_SI = SI

    root_node.final_SI = best_root_SI
    root_node.final_S = root_node.best_ans[('g', best_root_SI)]

    reversed_nodes = list(reversed(all_nodes_ordered))

    for k in reversed_nodes[1:]:
        p = k.p
        if p in k.successors:
            requested_S = p.final_SI
            actual_lookup_S = min(requested_S, k.cumulative_time)
            k.final_S = actual_lookup_S

            k.final_SI = k.best_ans[('f', actual_lookup_S)]
            
        elif p in k.predecessors:
            k.final_SI = p.final_S
            k.final_S = k.best_ans[('g', k.final_SI)]

    for k in all_nodes_ordered:
        tau = k.final_SI + k.lead_time - k.final_S
        if tau < 0:
            k.final_base_stock = 0
        else:
            k.final_base_stock = tau * mean + safety_coefficient * k.effective_sigma * math.sqrt(tau)

    return min_total_cost

if __name__ == '__main__':
    A = Stage('A', 2, 1, False)
    B = Stage('B', 3, 3, False)
    C = Stage('C', 2, 4, False)
    D = Stage('D', 4, 6, False)
    E = Stage('E', 2, 12, False)
    F = Stage('F', 6, 20, False)
    G = Stage('G', 3, 13, False)
    H = Stage('H', 4, 8, False)
    I = Stage('I', 3, 4, False)
    J = Stage('J', 2, 50, True)

    A.add_successor(B)
    B.add_successor(C)
    C.add_successor(E)
    D.add_successor(E)
    E.add_successor(G)
    
    F.add_successor(J)
    G.add_successor(J)
    H.add_successor(J)
    I.add_successor(J)

    lis = [A, B, C, D, E, F, G, H, I, J]

    for i in lis:
        i.calculate_cumulative_time()
 
    calculate_sigma_propagation(lis, P_VALUE)
    
    print(f"P-Value set to: {P_VALUE}")
    print("Effective Sigma for each node:")
    for i in lis:
        print(f"  Node {i.name}: {i.effective_sigma:.2f}")

    ans = label(lis)
    root_node = dp(ans)
    min_cost = get_optimal_solution(root_node, ans)

    print("-" * 50)
    print(f"Minimum Total Cost: {min_cost:,.2f}")
    print("-" * 50)
    print(f"{'Stage':<10} | {'SI':<6} | {'S':<6} | {'Safety Stock':<12}")
    print("-" * 50)

    for node in lis:
        print(f"{node.name:<10} | {node.final_SI:<6} | {node.final_S:<6} | {node.final_base_stock:<12.2f}")