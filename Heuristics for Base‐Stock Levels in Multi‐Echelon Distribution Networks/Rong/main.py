from PMU import pmu
#from RD import rd
from RO import ro
from DA import da
from calculate_cost import evaluate_cost
import random
import numpy as np
import math

lambda_i = 8

class Stage:
    def __init__(self,node,h,b,l,is_leaf):
        self.node = node
        self.h = h
        self.b = b if is_leaf else 0
        self.l = l
        self.is_leaf = is_leaf

        self.H = h
        self.demand_rate = lambda_i if self.is_leaf else 0
        self.theta = 0
        self.predecessor = None
        self.successors = []
        self.S_r0 = 0
        self.s_r = 0
    def add_successor(self,stage):
        stage.predecessor = self
        self.successors.append(stage)
        stage.H -= self.h

    def cal_cumulative(self):
        for node in self.successors:
            self.demand_rate += node.demand_rate
        for node in self.successors:
            node.theta = node.demand_rate / self.demand_rate
        
def get_layer(i):
    return math.ceil(math.log2(i + 2))

def liner(i, total):
    return get_layer(i) / total

def concave(i, total):
    return math.sqrt(get_layer(i) / total)

def convex(i, total):
    return math.pow(2, get_layer(i) - total)

if __name__ == "__main__":
    cost_func_names = ["Linear", "Concave", "Convex"]
    cost_func = [liner, concave, convex]

    header = f"{'Structure':<10} | {'Echelons':<8} | {'Metric':<12} | {'RD':>8} {'RO':>8} {'DA':>8} | {'PMU(s)':>8}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for cost_idx,cost in enumerate(cost_func):
        for m,echelon in enumerate([2]):
            ep = [[0.0 for _ in range(20)] for _ in range(3)]
            t = [[0.0 for _ in range(20)] for _ in range(4)]
            for count in range(20):
                random.seed(cost_idx*80+m*20+count)
                node_lis = []
                for i in range(2**echelon-1):
                    leaf = False if i < 2**(echelon-1)-1 else True
                    if leaf:
                        l = random.uniform(0.1,0.25)
                        b = random.uniform(9,39)
                    else:
                        l = random.uniform(0.1,0.5)
                        b = None
                    node_lis.append(Stage(i,cost(i,echelon),b,l,leaf))
                for i in range(2**(echelon-1)-1):
                    node_lis[i].add_successor(node_lis[2*i+1])
                    node_lis[i].add_successor(node_lis[2*i+2])
                for i in range(2**(echelon-1)-2,-1,-1):
                    node_lis[i].cal_cumulative()
                pmu_base,t[0][count] = pmu(node_lis)
                #rd_base,t[1][count] = rd(node_lis)
                ro_base,t[2][count] = ro(node_lis)
                da_base,t[3][count] = da(node_lis)
                c1 = evaluate_cost(node_lis,pmu_base)
                #c2 = evaluate_cost(node_lis,rd_base)
                c3 = evaluate_cost(node_lis,ro_base)
                c4 = evaluate_cost(node_lis,da_base)
                #ep[0][count] = 100*(c2-c1)/c1
                ep[1][count] = 100*(c3-c1)/c1
                ep[2][count] = 100*(c4-c1)/c1
            means = [np.mean(ep[k]) for k in range(3)]
            medians = [np.median(ep[k]) for k in range(3)]
            maxs = [np.max(ep[k]) for k in range(3)]
            stds = [np.std(ep[k], ddof=1) for k in range(3)]
            time_avgs = [np.mean(t[k]) for k in range(4)] # PMU, RD, RO, DA

            print(f"{cost_func_names[cost_idx]:<10} | {echelon:<8} | {'Mean %':<12} | {means[0]:8.2f} {means[1]:8.2f} {means[2]:8.2f} |")
            print(f"{'':<10} | {'':<8} | {'Median %':<12} | {medians[0]:8.2f} {medians[1]:8.2f} {medians[2]:8.2f} |")
            print(f"{'':<10} | {'':<8} | {'Max %':<12}    | {maxs[0]:8.2f} {maxs[1]:8.2f} {maxs[2]:8.2f} |")
            print(f"{'':<10} | {'':<8} | {'Std Dev':<12}  | {stds[0]:8.2f} {stds[1]:8.2f} {stds[2]:8.2f} |")
            print(f"{'':<10} | {'':<8} | {'Time (s)':<12} | {time_avgs[1]:8.3f} {time_avgs[2]:8.3f} {time_avgs[3]:8.3f} | {time_avgs[0]:8.3f}")
            print("-" * len(header))