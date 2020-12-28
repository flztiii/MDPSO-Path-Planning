#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

Multimodal Delayed Particle Swarm Optimization
author:flztiii

"""

import numpy as np
import matplotlib.pyplot as plt
import random

# 定义全局变量
DEBUG = True

# 利用MDPSO算法计算目标函数objective_function的最小值
class MDPSO:
    # 参数初始化
    def __init__(self, objective_function, dimension, search_space, swarm_population = 20, max_iteration = 20000, threshold = 0.01):
        # 优化目标函数
        self.objective_function_ = objective_function
        # 输入维度
        self.dimension_ = dimension
        # 每个维度的边界
        self.lower_boundary_, self.upper_bondary_ = np.array(search_space[0]) * np.ones(dimension), np.array(search_space[1]) * np.ones(dimension)
        # 粒子群数量
        self.swarm_population_ = swarm_population
        # 最大迭代次数
        self.max_iteration_ = max_iteration
        # 收敛阈值
        self.threshold_ = threshold;
        # 当前迭代次数
        self.iteration_ = 0
        # 初始惯性权重
        self.w1_ = 0.9
        # 结束惯性权重
        self.w2_ = 0.4
        # 初始加速度参量
        self.c_1i_ = 2.5
        self.c_2i_ = 0.5
        # 结束加速度参量
        self.c_1f_ = 0.5
        self.c_2f_ = 2.5

    # 初始化粒子群计算
    def __initSwarms(self):
        # 初始化粒子群
        self.swarms_ = np.random.uniform(low=self.lower_boundary_, high=self.upper_bondary_, size=(self.swarm_population_, self.dimension_))
        # 每一个粒子的历史最优位置
        self.pbest_ = self.swarms_.copy()
        # 每一个粒子的历史最优值
        self.pbest_value_ = np.array([np.inf] * self.swarm_population_)
        # 粒子群的最优位置
        self.gbest_ = self.pbest_.mean(axis=0).reshape(1, -1)
        # 粒子群的最优值
        self.gbest_value_ = np.inf
        # 初始化粒子群进化速度
        v_high = self.upper_bondary_ - self.lower_boundary_
        self.evolve_velocity_ = np.random.uniform(low=-v_high, high=v_high, size=(self.swarm_population_, self.dimension_))
        # 粒子群记录器
        self.pbest_recorder_ = []
        self.pbest_value_recorder_ = []
        self.gbest_recorder_ = []
        self.gbest_value_recorder_ = []

    # 计算粒子与粒子群之间的距离
    def __distance(self, swarm_calc):
        distance = 0.0
        for swarm in self.swarms_:
            distance += np.linalg.norm(swarm_calc - swarm)
        distance = distance / float(self.swarm_population_ - 1)
        return distance
    
    # 得到粒子群中的最优
    def __bestSwarm(self, swarms):
        best = float("inf")
        best_swarm = swarms[0]
        for swarm in swarms:
            if self.objective_function_(swarm) < best:
                best = self.objective_function_(swarm)
                best_swarm = swarm
        return best_swarm

    # 粒子群进化
    def __evolve(self):
        # 计算惯性权重
        w = (self.w1_ - self.w2_) * (self.max_iteration_ - self.iteration_) / self.max_iteration_ + self.w2_
        # 计算加速度参量
        c1 = (self.c_1i_ - self.c_1f_) * (self.max_iteration_ - self.iteration_) / self.max_iteration_ + self.c_1f_
        c2 = (self.c_2i_ - self.c_2f_) * (self.max_iteration_ - self.iteration_) / self.max_iteration_ + self.c_2f_
        c3 = c1
        c4 = c2
        # 对于每一个粒子计算到其他粒子的距离
        d_s = []
        for swarm in self.swarms_:
            d_s.append(self.__distance(swarm))
        d_max = max(d_s)
        d_min = min(d_s)
        #　计算全局最优粒子到其他粒子距离
        d_g = self.__distance(self.gbest_)
        # 计算当前进化模式
        E_f = (d_g - d_min) / (d_max - d_min)
        # assert(E_f <= 1)
        # 根据进化模式计算进化参数
        s_i, s_g = 0.0, 0.0
        tau_i, tau_g = 0, 0

        if E_f < 0.25:
            # 当前已经在全局最优附近
            pass
        elif E_f < 0.5:
            # 探索局部最优附近区域
            s_i = E_f
            tau_i = int(np.floor(random.uniform(0,1) * float(self.iteration_)))
        elif E_f < 0.75:
            # 尽量搜索优化区域
            s_g = E_f
            tau_g = int(np.floor(random.uniform(0,1) * float(self.iteration_)))
        else:
            # 需要跳出局部最优
            s_i = E_f
            s_g = E_f
            tau_i = int(np.floor(random.uniform(0,1) * float(self.iteration_)))
            tau_g = int(np.floor(random.uniform(0,1) * float(self.iteration_)))

        assert(len(self.pbest_recorder_) == self.iteration_ + 1 and len(self.gbest_recorder_) == self.iteration_ + 1)
        assert(self.iteration_ - tau_i >= 0 and self.iteration_ - tau_g >= 0)
        
        # 开始进行进化
        for i,_ in enumerate(self.swarms_):
            # 计算随机量
            r1, r2, r3, r4 = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
            # 更新进化速度
            self.evolve_velocity_[i] = w * self.evolve_velocity_[i] + c1 * r1 * (self.pbest_[i] - self.swarms_[i]) + c2 * r2 * (self.gbest_ - self.swarms_[i]) + s_i * c3 * r3 * (self.pbest_recorder_[self.iteration_ - tau_i][i] - self.swarms_[i]) + s_g * c4 * r4 * (self.gbest_recorder_[self.iteration_ - tau_g] - self.swarms_[i])
            # self.evolve_velocity_[i] = w * self.evolve_velocity_[i] + c1 * r1 * (self.pbest_[i] - self.swarms_[i]) + c2 * r2 * (self.gbest_ - self.swarms_[i])
            # 进行进化
            self.swarms_[i] += self.evolve_velocity_[i]
            self.swarms_[i] = np.clip(self.swarms_[i], self.lower_boundary_, self.upper_bondary_)
    # 进行优化
    def startOptimization(self):
        # 初始化粒子群
        self.__initSwarms()
        # 初始化迭代次数
        self.iteration_ = 0
        # 开始进行优化
        while self.iteration_ < self.max_iteration_:
            # 计算pbest和gbest
            for i, swarm in enumerate(self.swarms_):
                value = self.objective_function_(swarm)
                if value < self.pbest_value_[i]:
                    self.pbest_value_[i] = value
                    self.pbest_[i] = swarm
            self.gbest_ = self.__bestSwarm(self.pbest_)
            self.gbest_value_ = self.objective_function_(self.gbest_)
            # 保存当前pbest和gbest信息
            self.pbest_recorder_.append(self.pbest_.copy())
            self.pbest_value_recorder_.append(self.pbest_value_.copy())
            self.gbest_recorder_.append(self.gbest_.copy())
            self.gbest_value_recorder_.append(self.gbest_value_.copy())
            # 判断是否迭代完成
            if self.gbest_value_ < self.threshold_:
                break;
            # 进行进化
            self.__evolve()
            # 更新当前迭代次数
            self.iteration_ += 1
        # 进行可视化
        if DEBUG:
            plt.figure()
            plt.plot(range(0, len(self.gbest_value_recorder_)), np.log(self.gbest_value_recorder_))
            plt.show()
        return self.gbest_

# 测试函数
def test1():
    # 定义目标函数
    def objective_function(x):
        return np.sum(x**2)
    # 定义输入维度
    dimension = 20
    # 定义搜索空间
    search_space = (-100, 100)
    # 初始化优化类
    optimizer = MDPSO(objective_function, dimension, search_space)
    # 开始进行优化
    result = optimizer.startOptimization()
    print(result, objective_function(result))

# 测试函数
def test2():
    # 定义目标函数
    def objective_function(x):
        return np.sum(np.abs(x)) + np.prod(np.abs(x))
    # 定义输入维度
    dimension = 20
    # 定义搜索空间
    search_space = (-10, 10)
    # 初始化优化类
    optimizer = MDPSO(objective_function, dimension, search_space)
    # 开始进行优化
    result = optimizer.startOptimization()
    print(result, objective_function(result))

# 测试函数
def test3():
    # 定义目标函数
    def objective_function(x):
        result = 0.0
        for i, _ in enumerate(x):
            for j in range(0, i):
                result += np.sum(x[0:i])**2
        return result
    # 定义输入维度
    dimension = 20
    # 定义搜索空间
    search_space = (-100, 100)
    # 初始化优化类
    optimizer = MDPSO(objective_function, dimension, search_space)
    # 开始进行优化
    result = optimizer.startOptimization()
    print(result, objective_function(result))

# 主函数
if __name__ == "__main__":
    test3()