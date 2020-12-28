#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

Path planning using MDPSO
Author: flztiii

"""

from costmap import Costmap
from MDPSO import MDPSO
from cubic_spline import CubicSpline2D
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime

# 惩罚损失系数
P_r = 50.0
# 栅格分辨率
resolution = 0.05

if __name__ == "__main__":
    # 构建栅格地图
    image = Image.open("./data/test2.png").convert('L')
    cost_map = np.array(image)
    cost_map = cost_map[::-1,:]
    grid = Costmap(cost_map, resolution)

    # 设置起点和终点
    start_x = 4.97397422791
    start_y = 8.07954311371

    goal_x = 11.9850091934
    goal_y = 41.4916610718

    # # 设置优化参数
    # # 定义输入维度
    # points_num = 5
    # dimension = points_num * 2
    # # 定义搜索空间
    # lower_boundary = np.array([0.0, 0.0] * points_num)
    # upper_boundary = [grid.resolution_ * (grid.width_ - 1), grid.resolution_ * (grid.height_ - 1)] * points_num
    # # 定义最大迭代次数
    # max_iteration = 100
    # # 定义粒子数量
    # swarm_population = 100
    # # 定义优化目标函数
    # def objective_function(x):
    #     # 得到路点
    #     waypoints_x = [start_x]
    #     waypoints_y = [start_y]
    #     for i in range(0, len(x)):
    #         if i%2 == 0:
    #             waypoints_x.append(x[i])
    #         else:
    #             waypoints_y.append(x[i])
    #     waypoints_x.append(goal_x)
    #     waypoints_y.append(goal_y)
    #     assert(len(waypoints_x) == len(waypoints_y))
    #     print("----------------waypoints-----------")
    #     print(waypoints_x, waypoints_y)
    #     # 清除重复点
    #     init_index = 1
    #     while init_index < len(waypoints_x):
    #         if np.abs(waypoints_x[init_index] - waypoints_x[init_index - 1]) + np.abs(waypoints_y[init_index] - waypoints_y[init_index - 1]) < 2 * resolution:
    #             del waypoints_x[init_index]
    #             del waypoints_y[init_index]
    #             init_index = 1
    #         init_index += 1
    #     # 构建三次样条
    #     cubic_spline = CubicSpline2D(waypoints_x, waypoints_y)
    #     # 得到样条总长度(不准确长度)
    #     total_length = cubic_spline.s_[-1]
    #     # 对路径进行采样
    #     gap = resolution
    #     sample_s = np.arange(0.0, cubic_spline.s_[-1], gap)
    #     points_x, points_y = cubic_spline.calcPosition(sample_s)
    #     # 判断路径经过的栅格
    #     path_occupied_grid = set()
    #     for i in range(0, len(sample_s)):
    #         ix, iy = grid.getGridCoordinate(points_x[i], points_y[i])
    #         # print(ix, iy)
    #         if grid.isVerify(ix, iy):
    #             if grid.isOccupied(ix, iy):
    #                 # 被占据
    #                 path_occupied_grid.add(grid.getIndex(ix, iy))
    #         else:
    #             # 超过边界
    #             path_occupied_grid.add(grid.getIndex(ix, iy))
    #     path_occupied_grid_num = len(path_occupied_grid)
    #     return total_length + path_occupied_grid_num * P_r

    # # 得到优化器
    # optimizer = MDPSO(objective_function, dimension, lower_boundary, upper_boundary, swarm_population, max_iteration)
    # # 进行优化
    # result = optimizer.startOptimization()
    # print(result)

    result = [18.07098424, 24.42865884, 22.23142493, 30.887967, 23.81531445, 41.21824627, 23.99152327, 41.80546431, 13.13170801, 42.74456121]
    # 得到最终路径
    # 得到路点
    waypoints_x = [start_x]
    waypoints_y = [start_y]
    for i in range(0, len(result)):
        if i%2 == 0:
            waypoints_x.append(result[i])
        else:
            waypoints_y.append(result[i])
    waypoints_x.append(goal_x)
    waypoints_y.append(goal_y)
    # 构建三次样条
    cubic_spline = CubicSpline2D(waypoints_x, waypoints_y)
    # 对路径进行采样
    gap = resolution
    sample_s = np.arange(0.0, cubic_spline.s_[-1], gap)
    points_x, points_y = cubic_spline.calcPosition(sample_s)
    # 保存路径点到csv文件中
    path_recorder_file = open("./log/" + datetime.datetime.now().strftime('%Y-%m-%d')+".csv", 'w')
    for i in range(0, len(sample_s)):
        path_recorder_file.write(str(points_x[i]) + "," + str(points_y[i]) + "\n")
    path_recorder_file.close()
    # 映射到栅格中
    grid_points_x, grid_points_y = [], []
    for i in range(0, len(sample_s)):
        ix, iy = grid.getGridCoordinate(points_x[i], points_y[i])
        grid_points_x.append(ix)
        grid_points_y.append(iy)
    # 原始点映射到栅格中
    grid_waypoints_x, grid_waypoints_y = [], []
    for i in range(0, len(waypoints_x)):
        ix, iy = grid.getGridCoordinate(waypoints_x[i], waypoints_y[i])
        grid_waypoints_x.append(ix)
        grid_waypoints_y.append(iy)
    # 进行可视化
    plt.figure()
    plt.plot(grid_points_x, grid_points_y)
    plt.scatter(grid_waypoints_x, grid_waypoints_y)
    plt.imshow(cost_map, cmap ='gray')
    plt.gca().invert_yaxis()
    plt.show()