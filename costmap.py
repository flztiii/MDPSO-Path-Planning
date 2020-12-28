#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

Costmap Generation
author: flztiii

"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Costmap:
    def __init__(self, cost_map, resolution, root_x=0.0, root_y=0.0, root_theta=0.0):
        # 设置底图
        self.cost_map_ = cost_map
        self.height_ = cost_map.shape[0]
        self.width_ = cost_map.shape[1]
        # 设置分辨率
        self.resolution_ = resolution
        # 设置左下角真实坐标
        self.root_x_ = root_x
        self.root_y_ = root_y
        self.root_theta_ = root_theta
    
    # 判断是否超过边界
    def isVerify(self, x, y):
        if (x >= 0 and x <= self.width_ and y >= 0 and y <= self.height_):
            return True
        else:
            return False

    # 判断栅格是否被占据
    def isOccupied(self, x, y):
        if self.cost_map_[y][x] == 255:
            return False
        else:
            return True
    
    # 计算对应栅格坐标
    def getGridCoordinate(self, px, py):
        x, y = self.calcNewCoordinate(px, py)
        ix = int(x / self.resolution_)
        iy = int(y / self.resolution_)
        return ix, iy
    
    # 计算对应真实坐标
    def getCartesianCoordiante(self, ix, iy):
        x = float(ix * self.resolution_)
        y = float(iy * self.resolution_)
        px, py = self.calcOldCoordinate(x, y)
        return px, py
    
    # 坐标转换
    def calcNewCoordinate(self, x, y):
        nx = (x - self.root_x_) * np.cos(self.root_theta_) + (y - self.root_y_) * np.sin(self.root_theta_)
        ny = -(x - self.root_x_) * np.sin(self.root_theta_) + (y - self.root_y_) * np.cos(self.root_theta_)
        return nx, ny

    # 坐标反转换
    def calcOldCoordinate(self, x, y):
        ox = self.root_x_ + x * np.cos(self.root_theta_) - y * np.sin(self.root_theta_)
        oy = self.root_y_ + x * np.sin(self.root_theta_) + y * np.cos(self.root_theta_)
        return ox, oy

if __name__ == "__main__":
    image = Image.open("./data/test1.png").convert('L')
    cost_map = np.array(image)
    cost_map = cost_map[::-1,:]
    grid = Costmap(cost_map, 0.05)
    plt.imshow(cost_map, cmap ='gray')
    plt.gca().invert_yaxis()
    plt.show()