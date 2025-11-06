import matplotlib.pyplot as plt
import time
import numpy as np
import range_attack
import networkx as nx
from align import *
from snapping import *
from metrics import *


class EvenLessExpRunner:
    def __init__(self, new_responses, points):
        self.new_responses = new_responses
        self.points = points
        self.original_result = None
        self.processed_result = None
        self.aligned_result = None
        self.snapped_result = None

    def run(self, dim=2):
        start = time.time()
        G, used = range_attack.general(self.new_responses)
        if "nh" in self.points or "crg" in self.points:
            dim = 3
            pos = nx.kamada_kawai_layout(G, dim=dim)
            self.original_result = pos
        else:
            dim = 2
            pos = nx.kamada_kawai_layout(G, dim=dim)
            self.original_result = pos
        end = time.time()
        print("Time original method: ", end - start)
        self.processed_result = []
        for i in self.original_result:
            self.processed_result.append(self.original_result[i])
        return self.processed_result

    def run_ori(self, dim=2):
        start = time.time()
        G, used = range_attack.general(self.new_responses)
        # if "nh" in self.points or "crg" in self.points:
        #     dim = 3
        #     pos = nx.kamada_kawai_layout(G, dim=dim)
        #     self.original_result = pos
        # else:
        #     dim = 2
        #     pos = nx.kamada_kawai_layout(G, dim=dim)
        #     self.original_result = pos
        pos = nx.kamada_kawai_layout(G, dim=dim)
        self.original_result = pos
        end = time.time()
        # print("Time original method: ", end - start)
        return self.original_result

    def align_coords(self, grid_size):
        # 显示原始布局并等待用户手动对齐
        print("\nPlease manually align the points in the figure.")
        print("Click and drag points to adjust their positions.")
        print("Press Enter when done.")
        
        # 使用plot_result的交互功能
        self.plot_result(self.processed_result)
        
        # 获取当前图形中的点位置
        fig = plt.gcf()
        ax = fig.gca()
        aligned_points = []
        
        # 获取scatter plot中的点
        for collection in ax.collections:
            if isinstance(collection, plt.matplotlib.collections.PathCollection):
                offsets = collection.get_offsets()
                aligned_points = offsets.tolist()
                break
        
        plt.close()
        
        if not aligned_points:
            print("Warning: No points were captured. Using original points.")
            aligned_points = self.processed_result
        
        # 对对齐后的点进行自动缩放
        from align import robust_scale_coords
        self.aligned_result = robust_scale_coords(aligned_points, [1, grid_size[0] - 1], [1, grid_size[1] - 1])
        
        # 显示缩放后的结果
        print("Scaled layout:")
        self.plot_result(self.aligned_result)
        plt.show()
        
        return self.aligned_result

    def snap_coords(self, grid_size):
        self.snapped_result = simulated_annealing_snap(self.aligned_result, grid_size)
        return self.snapped_result

    def plot_original_result(self):
        if "nh" in self.points or "crg" in self.points:
            self.plot3D()
        else:
            self.plot2D()

    def plot2D(self):
        X = []
        Y = []

        for i in self.original_result:
            point = self.original_result[i]
            X.append(point[0])
            Y.append(point[1])

        fig = plt.figure()

        plt.scatter(X, Y, s=5)
        plt.gca().set_aspect('equal')
        plt.show()

    def plot3D(self):
        X = []
        Y = []
        Z = []

        for i in self.original_result:
            point = self.original_result[i]
            X.append(point[0])
            Y.append(point[1])
            Z.append(point[2])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(X, Y, Z, s=5)
        plt.gca().set_aspect('equal')
        plt.show()

    def plot_result(self, coords):
        if "nh" in self.points or "crg" in self.points:
            self.plot3D()
        else:
            coords = np.array(coords)
            plt.figure()
            scatter = plt.scatter(coords[:, 0], coords[:, 1])
            plt.title('2D Visualization of Points after Dimensionality Reduction')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.grid(True)
            
            # 添加交互式调整功能
            def on_key(event):
                nonlocal coords  # 声明coords为非局部变量
                if event.key == 'enter':
                    plt.close()
                elif event.key == 'r':  # 按r键旋转
                    # 获取当前图形的中心点
                    center = np.mean(coords, axis=0)
                    # 旋转1度
                    angle = np.pi/180
                    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                              [np.sin(angle), np.cos(angle)]])
                    # 将点移动到原点，旋转，再移回
                    coords_centered = coords - center
                    coords_rotated = np.dot(coords_centered, rotation_matrix)
                    coords = coords_rotated + center
                    scatter.set_offsets(coords)
                    plt.draw()
            
            # 绑定事件
            fig = plt.gcf()
            fig.canvas.mpl_connect('key_press_event', on_key)
            
            print("\nControls:")
            print("Press 'r' to rotate the entire plot by 90 degrees")
            print("Press 'Enter' when done")
            
            plt.show()
