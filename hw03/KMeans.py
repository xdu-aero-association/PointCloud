# 文件功能： 实现 K-Means 算法

from math import sqrt
import numpy as np


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers = None

    def distance_sqrt(self, a, b):
        difference = (a - b) ** 2
        _sum = np.sum(difference, axis=0)
        return sqrt(_sum)

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        # step1: 随机选择ｋ个点作为中心点
        num_points = data.shape[0]
        cluster_assment = np.mat(np.zeros((num_points, 2)))
        # print(cluster_assment)
        cluster_change = True
        self.centers = np.empty(shape=[0, data.shape[1]])
        for i in np.random.choice(data.shape[0], self.k_):
            self.centers = np.append(self.centers, [data[i]], axis=0)


        num_loop = 0

        while num_loop < self.max_iter_ and cluster_change:
            cluster_change = False

            center_difference = np.zeros((num_points, self.k_))
            for center_index in range(self.k_):
                difference = data - self.centers[center_index, :]
                difference = np.sqrt(np.sum(np.square(difference), axis=1))
                center_difference[:, center_index] = difference.reshape(num_points)

            min_index = np.argmin(center_difference, axis=1)
            dist = np.min(center_difference, axis=1)

            for point_index in range(num_points):
                if abs(cluster_assment[point_index, 0] - dist[point_index]) > self.tolerance_:
                    cluster_change = True
                    cluster_assment[point_index, :] = min_index[point_index], dist[point_index]



            # step4: 更新中心
            for center_index in range(self.k_):
                points_incluster = data[np.nonzero(cluster_assment[:, 0].A == center_index)[0]]
                if len(points_incluster) != 0:
                    self.centers[center_index, :] = np.mean(points_incluster, axis=0)
            num_loop += 1

        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始

        num_points = p_datas.shape[0]
        center_difference = np.zeros((num_points, self.k_))
        for center_index in range(self.k_):
            difference = p_datas - self.centers[center_index, :]
            difference = np.sqrt(np.sum(np.square(difference), axis=1))
            center_difference[:, center_index] += difference.reshape(num_points)

        result = np.argmin(center_difference, axis=1)

        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)
    print(k_means.centers)

    cat = k_means.predict(x)
    print(cat)

