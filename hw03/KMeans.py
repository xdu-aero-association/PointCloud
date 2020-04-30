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
        """
        """
        dimensions = len(a)

        _sum = 0
        for dimension in range(dimensions):
            difference_sq = (a[dimension] - b[dimension]) ** 2
            _sum += difference_sq
        return sqrt(_sum)

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        # step1: 随机选择ｋ个点作为中心点
        num_points = data.shape[0]
        cluster_assment = np.mat(np.zeros((num_points, 2)))
        cluster_change = True
        self.centers = np.empty(shape=[0, data.shape[1]])
        for i in np.random.choice(data.shape[0], self.k_):
            self.centers = np.append(self.centers, [data[i]], axis=0)


        num_loop = 0

        while num_loop < self.max_iter_ and cluster_change:
            cluster_change = False
            for point_index in range(num_points):
                min_dist = 9999999999
                min_index = 0
                # print(point)
                # step2: 计算点到中心的距离，归类到距离最小的一组
                for center_index in range(self.k_):
                    dis_sqrt = self.distance_sqrt(self.centers[center_index, :], data[point_index, :])
                    if dis_sqrt < min_dist:
                        min_dist = dis_sqrt
                        min_index = center_index

                # step3: 将距离和索引保存，若所有样本没变化就退出循环
                if abs(cluster_assment[point_index, 0] - min_dist) > self.tolerance_:
                    cluster_change = True
                    cluster_assment[point_index, :] = min_index, min_dist

            # step4: 更新中心
            for center_index in range(self.k_):
                points_incluster = data[np.nonzero(cluster_assment[:, 0].A == center_index)[0]]
                self.centers[center_index, :] = np.mean(points_incluster, axis=0)
            num_loop += 1



        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        num_p_data = p_datas.shape[0]
        for p_index in range(num_p_data):
            p_min_dis = 9999999999
            p_min_index = 0
            for c_index in range(self.centers.shape[0]):
                d_sqrt = self.distance_sqrt(p_datas[p_index, :], self.centers[c_index, :])
                # print(dis_sqrt)
                if p_min_dis > d_sqrt:
                    p_min_dis = d_sqrt
                    p_min_index = c_index
            result.append(p_min_index)


        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)
    # print(k_means.centers)

    cat = k_means.predict(x)
    print(cat)

