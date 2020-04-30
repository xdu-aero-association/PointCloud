# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random, math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import KMeans

plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, max_iter=200):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        # 屏蔽开始
        self.Mu = None
        self.Var = None
        self.Pi = None
        self.W = None
        self.data = None
        self.n_points = None
        self.loglh = None

    def initialize(self, data):
        self.n_points = data.shape[0]
        self.data = data
        np.random.seed(0)
        a = np.random.randint(0, self.n_points, self.n_clusters)
        if self.n_clusters == 2:
            self.Mu = np.array([data[a[0], :], data[a[1], :]])
            self.Var = [[10, 10], [10, 10]]
        if self.n_clusters == 3:
            self.Mu = np.array([data[a[0], :], data[a[1], :], data[a[2], :]])
            self.Var = [[10, 10], [10, 10], [10, 10]]
        self.Pi = [1 / self.n_clusters] * self.n_clusters
        self.W = np.ones((self.n_points, self.n_clusters)) / self.n_clusters

        self.loglh = []


    # 更新W

    # 更新pi

    # 更新Mu

    # 更新Var
    # 更新W
    def update_W(self):

        pdfs = np.zeros(((self.n_points, self.n_clusters)))
        for i in range(self.n_clusters):
            pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf(self.data, self.Mu[i], np.diag(self.Var[i]))
        self.W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        return self.W

    # 更新pi
    def update_Pi(self):
        self.Pi = self.W.sum(axis=0) / self.W.sum()
        return self.Pi

    def update_Mu(self):
        self.Mu = np.zeros((self.n_clusters, 2))
        for i in range(self.n_clusters):
            self.Mu[i] = np.average(self.data, axis=0, weights=self.W[:, i])

    # 更新Var
    def update_Var(self):
        self.Var = np.zeros((self.n_clusters, 2))
        for i in range(self.n_clusters):
            self.Var[i] = np.average((self.data - self.Mu[i]) ** 2, axis=0, weights=self.W[:, i])

    def logLH(self):
        pdfs = np.zeros(((self.n_points, self.n_clusters)))
        for i in range(self.n_clusters):
            pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf(self.data, self.Mu[i], np.diag(self.Var[i]))
        return np.mean(np.log(pdfs.sum(axis=1)))

    # 屏蔽结束

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        self.initialize(data)
        n_loop = 0
        self.loglh.append(self.logLH())
        while n_loop < self.max_iter:
            self.update_W()
            self.update_Pi()
            self.update_Mu()
            self.update_Var()
            self.loglh.append(self.logLH())
            # print(self.loglh)
            if abs(self.loglh[-1] - self.loglh[-2]) < 1e-9:
                break
            n_loop += 1

        # 屏蔽结束

    def predict(self, data):
        # 屏蔽开始
        result = []

        pdfs = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf(data, self.Mu[i], np.diag(self.Var[i]))
        W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        # print(W)
        result = np.argmax(W, axis=1)


        return result
        # 屏蔽结束


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    # print(cat)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5, c=cat)
    plt.show()
    # kmeans = KMeans.K_Means(n_clusters=3)
    # kmeans.fit(X)
    # cat = kmeans.predict(X)
    # plt.figure(figsize=(10, 8))
    # plt.axis([-10, 15, -5, 15])
    # plt.scatter(X[:, 0], X[:, 1], s=5, c=cat)
    # plt.show()
    # 初始化
