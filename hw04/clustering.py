# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt

from sklearn import neighbors


from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(data)
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.5,
    #                                          ransac_n=3,
    #                                          num_iterations=250)
    # [a, b, c, d] = plane_model
    # print(f"Plane model: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    # inlier_cloud = pcd.select_down_sample(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    #
    # outlier_cloud = pcd.select_down_sample(inliers, invert=True)

    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    #
    # voxel_down_pcd = outlier_cloud.voxel_down_sample(voxel_size=0.16)
    #
    # o3d.visualization.draw_geometries([inlier_cloud, voxel_down_pcd])

    # segmengted_cloud = np.asarray(outlier_cloud.points)
    # inlier_cloud = np.asarray(inlier_cloud.points)


    # ax + by +cz + d = 0
    distance_threshold = 0.5  # 0.4
    ransac_n = 3
    num_iterations = 50
    inlier_cloud = np.empty(shape=[0, data.shape[1]])
    # print(data.shape)
    for _ in range(num_iterations):
        # step 1
        # np.random.seed(0)
        p = np.empty(shape=[0, data.shape[1]])
        for i in np.random.choice(data.shape[0], ransac_n):
            p = np.append(p, [data[i]], axis=0)

        # print(p)
        # step 2
        a = (p[1][1] - p[0][1]) * (p[2][2] - p[0][2]) - (p[1][2] - p[0][2]) * (p[2][1] - p[0][1])
        b = (p[1][2] - p[0][2]) * (p[2][0] - p[0][0]) - (p[1][0] - p[0][0]) * (p[2][2] - p[0][2])
        c = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0])
        d = 0 - (a * p[0][0] + b * p[0][1] + c * p[0][2])

        #step 3
        inlier = np.empty(shape=[0, data.shape[1]])
        # segmengted_cloud_temp = data
        inlier_idx = np.empty(shape=[0, 1], dtype=int)
        for idx in range(data.shape[0]):
            p = data[idx, :]
            point_distance = abs(a*p[0] + b*p[1] + c*p[2] + d) / np.sqrt(a*a + b*b + c*c)
            if point_distance < distance_threshold:
                inlier = np.append(inlier, [data[idx]], axis=0)
                inlier_idx = np.append(inlier_idx, idx)

        if inlier.shape[0] > inlier_cloud.shape[0]:
            inlier_cloud = inlier
            segmengted_cloud = np.delete(data, inlier_idx, axis=0)

        # print(inlier_cloud.shape)
        # print(segmengted_cloud.shape)

    # print(inlier_cloud.shape)
    # print(segmengted_cloud.shape)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(segmengted_cloud)
    # o3d.visualization.draw_geometries([pcd])
    # 屏蔽结束q

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud, inlier_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    clusters_index = []
    # print('segmented data points num:', data.shape[0])
    # dbscan = cluster.DBSCAN(eps=.2)
    # # dbscan.fit(data)
    # clusters_index = dbscan.fit_predict(data)
    # km = K_Means(n_clusters=30)
    # km.fit(data)
    # clusters_index = km.predict(data)

    # DBSCAN
    distance = 0.2
    min_samples = 5
    k = -1

    fil = []
    unvisited = [x for x in range(len(data))]
    clusters_index = [-1 for y in range(len(data))]

    neighbor_points = neighbors.radius_neighbors_graph(data, distance).toarray()
    neighbor_num = np.sum(neighbor_points, axis=0)

    while len(unvisited) > 0:
        point_idx = np.random.choice(unvisited)
        unvisited.remove(point_idx)
        fil.append(point_idx)

        nb_num = neighbor_num[point_idx]
        nb_points = neighbor_points[point_idx]
        neighbor_p = []
        new_neighbor_p = []
        for idx in range(len(nb_points)):
            if nb_points[idx] != 0:
                neighbor_p.append(idx)
        if nb_num < min_samples:
            clusters_index[point_idx] = -1
        else:
            k += 1
            clusters_index[point_idx] = k
            for p in neighbor_p:
                if p not in fil:
                    unvisited.remove(p)
                    fil.append(p)

                    new_nb_points = neighbor_points[p]
                    for new_idx in range(len(new_nb_points)):
                        if new_nb_points[new_idx] != 0:
                            new_neighbor_p.append(new_idx)
                    if len(new_neighbor_p) >= min_samples:
                        for a in new_neighbor_p:
                            if a not in neighbor_p:
                                neighbor_p.append(a)
                    if clusters_index[p] == -1:
                        clusters_index[p] = k
    # 屏蔽结束

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index, inliers_data):
    ax = plt.figure().add_subplot(111, projection='3d')
    colors = np.array(list(islice(cycle(['#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']), int(max(cluster_index) + 2))))
    # print(colors.shape)
    # print(cluster_index.shape)
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    # print(data.shape)
    # print(inliers_data.shape)
    ax.scatter(inliers_data[:, 0], inliers_data[:, 1], inliers_data[:, 2], s=2, color=['#377eb8'])
    plt.axis('off')
    plt.show()

def main():
    root_dir = '/home/tfwang/baidunetdiskdownload/KITTI_object/data_object_velodyne/training'# 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[1:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points, inliers_points = ground_segmentation(data=origin_points)
        cluster_index = clustering(segmented_points)

        plot_clusters(segmented_points, cluster_index, inliers_points)
if __name__ == '__main__':
    main()
