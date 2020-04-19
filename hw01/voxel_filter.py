# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pandas import DataFrame
from pyntcloud import PyntCloud
import math

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    point_cloud = np.array(point_cloud, dtype=np.float64)
    # print(point_cloud)
    x_array = point_cloud[:, 0]
    y_array = point_cloud[:, 1]
    z_array = point_cloud[:, 2]
    x_max, x_min = np.max(x_array), np.min(x_array)
    y_max, y_min = np.max(y_array), np.min(y_array)
    z_max, z_min = np.max(z_array), np.min(z_array)

    D_x = math.ceil((x_max - x_min) / leaf_size)
    D_y = math.ceil((y_max - y_min) / leaf_size)
    D_z = math.ceil((z_max - z_min) / leaf_size)



    h_array = np.array([], dtype=np.float64)
    for point in point_cloud:
        point = np.array(point, dtype=np.float64)
        h_x = np.floor((point[0] - x_min) / leaf_size)
        h_y = np.floor((point[1] - y_min) / leaf_size)
        h_z = np.floor((point[2] - z_min) / leaf_size)
        h = h_x + h_y * D_x + h_z * D_x * D_y
        h_array = np.append(h_array, h)

    sort = h_array.argsort()
    # print(h_array)
    # print(sort)
    point_cloud = point_cloud[sort]
    # print(point_cloud[0:7, :])
    # print(h_array[sort])
    # print(len(point_cloud))


    voxel = []
    for i in range(len(point_cloud)):
        # print(i)
        if i < len(point_cloud) - 1 and h_array[sort][i] == h_array[sort][i+1]:
            voxel.append(i)
        else:
            voxel.append(i)
            # random
            # rand_array = np.arange(point_cloud[i-len(voxel)+1:i+1, :].shape[0])
            # np.random.shuffle(rand_array)
            # choice_point = point_cloud[i-len(voxel)+1:i+1, :][rand_array[0:1]]

            # centroid
            choice_point = np.mean(point_cloud[i-len(voxel)+1:i+1, :], axis=0)

            point_cloud[i-len(voxel)+1:i+1, :] = choice_point
            voxel = []

    filtered_points = point_cloud
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)



    """txt格式点云
    """
    points = np.genfromtxt("/home/tfwang/PointCloud/homework/modelnet40_normal_resampled/glass_box/glass_box_0072.txt", delimiter=",")
    points = DataFrame(points[:, 0:3])
    points.columns = ['x', 'y', 'z']
    point_cloud_pynt = PyntCloud(points)


    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 0.08)
    # filtered_cloud = point_cloud_o3d.voxel_down_sample(voxel_size=0.05)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])
    # o3d.visualization.draw_geometries([filtered_cloud])

if __name__ == '__main__':
    main()
