import numpy as np
from scipy.spatial.transform import Rotation as Rot


def create_homogeneous_matrix(pos, rpy):
    """
    创建齐次变换矩阵
    :param pos: [x, y, z]
    :param rpy: [roll, pitch, yaw]
    """
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = Rot.from_euler('xyz', rpy).as_matrix()
    return T


def create_joint_transform(joint_type, axis, q):
    """
    创建关节运动带来的局部变换矩阵
    ⚠️ 已加入强防御：如果 axis 是零向量，直接返回单位阵，绝不计算除法
    """
    T = np.eye(4)

    # 1. 基础检查：axis 是否为 None
    if axis is None:
        # print("⚠️ 警告：关节轴为 None，视为固定关节。")
        return T

    # 2. 转换为 numpy 数组
    axis = np.array(axis, dtype=np.float64)

    # 3. 计算长度
    norm = np.linalg.norm(axis)

    # 4. 【关键修复】如果长度接近 0，直接返回单位阵，跳过后续所有计算
    if norm < 1e-9:
        # 这里不再打印警告以免刷屏，因为初始化时已经报过错了
        # 但为了调试，你可以取消下面这行的注释
        # print(f"⚠️ 检测到零长度轴 {axis}，已安全跳过。")
        return T

    # 5. 只有当 norm > 0 时，才进行归一化 (这里绝对不会报 divided by zero 了)
    axis = axis / norm

    if joint_type == 'revolute':
        x, y, z = axis
        c, s = np.cos(q), np.sin(q)

        # 罗德里格斯旋转公式
        K = np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])
        R_mat = np.eye(3) + s * K + (1 - c) * (K @ K)
        T[:3, :3] = R_mat

    elif joint_type == 'prismatic':
        T[:3, 3] = axis * q

    return T


def matrix_to_pos_rpy(T):
    """
    从齐次变换矩阵提取位置和欧拉角
    """
    pos = T[:3, 3]
    rpy = Rot.from_matrix(T[:3, :3]).as_euler('xyz')
    return pos, rpy