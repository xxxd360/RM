import numpy as np
from urdf_parser_py.urdf import URDF
from scipy.optimize import least_squares
from .transforms import create_homogeneous_matrix, create_joint_transform, matrix_to_pos_rpy


class RobotKinematics:
    def __init__(self, urdf_path):
        self.robot = URDF.from_xml_file(urdf_path)
        self.joints_map = {j.name: j for j in self.robot.joints}

        # 筛选有效的可动关节
        valid_movable_joints = []
        for j in self.robot.joints:
            if j.type not in ['revolute', 'prismatic']:
                continue

            # ✅ 关键修复：检查轴向量是否有效
            axis_valid = False
            if j.axis is not None:
                norm = np.linalg.norm(j.axis)
                if norm > 1e-6:  # 如果长度大于极小值，视为有效
                    axis_valid = True

            if axis_valid:
                valid_movable_joints.append(j)
            else:
                print(f"⚠️ 警告：跳过无效关节 '{j.name}' (轴向量: {j.axis})")

        # 只保留有效的关节用于运动学计算
        self.movable_joints = valid_movable_joints
        self.joint_names = [j.name for j in self.movable_joints]

    def get_link_transform(self, link_name, q_dict):
        """
        递归计算从 base_link 到指定 link 的变换矩阵
        :param link_name: 目标连杆名称
        :param q_dict: {joint_name: value} 当前关节状态
        """
        # 基础情况：如果是基座，返回单位阵
        if link_name == 'base_link' or link_name == 'world':
            return np.eye(4)

        # 查找指向该 link 的关节
        parent_joint = None
        for j in self.robot.joints:
            if j.child == link_name:
                parent_joint = j
                break

        if parent_joint is None:
            # 可能是浮动的 link 或者名字错了，尝试找父 link 的变换
            # 在标准 URDF 树中，每个 link (除 root) 都有一个 parent joint
            raise ValueError(f"无法找到指向 link '{link_name}' 的关节")

        # 1. 获取父连杆的变换
        T_parent = self.get_link_transform(parent_joint.parent, q_dict)

        # 2. 获取关节原点偏移 (Parent Link -> Joint Origin)
        T_origin = np.eye(4)
        if parent_joint.origin:
            T_origin = create_homogeneous_matrix(parent_joint.origin.position, parent_joint.origin.rpy)

        # 3. 获取关节运动变换 (Joint Origin -> Child Link frame)
        q_val = q_dict.get(parent_joint.name, 0.0)
        axis = parent_joint.axis if parent_joint.axis else [0, 0, 1]
        T_motion = create_joint_transform(parent_joint.type, axis, q_val)

        # 链式法则：T_total = T_parent * T_origin * T_motion
        return T_parent @ T_origin @ T_motion

    def forward_kinematics(self, q_dict, end_link_name="hand_base_link"):
        """
        正运动学：给定关节角，计算末端位姿
        :param q_dict: {joint_name: value}
        :param end_link_name: 末端连杆名称 (根据你的 URDF 修改，例如 'j6_Link' 或 'hand_base_link')
        :return: 4x4 齐次变换矩阵
        """
        try:
            return self.get_link_transform(end_link_name, q_dict)
        except Exception as e:
            print(f"FK 计算错误：{e}")
            return None

    def inverse_kinematics(self, T_target, q_init=None, max_iter=100):
        """
        逆运动学：数值法求解
        :param T_target: 4x4 目标位姿矩阵
        :param q_init: 初始猜测值 (列表)
        :return: (q_solution, success, error_norm)
        """
        n = len(self.movable_joints)

        # 设置初始值和边界
        if q_init is None:
            q_init = np.zeros(n)

        bounds_low = []
        bounds_high = []
        for j in self.movable_joints:
            if j.limit:
                bounds_low.append(j.limit.lower)
                bounds_high.append(j.limit.upper)
            else:
                # 默认范围
                bounds_low.append(-np.pi if j.type == 'revolute' else -1.0)
                bounds_high.append(np.pi if j.type == 'revolute' else 1.0)

        def objective_function(q_vec):
            # 将向量转为字典
            current_q = {name: val for name, val in zip(self.joint_names, q_vec)}

            # 计算当前 FK
            T_current = self.forward_kinematics(current_q)
            if T_current is None:
                return np.ones(6) * 1e5

            # 计算误差：位置误差 + 姿态误差
            pos_err = T_current[:3, 3] - T_target[:3, 3]

            # 姿态误差 (使用旋转矩阵对数映射)
            R_curr = T_current[:3, :3]
            R_tar = T_target[:3, :3]
            R_err_mat = R_tar.T @ R_curr
            # 将旋转矩阵转为轴角向量 (3维)
            from scipy.spatial.transform import Rotation as Rot
            rot_err = Rot.from_matrix(R_err_mat).as_rotvec()

            return np.hstack([pos_err, rot_err])

        # 优化求解
        result = least_squares(
            objective_function,
            q_init,
            bounds=(bounds_low, bounds_high),
            max_nfev=max_iter,
            ftol=1e-9,
            xtol=1e-9
        )

        return result.x, result.success, np.linalg.norm(result.fun)


# 辅助函数：打印矩阵信息
def print_pose(title, T):
    if T is None: return
    pos, rpy = matrix_to_pos_rpy(T)
    print(f"\n{title}:")
    print(f"  位置 (XYZ): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print(f"  姿态 (RPY): [{rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}] rad")