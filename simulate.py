import pybullet as p
import pybullet_data
import time
import numpy as np
import os

# 导入你之前写好的运动学类
from scripts.FK_IK import RobotKinematics


def main():
    # 1. 连接物理引擎 (GUI 模式)
    physicsClient = p.connect(p.GUI)

    # 设置额外搜索路径，方便加载地面等内置模型
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 2. 设置环境
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")  # 加载地面

    # 3. 加载你的机器人
    # ⚠️ 请确保这里的文件名和你实际的文件名一致
    urdf_path = os.path.abspath("fr5v6.urdf")
    if not os.path.exists(urdf_path):
        # 尝试在子目录找
        if os.path.exists("robots/fr5v6.urdf"):
            urdf_path = os.path.abspath("robots/fr5v6.urdf")
        elif os.path.exists("RM/fr5v6.urdf"):  # 根据实际结构调整
            pass

    print(f"🤖 正在加载机器人: {urdf_path}")

    # 关键参数：useFixedBase=True 表示基座固定（不像人形机器人会倒）
    # flags=p.URDF_USE_INERTIA_FROM_FILE 使用URDF中定义的惯性参数
    robotId = p.loadURDF(urdf_path,
                         basePosition=[0, 0, 0],
                         useFixedBase=True,
                         flags=p.URDF_USE_INERTIA_FROM_FILE)

    # 4. 初始化运动学求解器
    # 注意：这里假设你的 RobotKinematics 类已经修复了 axis=0 的 bug
    try:
        robot_solver = RobotKinematics(urdf_path)
        print("✅ 运动学求解器初始化成功")
    except Exception as e:
        print(f"❌ 运动学求解器初始化失败: {e}")
        return

    # 获取 PyBullet 中的关节名称映射
    # PyBullet 会给每个 joint 一个 index，我们需要通过名字找到它
    numJoints = p.getNumJoints(robotId)
    joint_name_to_index = {}
    for i in range(numJoints):
        info = p.getJointInfo(robotId, i)
        name = info[1].decode('utf-8')
        joint_name_to_index[name] = i
        # print(f"Debug: Joint {i} -> {name}")

    print(f"🔗 识别到 {len(joint_name_to_index)} 个 PyBullet 关节")

    # 5. 演示循环
    # 我们让机械臂在几个目标点之间移动
    targets = [
        [0.3, 0.0, 0.4],  # 前方
        [0.0, 0.3, 0.3],  # 右侧
        [-0.2, 0.0, 0.5],  # 后方偏上
        [0.0, 0.0, 0.6]  # 垂直向上
    ]

    # 定义我们要控制的关节（通常是 j1-j6）
    # 注意：必须和 URDF 中的名字完全一致
    controlled_joints = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    control_indices = [joint_name_to_index[name] for name in controlled_joints if name in joint_name_to_index]

    if len(control_indices) != 6:
        print(f"⚠️ 警告：只找到了 {len(control_indices)} 个控制关节，期望 6 个。检查关节名称是否匹配。")
        # 如果名称不匹配，这里可能需要手动调整列表

    print("🎬 开始仿真演示... (按 ESC 或关闭窗口退出)")

    t = 0
    target_idx = 0

    while p.isConnected():
        # 获取当前目标位置
        target_pos = targets[target_idx]
        target_rpy = [np.pi, 0, 0]  # 默认姿态，可根据需要修改

        # 调用你的 IK 求解器
        # q_init_guess 可以设为上一时刻的解，以加快收敛
        q_sol, success, error = robot_solver.inverse_kinematics(
            target_pos=target_pos,
            target_rpy=target_rpy,
            q_init_guess=None  # 第一次可以为 None，后续可以传入 q_sol
        )

        if success:
            print(f"Step {t}: 到达目标 {target_pos}, 误差: {error:.4f}")

            # 🚀 关键步骤：将解算出的角度应用到 PyBullet
            # q_sol 是一个包含所有 movable_joints 角度的数组
            # 我们需要把它映射到 controlled_joints 的顺序

            # 假设 robot_solver.joint_names 的顺序和 controlled_joints 一致
            # 如果不一致，需要根据名字映射
            for i, joint_name in enumerate(robot_solver.joint_names):
                if joint_name in joint_name_to_index:
                    idx = joint_name_to_index[joint_name]
                    angle = q_sol[i]
                    # 设置关节位置 (位置控制模式)
                    p.setJointMotorControl2(robotId, idx, p.POSITION_CONTROL, targetPosition=angle)

            # 切换下一个目标 (每 2 秒切换一次)
            if t % 200 == 0 and t > 0:
                target_idx = (target_idx + 1) % len(targets)
                print(f"🎯 切换新目标: {targets[target_idx]}")
        else:
            print(f"Step {t}: IK 求解失败，保持原位")

        # 步进物理仿真
        p.stepSimulation()
        time.sleep(1. / 240.)  # 限制帧率
        t += 1

    p.disconnect()


if __name__ == "__main__":
    main()