import numpy as np
import scripts.transforms
from scripts.FK_IK import RobotKinematics, print_pose


def main():
    # 1. 初始化机器人 (请确保 fr5v6.urdf 在同一目录或填写绝对路径)
    urdf_file = r"D:\pytorch.pycharm\RM\datasets\fr5_description\fr5_description\urdf\fr5v6.urdf"
    try:
        robot = RobotKinematics(urdf_file)
        print(f"✅ 成功加载机器人：{robot.robot.name}")
        print(f"   可动关节：{robot.joint_names}")
    except Exception as e:
        print(f"❌ 加载失败：{e}")
        return

    # --- 任务 2: 正运动学 (Forward Kinematics) ---
    print("\n" + "=" * 30)
    print("任务 2: 正运动学验证")
    print("=" * 30)

    # 设定一组关节角 (示例：全零配置，或者你可以手动输入一些角度)
    # 注意：键名必须与 URDF 中的 joint name 完全一致
    q_test = {
        # 根据你的 URDF 实际关节名填充，这里假设是 j1, j2...
        # 如果不确定名字，可以查看 robot.joint_names
        # 下面是一个假设的填充，请根据实际情况修改！
        # 'j1': 0.0,
        # 'j2': 0.0,
        # ...
    }

    # 自动填充 0 作为演示 (实际使用时请替换为真实关节名)
    for name in robot.joint_names:
        q_test[name] = 0.0

    # 计算末端位姿 (注意：修改 end_link_name 为你 URDF 中真正的末端连杆名，如 'j6_Link' 或 'hand_base_link')
    # 提示：你可以先打印 robot.robot.links 查看所有连杆名
    end_link = "j5_Link"  # <--- 【重要】请在此处修改为你的末端连杆名称

    # 检查末端连杆是否存在
    link_names = [l.name for l in robot.robot.links]
    if end_link not in link_names:
        print(f"⚠️ 警告：未找到末端连杆 '{end_link}'。可用连杆：{link_names[-5:]}...")
        # 临时使用最后一个连杆
        end_link = link_names[-1]
        print(f"   已自动切换为：{end_link}")

    T_fk = robot.forward_kinematics(q_test, end_link_name=end_link)
    print_pose(f"零位配置下 ({end_link}) 的末端位姿", T_fk)

    # --- 任务 3: 逆运动学 (Inverse Kinematics) ---
    print("\n" + "=" * 30)
    print("任务 3: 逆运动学求解")
    print("=" * 30)

    # 场景 A: 自洽性测试 (用 FK 生成一个目标，再用 IK 解回来)
    print("\n[测试 A] 自洽性验证 (FK -> IK -> FK)")

    # 1. 随机生成一组合法的关节角
    q_random = {}
    for name in robot.joint_names:
        j = robot.joints_map[name]
        if j.limit:
            q_random[name] = np.random.uniform(j.limit.lower, j.limit.upper)
        else:
            q_random[name] = np.random.uniform(-np.pi, np.pi)

    # 2. 计算目标位姿
    T_target = robot.forward_kinematics(q_random, end_link_name=end_link)
    print_pose("随机生成的目标位姿", T_target)

    # 3. 调用 IK 求解 (初始猜测设为全 0)
    q_init_guess = np.zeros(len(robot.joint_names))
    q_sol, success, error = robot.inverse_kinematics(T_target, q_init=q_init_guess)

    print(f"\nIK 求解结果:")
    print(f"  收敛状态：{'✅ 成功' if success else '❌ 失败'}")
    print(f"  最终误差范数：{error:.6e}")

    # 4. 将解映射回字典
    q_sol_dict = {name: val for name, val in zip(robot.joint_names, q_sol)}

    # 5. 验证：将解代入 FK
    T_check = robot.forward_kinematics(q_sol_dict, end_link_name=end_link)
    print_pose("IK 解反算的位姿", T_check)

    # 6. 对比原始关节角
    print("\n关节角对比 (Rad):")
    print(f"{'关节名':<15} | {'原始值':<10} | {'IK 解':<10} | {'差值':<10}")
    print("-" * 55)
    for name in robot.joint_names:
        orig = q_random[name]
        sol = q_sol_dict[name]
        diff = abs(orig - sol)
        # 处理 2pi 周期性差异
        if diff > np.pi:
            diff = 2 * np.pi - diff
        print(f"{name:<15} | {orig:>8.4f}   | {sol:>8.4f}   | {diff:>8.4f}")


if __name__ == "__main__":
    main()