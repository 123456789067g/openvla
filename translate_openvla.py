# translate_openvla_to_fr.py

# 假设 OpenVLA 输出的 action 是一个长度为 7 的数组：
# [x, y, z, rx, ry, rz, gripper]
# 其中单位：位置(mm)，角度(deg)，gripper(0=关, 1=开)

def translate_action_to_fr(action, point_id=1, speed=10):
    """
    将 OpenVLA 的动作数组翻译成 FR API 调用字符串
    """
    x, y, z, rx, ry, rz, gripper = action

    # 1. 笛卡尔坐标点
    cart_point = f"CARTPoint({point_id},{x},{y},{z},{rx},{ry},{rz})"

    # 2. 直线运动命令 (MoveL)
    move_cmd = f"MoveL(CART{point_id},{speed})"

    # 3. 夹爪控制
    if gripper > 0.5:
        gripper_cmd = "MoveGripper(1,100)"   # 打开
    else:
        gripper_cmd = "MoveGripper(1,0)"     # 关闭

    return [cart_point, move_cmd, gripper_cmd]


# 测试
if __name__ == "__main__":
    # 一个假设的 OpenVLA 输出
    action = [100, 150, 200, 0, 0, 90, 1]  # x,y,z,rx,ry,rz,gripper
    cmds = translate_action_to_fr(action)
    print("Generated FR commands:")
    for cmd in cmds:
        print("   ", cmd)
