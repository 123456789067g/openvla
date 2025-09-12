# run_vla_and_move.py
from typing import List, Tuple
import time, math
import Robot as fr   # 你的 robot.py 模块，别名 fr，避免名字冲突

# ========== 初始化连接 ==========
def connect_and_init_robot(ip: str) -> "fr.RPC":
    r = fr.RPC(ip)  # 连接
    time.sleep(2)

    # 配置夹爪
    ret = r.SetGripperConfig(4, 0)
    print("SetGripperConfig ->", ret)
    time.sleep(1)

    cfg = r.GetGripperConfig()
    print("GetGripperConfig ->", cfg)

    # 按你示例，先后两次激活
    err = r.ActGripper(1, 0)
    print("ActGripper(1,0) ->", err)
    time.sleep(1)
    err = r.ActGripper(1, 1)
    print("ActGripper(1,1) ->", err)
    time.sleep(2)
    return r

# ========== 缩放动作 ==========
def scale_pose(action: List[float],
               scale_xyz=1.0, offset_xyz=(0.0,0.0,0.0),
               scale_rpy=1.0) -> Tuple[List[float], int]:
    x,y,z,rx,ry,rz,g = action
    x = x*scale_xyz + offset_xyz[0]
    y = y*scale_xyz + offset_xyz[1]
    z = z*scale_xyz + offset_xyz[2]
    rx,ry,rz = rx*scale_rpy, ry*scale_rpy, rz*scale_rpy
    grip = 1 if g > 0.5 else 0
    return [float(x),float(y),float(z),float(rx),float(ry),float(rz)], grip

# ========== 执行运动 ==========
def execute_cartesian_or_joint(r: "fr.RPC",
                               pose6: List[float],
                               joint_vel_deg_s=32.0,
                               linear_vel=200.0) -> None:
    """优先尝试笛卡尔直线；不行再走逆解→关节运动"""
    try:
        if hasattr(r, "MoveL"):
            print("MoveL with pose:", pose6)
            r.MoveL(pose6, 0, 0, vel=linear_vel)
            return
    except Exception as e:
        print("MoveL failed ->", e)

    if hasattr(r, "KineInverse"):
        joints = r.KineInverse(pose6)
        print("IK joints:", joints)
        r.MoveJ(joints, 0, 0, vel=joint_vel_deg_s)
    else:
        print("没有可用的 KineInverse，请改为已知关节角或使用你们的逆解接口。")

# ========== 控制夹爪 ==========
def control_gripper(r: "fr.RPC",
                    open_: bool,
                    open_args=(100,48,46),
                    close_args=(0,50,0),
                    current=30000):
    pos, spd, frc = open_args if open_ else close_args
    err = r.MoveGripper(1, pos, spd, frc, current, 0,0,0,0,0)
    print("MoveGripper ->", err)

# ========== 打印命令字符串（调试用） ==========
def translate_action_to_strings(action: List[float], point_id=1, speed=10) -> list:
    x,y,z,rx,ry,rz,g = action
    return [
        f"CARTPoint({point_id},{x},{y},{z},{rx},{ry},{rz})",
        f"MoveL(CART{point_id},{speed})",
        "MoveGripper(1,100)" if g>0.5 else "MoveGripper(1,0)"
    ]

# ========== 主程序 ==========
if __name__ == "__main__":
    ip = "192.168.58.2"
    r = connect_and_init_robot(ip)

    # —— 替换为你从 OpenVLA 得到的动作 ——
    vla_action = [-0.01877375, 0.0002234, -0.00076089,
                  -0.03015281, -0.01283816, 0.01605127, 0.0]

    # 打印调试命令
    print("FR script-like commands:")
    for s in translate_action_to_strings(vla_action):
        print("  ", s)

    # 转换坐标（如 VLA 输出是米/弧度，请改成 scale_xyz=1000.0, scale_rpy=180.0/math.pi）
    pose6, grip = scale_pose(vla_action, scale_xyz=1.0, scale_rpy=1.0)

    # 执行动作
    execute_cartesian_or_joint(r, pose6, joint_vel_deg_s=32.0, linear_vel=200.0)

    # 控制夹爪
    control_gripper(r, open_=(grip==1))
