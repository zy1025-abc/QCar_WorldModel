import torch
import torch.nn as nn
import numpy as np
import math


# ==========================================
# 1. 网络类 (保持 9入7出 结构，一行不改)
# ==========================================
class QCarWorldModel(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, num_layers=3, output_dim=7):
        super(QCarWorldModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ==========================================
# 2. 核心数学：四元数与欧拉角的双向转换
# ==========================================
def quat_to_euler(w, x, y, z):
    """将四元数转换为欧拉角 (Roll, Pitch, Yaw)"""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """将欧拉角 (Roll, Pitch, Yaw) 转换回四元数"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return w, x, y, z


# ==========================================
# 3. 运动学推演引擎：将局部偏差还原为绝对坐标
# ==========================================
def apply_kinematics(s_curr, delta):
    """
    这就像游戏引擎里的 Transform.Translate()。
    它把车体坐标系下的微小移动，叠加到世界绝对坐标系上。
    """
    # 解包当前绝对状态 (遵循 OptiTrack 导出的 x, y, z, w 顺序)
    pos_x_c, pos_y_c, pos_z_c, x_c, y_c, z_c, w_c = s_curr

    # 解包 AI 预测的 7 维局部变化量
    dx_ego, dy_ego, dz_world, droll, dpitch, dyaw, _ = delta

    # 1. 解析当前真实朝向 (斗转星移：Y与Z互换，适配 OptiTrack 的 Y轴朝上)
    roll_c, pitch_c, yaw_c = quat_to_euler(w_c, x_c, z_c, y_c)

    # 2. 计算下一帧的世界绝对位移 (结合三角函数)
    # 车头朝向(yaw)决定了 dx_ego(往前开) 会在地图 X轴 和 Y轴 上分别产生多少投影
    pos_x_n = pos_x_c + dx_ego * math.cos(yaw_c) - dy_ego * math.sin(yaw_c)
    pos_y_n = pos_y_c + dx_ego * math.sin(yaw_c) + dy_ego * math.cos(yaw_c)
    pos_z_n = pos_z_c + dz_world  # 高度不受平面朝向影响

    # 3. 计算下一帧的世界绝对欧拉角姿态
    roll_n = roll_c + droll
    pitch_n = pitch_c + dpitch
    yaw_n = yaw_c + dyaw

    # 4. 转回四元数 (再次斗转星移：把Z轴和Y轴的坑位换回去)
    w_n, x_n, z_n, y_n = euler_to_quat(roll_n, pitch_n, yaw_n)

    # 返回推演出的全新 7 维绝对状态
    return np.array([pos_x_n, pos_y_n, pos_z_n, x_n, y_n, z_n, w_n]), yaw_c, yaw_n


# ==========================================
# 4. 交互式沙盒主程序
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎮 启动交互式沙盒... (使用设备: {device})")

    # 默认使用 forward 模型进行测试
    prefix = 'forward'

    # 1. 加载模型与翻译官 (归一化参数)
    model = QCarWorldModel().to(device)
    model.load_state_dict(torch.load(f"models_saved/{prefix}_world_model.pth", map_location=device, weights_only=True))
    model.eval()  # 考试模式，不更新权重

    norm_stats = torch.load(f"models_saved/{prefix}_normalization.pt", map_location=device, weights_only=True)
    x_mean, x_std = norm_stats['x_mean'].to(device), norm_stats['x_std'].to(device)
    y_mean, y_std = norm_stats['y_mean'].cpu().numpy(), norm_stats['y_std'].cpu().numpy()

    # 2. 加载测试卷 (保留了真实的绝对坐标用于推演)
    test_data = torch.load(f"results_evaluation/{prefix}_test_data.pt", map_location=device, weights_only=True)
    X_test = test_data['X_test']  # 形状: [样本数, 20帧, 9特征]
    S_curr = test_data['S_curr'].cpu().numpy()  # 形状: [样本数, 7特征] 当前真实绝对坐标

    print(f"📦 成功加载 {prefix} 测试集，共有 {len(X_test)} 条可用历史轨迹。")

    while True:
        try:
            # --- 第一步：抽取历史场景 ---
            idx_str = input("\n🔍 请输入你想载入的测试集索引 (输入 s 自动帮你找一个'高速狂飙'的帧，输入 q 退出): ")
            if idx_str.lower() == 'q': break

            if idx_str.lower() == 's':
                print("🔎 正在全盘扫描高速飞驰的场景...")
                import random
                search_indices = list(range(len(X_test)))
                random.shuffle(search_indices)  # 每次随机找，保证不重样
                found = False
                for i in search_indices:
                    # 寻找历史最后一帧真实油门大于 0.1 的“狂飙”时刻
                    if X_test[i, -1, 7].item() > 0.1:
                        idx = i
                        found = True
                        print(f"🎯 雷达锁定！为你找到索引 [{idx}]，此时车子正处于高速状态！")
                        break
                if not found:
                    print("⚠️ 测试集里居然没找到高速帧？退回手动输入。")
                    continue
            else:
                idx = int(idx_str)

            # 提取 20 帧的历史序列，准备喂给 LSTM
            history_seq = X_test[idx].clone().unsqueeze(0).to(device)  # [1, 20, 9]
            current_abs_state = S_curr[idx]  # 当前真实的 7 维绝对坐标

            # 读取原始记录中的动作指令 (特征索引 7是油门，8是转向)
            real_throttle = history_seq[0, -1, 7].item()
            real_steer = history_seq[0, -1, 8].item()

            print(f"\n--- 🎬 场景 [{idx}] 已载入 ---")
            print(f"📍 当前绝对坐标 X: {current_abs_state[0]:.4f}, Y: {current_abs_state[1]:.4f}")
            print(f"🚗 当时人类驾驶员的真实指令 -> 油门: {real_throttle:.4f}, 转向: {real_steer:.4f}")

            # --- 第二步：用户下达驾驶指令 ---
            custom_input = input(
                "👉 请下达新的驾驶指令，格式为 '油门,转向' (如 '0.07,-0.5' 向左打死，直接回车保持真实指令): ")

            if custom_input.strip():
                new_throttle, new_steer = map(float, custom_input.split(','))
                # 强行篡改历史序列最后一帧的动作指令
                history_seq[0, -1, 7] = new_throttle
                history_seq[0, -1, 8] = new_steer
                print(f"🎮 已接管车辆控制权！强行输入 -> 油门: {new_throttle}, 转向: {new_steer}")
            else:
                print("🔒 保持自动驾驶/真实指令，正在推演...")

            # --- 第三步：AI 模型推理 (预测车体相对位移) ---
            with torch.no_grad():
                # 数据翻译进大脑
                input_norm = (history_seq - x_mean) / x_std
                # 大脑吐出相对变化量
                pred_delta_norm = model(input_norm).cpu().numpy()
                # 翻译回物理数值 [dx_ego, dy_ego, dz, droll, dpitch, dyaw, _]
                pred_delta = pred_delta_norm * y_std + y_mean

            delta_ego = pred_delta[0]  # 取出 1D 数组

            # --- 第四步：物理引擎推演绝对轨迹 ---
            # 魔法时刻：把相对偏差转化为世界地图上的新位置
            next_state_abs, yaw_curr, yaw_next = apply_kinematics(current_abs_state, delta_ego)

            # --- 第五步：生成战报 ---
            print("\n✅ AI 物理推演结果 (0.04秒后的状态):")

            print(f"[相对变化(车体视角)]: ")
            print(f"  🏎️ 纵向移动 (dx_ego): {delta_ego[0]:.6f} 米")
            print(f"  🦀 横向滑移 (dy_ego): {delta_ego[1]:.6f} 米")
            print(f"  🧭 车头转向 (dyaw):   {math.degrees(delta_ego[5]):.4f} 度")

            print(f"[绝对坐标(地图上帝视角)]: ")
            print(
                f"  🌍 X 坐标: {current_abs_state[0]:.6f} -> {next_state_abs[0]:.6f} (差值: {next_state_abs[0] - current_abs_state[0]:.6f} 米)")
            print(
                f"  🌍 Y 坐标: {current_abs_state[1]:.6f} -> {next_state_abs[1]:.6f} (差值: {next_state_abs[1] - current_abs_state[1]:.6f} 米)")
            print(f"  🌍 绝对偏航角: {math.degrees(yaw_curr):.2f}° -> {math.degrees(yaw_next):.2f}°")

            # 启发性点评
            if abs(delta_ego[5]) > 0.05:
                direction = "向左" if delta_ego[5] < 0 else "向右"
                print(f"\n💡 物理点评：模型感知到了剧烈的转向输入，车身成功产生 {direction}侧偏！")
            elif delta_ego[0] > 0.02:
                print(f"\n💡 物理点评：模型感知到强劲的油门输入，车辆正在地图上向前推进！")
            else:
                print(f"\n💡 物理点评：车辆处于怠速或微调状态，符合当前惯性。")

        except Exception as e:
            print(f"❌ 输入有误，请重试。错误信息: {e}")


if __name__ == "__main__":
    main()