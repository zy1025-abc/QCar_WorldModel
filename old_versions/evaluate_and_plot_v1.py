import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    s_curr: 当前绝对坐标 [pos_x, pos_y, pos_z, rot_0(x), rot_1(y), rot_2(z), rot_3(w)]
    delta:  模型预测/真实的局部变化 [dx_ego, dy_ego, dz_world, droll, dpitch, dyaw, 0]
    返回:   推演出的下一帧绝对坐标 [pos_x, pos_y, pos_z, x, y, z, w]
    """
    pos_x_c, pos_y_c, pos_z_c, x_c, y_c, z_c, w_c = s_curr
    dx, dy, dz, droll, dpitch, dyaw, _ = delta

    # 1. 解析当前真实朝向 (斗转星移：Y与Z互换适配OptiTrack)
    roll_c, pitch_c, yaw_c = quat_to_euler(w_c, x_c, z_c, y_c)

    # 2. 计算绝对位移
    pos_x_n = pos_x_c + dx * math.cos(yaw_c) - dy * math.sin(yaw_c)
    pos_y_n = pos_y_c + dx * math.sin(yaw_c) + dy * math.cos(yaw_c)
    pos_z_n = pos_z_c + dz

    # 3. 计算绝对姿态
    roll_n = roll_c + droll
    pitch_n = pitch_c + dpitch
    yaw_n = yaw_c + dyaw

    # 4. 转回四元数 (再次斗转星移：把Z轴和Y轴的坑位塞回去)
    w_n, x_n, z_n, y_n = euler_to_quat(roll_n, pitch_n, yaw_n)

    return np.array([pos_x_n, pos_y_n, pos_z_n, x_n, y_n, z_n, w_n])


# ==========================================
# 4. 绘图与评估主函数
# ==========================================
def evaluate_and_plot(prefix):
    print(f"\n📊 开始生成 [{prefix.upper()}] 数据集的评估报告...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备路径
    model_path = f'models_saved/{prefix}_world_model.pth'
    norm_path = f'models_saved/{prefix}_normalization.pt'
    test_data_path = f'results_evaluation/{prefix}_test_data.pt'

    output_dir = f'results_evaluation/{prefix}_evaluation'
    os.makedirs(output_dir, exist_ok=True)

    # 2. 加载数据、翻译官和模型
    test_data = torch.load(test_data_path, map_location=device, weights_only=True)
    X_test = test_data['X_test'].to(device)
    Y_test = test_data['Y_test'].cpu().numpy()  # 真实的局部变化
    S_curr = test_data['S_curr'].cpu().numpy()  # 真实的当前绝对坐标

    norm_stats = torch.load(norm_path, map_location=device, weights_only=True)
    x_mean, x_std = norm_stats['x_mean'].to(device), norm_stats['x_std'].to(device)
    y_mean, y_std = norm_stats['y_mean'].cpu().numpy(), norm_stats['y_std'].cpu().numpy()

    model = QCarWorldModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. 批量推理：让大脑吐出预测的局部变化
    with torch.no_grad():
        X_test_norm = (X_test - x_mean) / x_std
        Y_pred_norm = model(X_test_norm).cpu().numpy()
        Y_pred = Y_pred_norm * y_std + y_mean

    # 4. 运动学推演：将局部偏差推演成绝对轨迹
    num_samples = len(S_curr)
    true_abs_traj = np.zeros((num_samples, 7))
    pred_abs_traj = np.zeros((num_samples, 7))

    for i in range(num_samples):
        true_abs_traj[i] = apply_kinematics(S_curr[i], Y_test[i])
        pred_abs_traj[i] = apply_kinematics(S_curr[i], Y_pred[i])

    # 5. 生成报告与完美图表
    state_names = ['pos_x', 'pos_y', 'pos_z', 'rot_0_x', 'rot_1_y', 'rot_2_z', 'rot_3_w']
    time_steps = np.arange(num_samples)

    for i, name in enumerate(state_names):
        true_vals = true_abs_traj[:, i]
        pred_vals = pred_abs_traj[:, i]

        # 计算极其精准的绝对误差
        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
        mae = np.mean(np.abs(true_vals - pred_vals))

        # 存为 CSV 数据表格
        df = pd.DataFrame({
            'Time Step': time_steps,
            f'True_{name}': true_vals,
            f'Predicted_{name}': pred_vals,
            'Absolute_Error': np.abs(true_vals - pred_vals)
        })
        df.to_csv(f"{output_dir}/{prefix}_{name}_error.csv", index=False)

        # 绘制最高质量的蓝橙对比图
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, true_vals, label='Ground Truth (Blue)', color='blue', linewidth=2)
        plt.plot(time_steps, pred_vals, label='AI Predicted (Orange)', color='darkorange', linestyle='--', linewidth=2)

        plt.title(f"{prefix.capitalize()} Trajectory: {name}\nRMSE: {rmse:.4f} | MAE: {mae:.4f}")
        plt.xlabel("Time Step (Test Set)")
        plt.ylabel(f"Absolute {name} Value")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()

        plt.savefig(f"{output_dir}/{prefix}_{name}_plot.png", dpi=300)
        plt.close()

    print(f"✅ [{prefix.upper()}] 评估完成！7 张高清对比图与 CSV 数据已保存至: {output_dir}/")


if __name__ == "__main__":
    # 先画 Forward 的 7 张图
    evaluate_and_plot('forward')

    # 再画 Backward 的 7 张图
    evaluate_and_plot('backward')

    print("\n🎉 大功告成！全部 14 张高保真对比图已生成！请打开文件夹查阅。")