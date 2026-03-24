import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


# ================= 1. 定义模型架构 =================
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


# ================= 2. 评估与可视化大盘生成 =================
def evaluate_and_plot_dashboard(direction="forward"):
    print(f"\n📊 === Generating {direction.upper()} Model Dashboard ===")

    model_path = f'./models_saved/{direction}_world_model.pth'
    norm_path = f'./models_saved/{direction}_normalization.pt'
    test_data_path = f'./results_evaluation/{direction}_test_data.pt'
    output_dir = f'./results_evaluation/{direction}_evaluation'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path) or not os.path.exists(test_data_path):
        print(f"⚠️ Data not found, please check path: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QCarWorldModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    norm_stats = torch.load(norm_path, map_location=device, weights_only=True)
    x_mean, x_std = norm_stats['x_mean'].to(device), norm_stats['x_std'].to(device)
    y_mean, y_std = norm_stats['y_mean'].to(device), norm_stats['y_std'].to(device)

    test_data = torch.load(test_data_path, map_location=device, weights_only=True)
    X_test = test_data['X_test'].to(device)
    S_curr = test_data['S_curr'].to(device)
    S_next_real = test_data['S_next_real'].to(device)
    True_delta = test_data['Y_test_real_delta'].to(device)

    throttle_real = X_test[:, -1, 7].cpu().numpy()
    steering_real = X_test[:, -1, 8].cpu().numpy()

    with torch.no_grad():
        X_test_norm = (X_test - x_mean) / x_std
        pred_delta_norm = model(X_test_norm)
        Pred_delta = pred_delta_norm * y_std + y_mean

    S_next_pred = S_curr + Pred_delta

    # ================= 🌟 核心修改：自定义画图区间 =================
    start_idx = 500  # <--- 想要从第几帧开始看，就修改这个数字！(比如 1000, 2000, 5000)
    plot_length = 500  # <--- 想要画多长的图 (500 帧 = 20 秒)

    # 越界保护：如果你输入的起始点比测试集总长度还大，就自动重置为 0
    if start_idx >= len(S_next_real):
        print(f"⚠️ 起始点 {start_idx} 超出数据总长度 {len(S_next_real)}，已自动重置为 0")
        start_idx = 0

    end_idx = min(start_idx + plot_length, len(S_next_real))
    time_steps = np.arange(start_idx + 1, end_idx + 1)
    # ===============================================================

    # --- 📏 XYZ 统一坐标轴比例尺 ---
    # 注意这里的数据切片从 [:plot_limit] 全部改成了 [start_idx:end_idx]
    pos_real_limit = S_next_real[start_idx:end_idx, 0:3].cpu().numpy()
    pos_pred_limit = S_next_pred[start_idx:end_idx, 0:3].cpu().numpy()
    global_min_pos = min(pos_real_limit.min(), pos_pred_limit.min())
    global_max_pos = max(pos_real_limit.max(), pos_pred_limit.max())
    margin = (global_max_pos - global_min_pos) * 0.05
    if margin == 0: margin = 1.0
    y_min_pos, y_max_pos = global_min_pos - margin, global_max_pos + margin

    # ================= 3. 生成全景大表 =================
    print(f"📝 Saving Master CSV Evaluation Data (Frames {start_idx} to {end_idx})...")
    csv_data = {
        'Time_Step': np.arange(1, len(S_next_real) + 1),
        'dt_seconds': 0.04,
        'Throttle_Control': throttle_real,
        'Steering_Control': steering_real
    }

    state_names = ['pos_x', 'pos_y', 'pos_z', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    for i, name in enumerate(state_names):
        csv_data[f'True_Abs_{name}'] = S_next_real[:, i].cpu().numpy()
        csv_data[f'Pred_Abs_{name}'] = S_next_pred[:, i].cpu().numpy()
        csv_data[f'True_Delta_{name}'] = True_delta[:, i].cpu().numpy()
        csv_data[f'Pred_Delta_{name}'] = Pred_delta[:, i].cpu().numpy()
        csv_data[f'Abs_Error_{name}'] = np.abs(csv_data[f'True_Abs_{name}'] - csv_data[f'Pred_Abs_{name}'])

    df_master = pd.DataFrame(csv_data)
    csv_filename = os.path.join(output_dir, f'{direction}_MASTER_evaluation_data.csv')
    df_master.to_csv(csv_filename, index=False)

    # ================= 4. 绘制 9D 联合监控大屏 =================
    print(f"📈 Plotting 9D Dashboard (Frames {start_idx} to {end_idx})...")
    fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(14, 28), sharex=True)
    fig.suptitle(f'{direction.upper()} Model: 9D Dashboard (Frames {start_idx}-{end_idx}, v=1)', fontsize=20, y=0.92)

    titles = [
        'pos_x (Horizontal Right)',
        'pos_y (Vertical Up)',
        'pos_z (Forward)',
        'rot_0 (Quaternion x)', 'rot_1 (Quaternion y)',
        'rot_2 (Quaternion z)', 'rot_3 (Quaternion w)'
    ]

    for i, ax in enumerate(axes[:7]):
        # 注意：这里也全部换成了 [start_idx:end_idx]
        true_vals = S_next_real[start_idx:end_idx, i].cpu().numpy()
        pred_vals = S_next_pred[start_idx:end_idx, i].cpu().numpy()

        ax.plot(time_steps, true_vals, label='True State', color='blue', linewidth=2)
        ax.plot(time_steps, pred_vals, label='Predicted State', color='orange', linestyle='--', linewidth=2)

        ax.set_title(titles[i], loc='left', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')

        if i < 3:
            ax.set_ylim(y_min_pos, y_max_pos)
        else:
            ax.set_ylim(-1.1, 1.1)

    # --- 📏 动态计算控制参数比例尺，放大细节 ---
    throttle_plot = throttle_real[start_idx:end_idx]
    steer_plot = steering_real[start_idx:end_idx]

    t_margin = (throttle_plot.max() - throttle_plot.min()) * 0.1
    if t_margin == 0: t_margin = 0.1

    s_margin = (steer_plot.max() - steer_plot.min()) * 0.1
    if s_margin == 0: s_margin = 0.1

    axes[7].plot(time_steps, throttle_plot, color='green', linewidth=2, label='Throttle Input')
    axes[7].set_title('Control: Throttle', loc='left', fontsize=12, fontweight='bold')
    axes[7].set_ylim(throttle_plot.min() - t_margin, throttle_plot.max() + t_margin)
    axes[7].grid(True, linestyle=':', alpha=0.7)
    axes[7].legend(loc='upper right')

    axes[8].plot(time_steps, steer_plot, color='purple', linewidth=2, label='Steering Input')
    axes[8].set_title('Control: Steering', loc='left', fontsize=12, fontweight='bold')
    axes[8].set_ylim(steer_plot.min() - s_margin, steer_plot.max() + s_margin)
    axes[8].grid(True, linestyle=':', alpha=0.7)
    axes[8].legend(loc='upper right')
    axes[8].set_xlabel('Time Step (1 Step = 0.04s)', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    dashboard_filename = os.path.join(output_dir, f'{direction}_9D_Dashboard.png')
    plt.savefig(dashboard_filename, dpi=200)
    plt.close()

    print(f"✅ Finished! 9D Dashboard saved to: {output_dir}")


if __name__ == "__main__":
    evaluate_and_plot_dashboard(direction="forward")
    evaluate_and_plot_dashboard(direction="backward")
    print("\n🎉 Dashboard generated successfully!")