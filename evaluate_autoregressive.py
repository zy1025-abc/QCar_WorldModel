import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil


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


def run_autoregressive_rollout(direction, w, start_idx, history_w=100):
    print(f"\n🚀 === 启动 {direction.upper()} 模型自回归推演 (历史={history_w}, 预测={w}) ===")

    output_base_dir = f'./results_evaluation_w={w}'
    output_dir = os.path.join(output_base_dir, f'{direction}_evaluation')
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_path = f'./models_saved/{direction}_world_model.pth'
    norm_path = f'./models_saved/{direction}_normalization.pt'
    test_data_path = f'./results_evaluation/{direction}_test_data.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QCarWorldModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    norm_stats = torch.load(norm_path, map_location=device, weights_only=True)
    x_mean, x_std = norm_stats['x_mean'].to(device), norm_stats['x_std'].to(device)
    y_mean, y_std = norm_stats['y_mean'].to(device), norm_stats['y_std'].to(device)

    test_data = torch.load(test_data_path, map_location=device, weights_only=True)
    X_test, S_curr, S_next_real = test_data['X_test'].to(device), test_data['S_curr'].to(device), test_data[
        'S_next_real'].to(device)

    # 越界保护
    actual_history = min(history_w, start_idx)
    if start_idx + w + 1 >= len(X_test): w = len(X_test) - start_idx - 2

    # --- 🌟 提取历史“助跑”数据 ---
    past_true_states = S_curr[start_idx - actual_history: start_idx].cpu().numpy()
    past_throttle = X_test[start_idx - actual_history: start_idx, -1, 7].cpu().numpy()
    past_steering = X_test[start_idx - actual_history: start_idx, -1, 8].cpu().numpy()

    # --- 提取未来真实数据 ---
    future_true_states = S_next_real[start_idx: start_idx + w].cpu().numpy()
    future_throttle = X_test[start_idx + 1: start_idx + w + 1, -1, 7].cpu().numpy()
    future_steering = X_test[start_idx + 1: start_idx + w + 1, -1, 8].cpu().numpy()

    print(f"🧠 AI 开始从第 {start_idx} 帧起，闭眼推演未来 {w} 步...")
    current_x = X_test[start_idx].clone().unsqueeze(0)
    current_abs_state = S_curr[start_idx].clone().unsqueeze(0)
    predicted_abs_states = []

    with torch.no_grad():
        for step in range(w):
            x_norm = (current_x - x_mean) / x_std
            pred_delta_real = model(x_norm) * y_std + y_mean
            next_abs_state = current_abs_state + pred_delta_real
            predicted_abs_states.append(next_abs_state.squeeze(0).cpu().numpy())

            next_action = X_test[start_idx + step + 1, -1, 7:9].unsqueeze(0).unsqueeze(0)
            new_frame = torch.cat([next_abs_state.unsqueeze(1), next_action], dim=2)
            current_x = torch.cat([current_x[:, 1:, :], new_frame], dim=1)
            current_abs_state = next_abs_state

    predicted_abs_states = np.array(predicted_abs_states)

    # ================= 画图逻辑升级 =================
    print("📈 正在绘制自回归 9D 大盘...")
    fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(15, 28), sharex=True)
    fig.suptitle(f'{direction.upper()} Model: Autoregressive Rollout (Past={actual_history}, Future={w})', fontsize=20,
                 y=0.92)

    titles = ['pos_x (Horizontal)', 'pos_y (Vertical)', 'pos_z (Forward)', 'rot_0 (x)', 'rot_1 (y)', 'rot_2 (z)',
              'rot_3 (w)']

    # 🌟 定义时间轴 (负数代表过去，0代表现在，正数代表未来)
    time_past = np.arange(-actual_history, 0)
    time_future = np.arange(0, w)

    # 统一 XYZ 比例尺
    all_pos = np.vstack([past_true_states[:, :3], future_true_states[:, :3], predicted_abs_states[:, :3]])
    margin = (all_pos.max() - all_pos.min()) * 0.05 if all_pos.max() != all_pos.min() else 1.0
    y_min_pos, y_max_pos = all_pos.min() - margin, all_pos.max() + margin

    for i, ax in enumerate(axes[:7]):
        # 画历史蓝线 (实线)
        ax.plot(time_past, past_true_states[:, i], color='blue', linewidth=2, label='History (Real)')
        # 画未来真实蓝线 (半透明实线)
        ax.plot(time_future, future_true_states[:, i], color='blue', linewidth=2, alpha=0.4,
                label='Future Ground Truth')
        # 画 AI 预测红线 (虚线)
        ax.plot(time_future, predicted_abs_states[:, i], color='red', linestyle='--', linewidth=2.5,
                label='AI Dream Rollout')

        # 🌟 画一条垂直的“现在”分割线
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5)

        ax.set_title(titles[i], loc='left', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.7)
        if i == 0: ax.legend(loc='upper left')
        if i < 3:
            ax.set_ylim(y_min_pos, y_max_pos)
        else:
            ax.set_ylim(-1.1, 1.1)

    axes[7].plot(time_past, past_throttle, color='green', linewidth=2)
    axes[7].plot(time_future, future_throttle, color='green', linewidth=2, alpha=0.4)
    axes[7].axvline(x=0, color='black', linestyle=':')
    axes[7].set_title('Control: Throttle', loc='left', fontsize=12, fontweight='bold')
    axes[7].grid(True)

    axes[8].plot(time_past, past_steering, color='purple', linewidth=2)
    axes[8].plot(time_future, future_steering, color='purple', linewidth=2, alpha=0.4)
    axes[8].axvline(x=0, color='black', linestyle=':')
    axes[8].set_title('Control: Steering', loc='left', fontsize=12, fontweight='bold')
    axes[8].grid(True)
    axes[8].set_xlabel('Time Steps (0 = Prediction Start Point)', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f'{direction}_9D_AutoRegressive.png'), dpi=200)
    plt.close()
    print(f"✅ 保存成功！请查看: {output_dir}")


if __name__ == "__main__":
    # ================= 操作台 =================
    PREDICT_W = 100  # 未来推演步数 (比如 50, 100, 250)
    START_FRAME = 100 # 起跑点
    HISTORY_W = 0  # 往前多画多少帧作为助跑线 (150帧=6秒)

    run_autoregressive_rollout(direction="forward", w=PREDICT_W, start_idx=START_FRAME, history_w=HISTORY_W)
    run_autoregressive_rollout(direction="backward", w=PREDICT_W, start_idx=START_FRAME, history_w=HISTORY_W)