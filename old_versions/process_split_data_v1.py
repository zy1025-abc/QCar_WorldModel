import os
import pandas as pd
import torch
import numpy as np
import math


# ==========================================
# 工具函数：四元数转欧拉角 (解决四元数直接做差的缺陷)
# ==========================================
def quat_to_euler(w, x, y, z):
    """将四元数转换为欧拉角(弧度)，从而计算真实的物理旋转偏差"""
    # Roll (翻滚角)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # Pitch (俯仰角)
    t2 = +2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # Yaw (航向角)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def process_dataset(parent_dir, output_file, seq_length=20):
    print(f"\n🚀 开始处理数据集: {parent_dir}")

    action_cols = ['throttle', 'steering']
    # 假设你的原始数据四元数顺序是 w, x, y, z (对应 rot_0, 1, 2, 3)
    state_cols = ['pos_x', 'pos_y', 'pos_z', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

    all_x_seqs = []
    all_y_targets = []
    total_files_processed = 0
    dropped_windows = 0

    if not os.path.exists(parent_dir):
        print(f"❌ 找不到文件夹: '{parent_dir}'，请检查目录结构！")
        return False

    # 遍历所有的子文件夹和 CSV
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        if not os.path.isdir(folder_path): continue

        print(f"  📂 正在清洗并切分数据: {folder_name}")

        for file_name in os.listdir(folder_path):
            if not file_name.endswith('.csv'): continue
            file_path = os.path.join(folder_path, file_name)

            try:
                df = pd.read_csv(file_path)
                if not set(action_cols + state_cols).issubset(df.columns):
                    continue
                df = df.dropna(subset=action_cols + state_cols)

                actions_np = df[action_cols].values.astype(np.float32)
                states_np = df[state_cols].values.astype(np.float32)

                # 🌟 修复队友问题 4：控制量单位异常
                # 将异常的方向盘数值强行截断在 [-1.0, 1.0] 范围内，与线上环境对齐
                actions_np[:, 1] = np.clip(actions_np[:, 1], -1.0, 1.0)

                # 🌟 核心架构升级：按文件单独提取滑动窗口，天然杜绝“跨文件拼接”！
                for i in range(len(df) - seq_length):
                    # 取出 21 帧 (20帧历史 + 1帧未来目标)
                    window_states = states_np[i: i + seq_length + 1]
                    window_actions = actions_np[i: i + seq_length]

                    # 🌟 修复队友问题 2：历史窗口脏数据污染
                    # 检查这 21 帧内部，是否有任意两帧的 XY 坐标跳变超过 0.5 米
                    diffs = np.abs(window_states[1:, :2] - window_states[:-1, :2])
                    if np.any(diffs > 0.5):
                        dropped_windows += 1
                        continue  # 只要有瞬间漂移，整个窗口彻底扔掉！

                    # 提取当前帧 (第20帧) 和下一帧 (第21帧)
                    curr_state = window_states[seq_length - 1]
                    next_state = window_states[seq_length]

                    # 1. 按照 OptiTrack 真实的导出顺序 (x, y, z, w) 解包
                    pos_x_c, pos_y_c, pos_z_c, x_c, y_c, z_c, w_c = curr_state
                    pos_x_n, pos_y_n, pos_z_n, x_n, y_n, z_n, w_n = next_state

                    # 2. 斗转星移：把 OptiTrack 的 Y 轴(朝上) 喂给公式里的 Z 轴坑位！
                    roll_c, pitch_c, yaw_c = quat_to_euler(w_c, x_c, z_c, y_c)
                    roll_n, pitch_n, yaw_n = quat_to_euler(w_n, x_n, z_n, y_n)

                    # 1. 计算绝对偏差
                    dx_world = pos_x_n - pos_x_c
                    dy_world = pos_y_n - pos_y_c
                    dz_world = pos_z_n - pos_z_c  # 高度不受车头朝向影响，保持绝对值

                    # 2. 旋转投影：把地图绝对坐标转化为相对于车头的纵向/横向位移
                    dx_ego = dx_world * math.cos(yaw_c) + dy_world * math.sin(yaw_c)
                    dy_ego = -dx_world * math.sin(yaw_c) + dy_world * math.cos(yaw_c)

                    # 3. 姿态偏差：用欧拉角相减，并处理转圈越界问题 (-pi 到 pi)
                    dyaw = yaw_n - yaw_c
                    dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi

                    droll = roll_n - roll_c
                    droll = (droll + math.pi) % (2 * math.pi) - math.pi

                    dpitch = pitch_n - pitch_c
                    dpitch = (dpitch + math.pi) % (2 * math.pi) - math.pi

                    # 4. 组装全新 7维 标签 Label (用 0.0 凑齐第 7 个维度以适配原网络)
                    # [纵向移动, 横向滑移, 高度变化, 翻滚变化, 俯仰变化, 航向转向, 占位符]
                    ego_target = np.array([dx_ego, dy_ego, dz_world, droll, dpitch, dyaw, 0.0], dtype=np.float32)

                    # 5. 组装 9维 历史输入特征
                    joint_input = np.concatenate([window_states[:-1], window_actions], axis=1)

                    all_x_seqs.append(joint_input)
                    all_y_targets.append(ego_target)

                total_files_processed += 1

            except Exception as e:
                print(f"    ❌ 读取 {file_name} 失败: {e}")

    if not all_x_seqs:
        return False

    final_x = np.stack(all_x_seqs)
    final_y = np.stack(all_y_targets)

    print(f"✅ 完成！合并了 {total_files_processed} 个文件。")
    print(f"🛡️ 成功拦截并丢弃了 {dropped_windows} 个包含瞬移尖刺的脏窗口！")
    print(f"📊 最终生成纯净数据帧: {len(final_x)} 条")

    # 注意：我们现在直接保存切好的 3D 张量 [Batch, 20, 9]
    torch.save({
        'inputs': torch.tensor(final_x),  # 已经是滑动窗口形态了
        'labels': torch.tensor(final_y)  # 已经是车体相对偏差目标了
    }, output_file)

    return True


if __name__ == "__main__":
    # 🌟 修复队友问题 1：统一修正仓库数据路径
    base_dir = './data_raw/QCarRawData'
    forward_dir = os.path.join(base_dir, 'Forward_Dataset')
    backward_dir = os.path.join(base_dir, 'Backward_Dataset')

    os.makedirs('./data_processed', exist_ok=True)

    process_dataset(forward_dir, './data_processed/qcar_forward_dataset.pt')
    process_dataset(backward_dir, './data_processed/qcar_backward_dataset.pt')