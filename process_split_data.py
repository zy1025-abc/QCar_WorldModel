import os
import pandas as pd
import torch
import numpy as np
import random  # 🌟 引入随机库进行洗牌

def process_dataset(parent_dir, output_file, seq_length=20):
    print(f"\n🚀 开始处理数据集: {parent_dir}")

    action_cols = ['throttle', 'steering']
    state_cols = ['pos_x', 'pos_y', 'pos_z', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

    if not os.path.exists(parent_dir):
        print(f"❌ 找不到文件夹: '{parent_dir}'，请检查你的 data_raw 目录结构！")
        return False

    # 1. 收集所有的 CSV 文件路径
    all_csv_files = []
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        if not os.path.isdir(folder_path): continue

        for file_name in os.listdir(folder_path):
            if not file_name.endswith('.csv'): continue
            all_csv_files.append(os.path.join(folder_path, file_name))

    if not all_csv_files:
        print("❌ 没有找到任何 CSV 文件！")
        return False

    # 🌟 核心魔法：文件级洗牌 (File-Level Shuffle)
    print(f"📂 共找到 {len(all_csv_files)} 个 CSV 文件，正在进行全局随机打乱...")
    random.seed(42)  # 固定随机种子，保证每次洗牌结果一样，方便实验复现
    random.shuffle(all_csv_files)

    x_seqs, y_seqs, s_curr_list, s_next_list = [], [], [], []
    dropped_frames = 0
    total_files_processed = 0

    print("✂️ 正在按文件独立提取连续滑动窗口并清洗脏数据...")

    # 2. 🌟 架构升级：逐个文件独立处理，彻底杜绝跨文件拼接！
    for file_path in all_csv_files:
        try:
            df = pd.read_csv(file_path)
            missing_cols = [col for col in (action_cols + state_cols) if col not in df.columns]
            if missing_cols: continue

            df = df.dropna(subset=action_cols + state_cols)
            actions = torch.tensor(df[action_cols].values.astype(np.float32))
            states = torch.tensor(df[state_cols].values.astype(np.float32))

            num_samples = len(states) - seq_length
            if num_samples <= 0: continue # 文件太短不够切一个窗口，直接跳过

            total_files_processed += 1

            # 在单个完美的连续文件内部进行滑动窗口切分
            for i in range(num_samples):
                window_states = states[i: i + seq_length + 1].clone()
                window_actions = actions[i: i + seq_length]

                # 补丁 1: 拦截 OptiTrack 自身的瞬移/丢失脏数据
                diffs = window_states[1:, :2] - window_states[:-1, :2]
                if torch.any(torch.abs(diffs) > 0.5):
                    dropped_frames += 1
                    continue

                # 补丁 2: 四元数半球平滑
                for t in range(1, len(window_states)):
                    q_prev = window_states[t - 1, 3:7]
                    q_curr = window_states[t, 3:7]
                    if torch.dot(q_prev, q_curr) < 0:
                        window_states[t, 3:7] = -q_curr

                seq_states = window_states[:-1]  # 20帧历史
                current_state = window_states[-2]  # 预测起点
                next_state = window_states[-1]   # 预测终点

                delta_target = next_state - current_state
                joint_input = torch.cat([seq_states, window_actions], dim=1)  # [20, 9]

                x_seqs.append(joint_input)
                y_seqs.append(delta_target)
                s_curr_list.append(states[i + seq_length - 1])  # 保存原始坐标供画图
                s_next_list.append(states[i + seq_length])      # 保存原始坐标供画图

        except Exception as e:
            print(f"    ❌ 处理 {os.path.basename(file_path)} 时发生错误: {e}")

    # 3. 打包成最终的完美数据集
    print(f"✅ 处理完毕 (共 {total_files_processed} 个有效文件)。")
    print(f"🧹 成功清洗掉 {dropped_frames} 个脏窗口！保留了 {len(x_seqs)} 个完美的乱序连续窗口。")

    if len(x_seqs) == 0:
        return False

    torch.save({
        'inputs': torch.stack(x_seqs),       # X: [N, 20, 9]
        'labels': torch.stack(y_seqs),       # Y: [N, 7]
        's_curr': torch.stack(s_curr_list),  # 真实原状态 [N, 7]
        's_next': torch.stack(s_next_list)   # 真实下一状态 [N, 7]
    }, output_file)

    print(f"💾 数据已成功打包至: {output_file}\n")
    return True

if __name__ == "__main__":
    base_dir = './data_raw/QCarRawData'
    forward_dir = os.path.join(base_dir, 'Forward_Dataset')
    backward_dir = os.path.join(base_dir, 'Backward_Dataset')

    os.makedirs('./data_processed', exist_ok=True)
    process_dataset(forward_dir, './data_processed/qcar_forward_dataset.pt')
    process_dataset(backward_dir, './data_processed/qcar_backward_dataset.pt')
    print("🎉 预处理完毕！洗牌后的完美数据集已生成，可直接送入显卡训练！")