import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import time


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
# 2. 训练主引擎
# ==========================================
def train_model(dataset_path, model_save_path, norm_save_path, test_data_save_path):
    # 🌟 GPU 加速核心：自动检测并挂载你的独立显卡！
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚡ 训练引擎启动！当前使用的计算硬件是: {device}")
    if device.type == 'cuda':
        print(f"🎮 成功点亮显卡: {torch.cuda.get_device_name(0)}")

    # 1. 直接加载刚才切分好的极品 3D 张量数据 (无需再做任何滑动窗口切割)
    print(f"📦 正在加载预处理数据集: {dataset_path}")
    data = torch.load(dataset_path, weights_only=True)
    X_all = data['inputs'].float()  # 形状: [样本总数, 20, 9]
    Y_all = data['labels'].float()  # 形状: [样本总数, 7] (车体局部变化量)

    # 2. 严格按时间顺序划分训练集 (90%) 和测试集 (10%)
    split_idx = int(len(X_all) * 0.9)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    Y_train, Y_test = Y_all[:split_idx], Y_all[split_idx:]

    print(f"📊 数据划分完毕 -> 训练集: {len(X_train)} 条 | 测试集: {len(X_test)} 条")

    # 3. 翻译官培训 (Z-Score 归一化)
    # 注意：绝对不能偷看测试集，只能用训练集的均值和方差！
    x_mean = X_train.mean(dim=(0, 1), keepdim=True)
    x_std = X_train.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)

    y_mean = Y_train.mean(dim=0, keepdim=True)
    y_std = Y_train.std(dim=0, keepdim=True).clamp(min=1e-6)

    # 对训练集进行归一化
    X_train_norm = (X_train - x_mean) / x_std
    Y_train_norm = (Y_train - y_mean) / y_std

    # 4. 构建数据装载机 (DataLoader)，填饱显卡的肚子
    batch_size = 256  # 🌟 如果显存大，可以继续调大到 512，训练会像飞一样快
    train_dataset = TensorDataset(X_train_norm, Y_train_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 5. 初始化大脑
    model = QCarWorldModel().to(device)  # 把模型送入显卡
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 加入学习率衰减：越到后面学得越精细
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_epochs = 100
    print("\n🚀 开始高强度训练 (请观察显卡起飞)...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            # 🌟 把当前批次的数据也送入显卡
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_dataset)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {avg_loss:.6f} | Lr: {scheduler.get_last_lr()[0]:.6f}")

    end_time = time.time()
    print(f"\n🎉 训练圆满结束！总耗时: {(end_time - start_time) / 60:.2f} 分钟")

    # 6. 保存所有的心血成果
    os.makedirs('models_saved', exist_ok=True)
    os.makedirs('results_evaluation', exist_ok=True)

    # 6.1 保存模型权重
    torch.save(model.state_dict(), model_save_path)

    # 6.2 保存翻译官字典 (留给线上 CARLA 驱动使用)
    torch.save({
        'x_mean': x_mean, 'x_std': x_std,
        'y_mean': y_mean, 'y_std': y_std,
        'seq_length': 20
    }, norm_save_path)

    # 6.3 提取当前绝对状态，保存期末考卷 (留给手工测试脚本验证)
    S_curr = X_test[:, -1, :7]  # 直接从历史窗口最后一帧提取当前 7 维状态
    torch.save({
        'X_test': X_test,  # 保存原始物理刻度的输入
        'Y_test': Y_test,  # 保存真实的局部偏差目标
        'S_curr': S_curr  # 顺手把当前绝对位置存下来，方便你用 manual_test 验算
    }, test_data_save_path)

    print(f"💾 模型、字典和期末考卷已全部保存至对应目录！\n" + "=" * 50)


if __name__ == "__main__":
    # --- 训练前向模型 ---
    train_model(
        dataset_path='./data_processed/qcar_forward_dataset.pt',
        model_save_path='./models_saved/forward_world_model.pth',
        norm_save_path='./models_saved/forward_normalization.pt',
        test_data_save_path='./results_evaluation/forward_test_data.pt'
    )

    # --- 训练倒车模型 ---
    train_model(
        dataset_path='./data_processed/qcar_backward_dataset.pt',
        model_save_path='./models_saved/backward_world_model.pth',
        norm_save_path='./models_saved/backward_normalization.pt',
        test_data_save_path='./results_evaluation/backward_test_data.pt'
    )