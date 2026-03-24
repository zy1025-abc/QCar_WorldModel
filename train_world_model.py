import torch
import torch.nn as nn
import torch.optim as optim
import os
import time


# ================= 1. 定义世界模型架构 =================
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


# ================= 2. 主训练流程 =================
def train_model(direction="forward", train_ratio=0.9):
    print(f"\n🚀 === 开始训练回调平滑版 {direction.upper()} 世界模型 ===")
    dataset_path = f'./data_processed/qcar_{direction}_dataset.pt'

    # 1. 直接加载已经切分好、平滑完毕的极品数据
    print(f"📖 正在加载预处理数据集: {dataset_path}")
    data = torch.load(dataset_path, weights_only=True)
    X_all = data['inputs']
    Y_all = data['labels']
    S_curr_all = data['s_curr']
    S_next_all = data['s_next']

    # 2. 划分训练集和测试集
    split_idx = int(len(X_all) * train_ratio)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    Y_train, Y_test = Y_all[:split_idx], Y_all[split_idx:]
    S_curr_test, S_next_test = S_curr_all[split_idx:], S_next_all[split_idx:]

    print(f"📊 数据划分完毕 -> 训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # 3. 归一化 (只用训练集的分布)
    x_mean, x_std = X_train.mean(dim=(0, 1), keepdim=True), X_train.std(dim=(0, 1), keepdim=True) + 1e-6
    y_mean, y_std = Y_train.mean(dim=0, keepdim=True), Y_train.std(dim=0, keepdim=True) + 1e-6

    X_train_norm = (X_train - x_mean) / x_std
    Y_train_norm = (Y_train - y_mean) / y_std

    # 4. 初始化训练环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎮 当前训练使用的计算硬件是: {device}")
    model = QCarWorldModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    epochs = 50
    batch_size = 256

    dataset = torch.utils.data.TensorDataset(X_train_norm, Y_train_norm)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 5. 开始疯狂炼丹
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 5 == 0:
            print(
                f"  ▶ Epoch [{epoch + 1}/{epochs}] | 平均 Loss: {epoch_loss / len(loader):.6f} | 当前学习率: {current_lr:.6f}")

    print(f"⏱️ 训练耗时: {(time.time() - start_time) / 60:.2f} 分钟")

    # 6. 保存所有的心血成果
    os.makedirs('models_saved_v1', exist_ok=True)
    os.makedirs('./results_evaluation', exist_ok=True)

    torch.save(model.state_dict(), f'models_saved_v1/{direction}_world_model.pth')
    torch.save({
        'x_mean': x_mean, 'x_std': x_std,
        'y_mean': y_mean, 'y_std': y_std,
        'seq_length': 20
    }, f'models_saved_v1/{direction}_normalization.pt')

    # 留给画图脚本的期末考卷
    torch.save({
        'X_test': X_test,
        'Y_test_real_delta': Y_test,
        'S_curr': S_curr_test,
        'S_next_real': S_next_test
    }, f'./results_evaluation/{direction}_test_data.pt')

    print(f"✅ {direction.upper()} 终极模型训练大功告成！\n" + "=" * 50)


if __name__ == "__main__":
    train_model(direction="forward")
    train_model(direction="backward")
    print("\n🎉 全部闭环！数据与模型分离架构升级完毕！")