import torch.nn as nn
import torch


class DeepLOBMultiTask(nn.Module):
    """
    DeepLOB + BiLSTM 混合架构，多任务输出
    用于预测中间价移动方向（5个时间跨度）
    """

    def __init__(self, num_classes=3, num_tasks=5, input_features=32):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.input_features = input_features

        # === DeepLOB CNN Blocks ===
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(2, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, input_features // 4)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # === Inception Modules ===
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 16, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
        )

        # === BiLSTM ===
        self.lstm_hidden = 64
        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=self.lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # === Multi-Task Output Heads ===
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.lstm_hidden, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, num_classes)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        # x: (batch, 1, 100, 32)

        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Inception modules
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)  # (batch, 48, seq_len, 1)

        # Reshape for LSTM: (batch, seq_len, features)
        x = x.squeeze(-1).permute(0, 2, 1)  # (batch, seq_len, 48)

        # BiLSTM
        x, _ = self.lstm(x)  # (batch, seq_len, 128)

        # Use last timestep output
        x = x[:, -1, :]  # (batch, 128)

        # Multi-task outputs
        outputs = []
        for head in self.task_heads:
            out = head(x)  # (batch, num_classes)
            out = torch.softmax(out, dim=-1)
            outputs.append(out)

        return outputs  # List of 5 tensors, each (batch, 3)


class DeepLOBSingle(nn.Module):
    """
    单任务DeepLOB模型（兼容原有example格式）
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

        # CNN blocks (same as DeepLOB)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(2, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 8)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # Inception modules
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 16, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
        )

        # FC output
        self.fc = nn.Sequential(
            nn.Linear(384, 64),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.reshape(-1, 48 * 8)
        x = self.fc(x)

        return torch.softmax(x, dim=1)


# 供相对路径导入使用
__all__ = ['DeepLOBMultiTask', 'DeepLOBSingle']