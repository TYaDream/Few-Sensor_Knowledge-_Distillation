import numpy as np
import os
from torch.utils.data import Dataset,DataLoader
import torch
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_channel_wise(data):
    """
    对每个通道单独进行标准化
    data: [batch_size, 8, 240]
    """
    # 计算每个通道的均值和标准差
    # 在batch维度和时间维度上计算
    channel_means = data.mean(dim=(0, 2), keepdim=True)  # [1, 8, 1]
    channel_stds = data.std(dim=(0, 2), keepdim=True)  # [1, 8, 1]

    # 避免除零，添加小常数
    channel_stds = torch.where(channel_stds == 0, torch.ones_like(channel_stds), channel_stds)

    # 标准化
    normalized_data = (data - channel_means) / channel_stds

    return normalized_data


def shuffle_all_with_replacement_and_concat(x):
    """
    打乱所有通道的顺序（允许重复），然后与指定通道拼接

    Args:
        x: 输入张量，形状为 [batch_size, 8, 240]

    Returns:
        processed_tensor: 处理后的张量，形状为 [batch_size, 8 + len(choose_dim), 240]
        channel_presence: 通道出现频率，形状为 [batch_size, 8]
    """
    original_channels = x.shape[1]
    if original_channels > 4:
        fixed_x = x[:, 4:, :]
        x = x[:, :4, :]
    else:
        fixed_x = None

    batch_size, num_channels, seq_len = x.shape

    # 对batch中的每个样本独立处理
    results = []
    channel_presence_list = []

    for i in range(batch_size):
        # 提取当前样本的所有通道数据
        current_x = x[i]  # [8, 240]

        # 随机选择通道（允许重复）
        indices = torch.randint(0, num_channels, (num_channels,))
        shuffled_x = current_x[indices]  # [8, 240]

        # 计算每个通道的出现频率
        frequency = torch.zeros(num_channels, dtype=torch.float)
        for idx in indices:
            frequency[idx] += 1
        frequency = frequency / num_channels  # 转换为频率 [0, 1] 范围

        results.append(shuffled_x)
        channel_presence_list.append(frequency)

    # 将batch维度重新组合
    result = torch.stack(results, dim=0)  # [batch_size, 8, 240]
    channel_presence = torch.stack(channel_presence_list, dim=0)  # [batch_size, 8]

    # 如果需要强制某个通道的频率，可以在这里设置
    # 例如：channel_presence[:, 6] = 1.0  # 强制第6个通道频率为1.0

    if fixed_x is not None:
        result = torch.cat([result, fixed_x], dim=1)

    return result, channel_presence


def sine_weighted_merge(x, base_freq=0.01):
    """
    使用正弦波加权合并多通道张量

    Args:
        x: 输入张量 [batch_size, num_channels, seq_length]
        base_freq: 基础频率，控制正弦波的频率范围

    Returns:
        合并后的张量 [batch_size, seq_length]
    """
    batch_size, num_channels, seq_length = x.shape

    # 生成正弦波权重
    t = torch.arange(seq_length, device=x.device, dtype=x.dtype)
    weights = []

    for i in range(num_channels):
        # 每个通道有不同的频率
        freq = base_freq * (i + 1)
        channel_weights = torch.sin(2 * math.pi * freq * t / seq_length)
        weights.append(channel_weights)

    weights.reverse()
    # 堆叠权重并应用到输入
    weights_tensor = torch.stack(weights)  # [num_channels, seq_length]
    weighted_x = x * weights_tensor.unsqueeze(0)  # [batch_size, num_channels, seq_length]

    # 平均所有通道
    return weighted_x.mean(dim=1)  # [batch_size, seq_length]

# def shuffle_all_with_replacement_and_concat(x_all):
#     """
#     打乱所有通道的顺序（允许重复），然后与指定通道拼接
#
#     Args:
#         x: 输入张量，形状为 [batch_size, 8, 240]
#
#     Returns:
#         processed_tensor: 处理后的张量，形状为 [batch_size, 8 + len(choose_dim), 240]
#         channel_presence: 通道出现掩码，形状为 [batch_size, 8]
#     """
#     x = x_all[:, :8, :]
#     x_2 = x_all[:, 8:, :]
#     batch_size, num_channels, seq_len = x.shape
#
#     # 对batch中的每个样本独立处理
#     results = []
#     channel_presence_list = []
#
#     for i in range(batch_size):
#         # 提取当前样本的所有通道数据
#         current_x = x[i]  # [8, 240]
#
#         # 随机选择一个通道索引，然后重复8次
#         chosen_index = torch.randint(0, num_channels, (1,))
#         indices = chosen_index.repeat(num_channels)  # [8]
#         shuffled_x = current_x[indices]  # [8, 240]
#
#         # 创建通道出现掩码：只有被选中的通道为1，其他为0
#         presence_mask = torch.zeros(num_channels, dtype=torch.long)
#         presence_mask[chosen_index] = 1
#
#         results.append(shuffled_x)
#         channel_presence_list.append(presence_mask)
#
#     # 将结果拼接起来
#     shuffled_x_all = torch.stack(results)  # [batch_size, 8, 240]
#     channel_presence = torch.stack(channel_presence_list)  # [batch_size, 8]
#
#     # 与x_2拼接
#     processed_tensor = torch.cat([shuffled_x_all, x_2], dim=1)  # [batch_size, 8 + len(choose_dim), 240]
#
#     return processed_tensor, channel_presence


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path


class My_Dataset(Dataset):
    def __init__(self, csv_path, sequence_length=50, change_buffer=10):
        """
        从CSV文件加载数据，每个样本是连续200个时间点的4个传感器值
        样本的最后一个点不能在浓度变化40个时间点内

        Args:
            csv_path: CSV文件路径
            sequence_length: 每个样本的时间点数量
            change_buffer: 浓度变化点前后的缓冲区大小（时间点数量）
            transform: 数据变换
        """
        self.sequence_length = sequence_length
        self.change_buffer = change_buffer

        # 读取CSV文件
        df = pd.read_csv(csv_path)
        df = df.iloc[::4]

        # 提取传感器数据和浓度标签
        self.sensor_data = df[['Sensor_Group_1', 'Sensor_Group_2', 'Sensor_Group_3', 'Sensor_Group_4']].values
        # self.co_concentration = df['CO_conc'].values / 533.33
        # self.ethylene_concentration = df['Ethylene_conc'].values / 20
        self.methane_concentration = df['methane_conc'].values / 296.67
        self.ethylene_concentration = df['Ethylene_conc'].values / 20

        # 检查数据长度是否足够
        if len(self.sensor_data) < self.sequence_length:
            raise ValueError(f"数据长度({len(self.sensor_data)})小于序列长度({self.sequence_length})")

        # 检测浓度变化点
        self.valid_indices = self._find_valid_indices()

        if len(self.valid_indices) == 0:
            raise ValueError("没有找到符合条件的有效样本索引")

    def _find_valid_indices(self):
        """
        找到所有有效的起始索引（样本的最后一个点不在浓度变化40个时间点内）

        Returns:
            valid_indices: 有效起始索引的列表
        """
        # 检测CO浓度变化点
        methane_changes = np.where(np.diff(self.methane_concentration) != 0)[0] + 1

        # 检测乙烯浓度变化点
        ethylene_changes = np.where(np.diff(self.ethylene_concentration) != 0)[0] + 1

        # 合并所有变化点
        all_changes = np.concatenate([methane_changes, ethylene_changes])
        all_changes = np.unique(all_changes)

        # 创建无效区域的标记数组
        invalid_mask = np.zeros(len(self.sensor_data), dtype=bool)

        # 标记每个变化点前后change_buffer个时间点为无效
        for change_point in all_changes:
            start = max(0, change_point - self.change_buffer)
            end = min(len(self.sensor_data), change_point + self.change_buffer)
            invalid_mask[start:end] = True

        # 找到所有有效的起始索引
        # 样本的最后一个点是 idx + sequence_length - 1
        valid_indices = []
        for idx in range(len(self.sensor_data) - self.sequence_length + 1):
            last_point = idx + self.sequence_length - 1
            if not invalid_mask[last_point]:
                valid_indices.append(idx)

        return valid_indices

    def __len__(self):
        # 返回有效样本的数量
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 忽略传入的idx，随机选择一个有效的起始索引
        #actual_idx = random.choice(self.valid_indices)

        # 获取连续sequence_length个时间点的传感器数据
        x = self.sensor_data[idx:idx + self.sequence_length]

        # 获取最后一个时间点的浓度标签
        y = np.array([
            self.methane_concentration[idx + self.sequence_length - 1],
            self.ethylene_concentration[idx + self.sequence_length - 1]
        ])

        # 转换为PyTorch张量
        x = torch.tensor(x, dtype=torch.float32).transpose(0,1)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

    def random_sample(self):
        """随机选取一个有效的起始点并返回样本"""
        # 随机选择一个有效索引
        idx = random.randint(0, len(self) - 1)
        return self[idx]


class ZhangGasDataset(Dataset):
    """
    读取按浓度目录组织的张成师兄数据集。

    目录名格式: gas1,gas2,gas3
    TXT内容: 每行4个传感器响应值，默认截取或插值到固定序列长度。
    """

    def __init__(
        self,
        root_dir,
        sequence_length=200,
        label_scale=(1000.0, 200.0, 500.0),
        normalize_sensor=True,
        normalize_method="zscore",
        preload=True,
        fit_stats=True,
        stats=None,
    ):
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.label_scale = torch.tensor(label_scale, dtype=torch.float32)
        self.normalize_sensor = normalize_sensor
        self.normalize_method = normalize_method.lower()
        self.preload = preload
        if self.normalize_method not in {"zscore", "minmax"}:
            raise ValueError("normalize_method must be 'zscore' or 'minmax'")

        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.root_dir}")

        self.samples = self._collect_samples()
        if len(self.samples) == 0:
            raise ValueError(f"没有在目录中找到TXT样本: {self.root_dir}")
        self.cached_x = [self._read_txt(path) for path, _ in self.samples] if self.preload else None

        if stats is not None:
            self.sensor_mean, self.sensor_std = stats
        elif fit_stats and self.normalize_sensor:
            self.sensor_mean, self.sensor_std = self._fit_sensor_stats()
        else:
            self.sensor_mean = torch.zeros(4, 1, dtype=torch.float32)
            self.sensor_std = torch.ones(4, 1, dtype=torch.float32)

    def _collect_samples(self):
        samples = []
        for label_dir in sorted([p for p in self.root_dir.iterdir() if p.is_dir()]):
            try:
                label = [float(v) for v in label_dir.name.split(",")]
            except ValueError:
                continue
            if len(label) != 3:
                continue

            for txt_path in sorted(label_dir.glob("*.TXT")) + sorted(label_dir.glob("*.txt")):
                samples.append((txt_path, torch.tensor(label, dtype=torch.float32)))
        return samples

    def _read_txt(self, path):
        data = np.loadtxt(path, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 4:
            raise ValueError(f"传感器列数不足4列: {path}")

        data = data[:, :4]
        data = self._fix_length(data)
        return torch.tensor(data.T, dtype=torch.float32)

    def _fix_length(self, data):
        if data.shape[0] == self.sequence_length:
            return data
        if data.shape[0] > self.sequence_length:
            return data[-self.sequence_length:]

        old_x = np.linspace(0.0, 1.0, data.shape[0], dtype=np.float32)
        new_x = np.linspace(0.0, 1.0, self.sequence_length, dtype=np.float32)
        resized = np.stack([np.interp(new_x, old_x, data[:, i]) for i in range(data.shape[1])], axis=1)
        return resized.astype(np.float32)

    def _fit_sensor_stats(self):
        all_x = torch.stack(
            self.cached_x if self.cached_x is not None else [self._read_txt(path) for path, _ in self.samples],
            dim=0,
        )
        if self.normalize_method == "minmax":
            mean = all_x.amin(dim=(0, 2), keepdim=False).view(4, 1)
            std = (all_x.amax(dim=(0, 2), keepdim=False).view(4, 1) - mean)
        else:
            mean = all_x.mean(dim=(0, 2), keepdim=False).view(4, 1)
            std = all_x.std(dim=(0, 2), keepdim=False).view(4, 1)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return mean, std

    def fit_sensor_stats(self, indices=None):
        if indices is None:
            selected_x = self.cached_x if self.cached_x is not None else [self._read_txt(path) for path, _ in self.samples]
        else:
            selected_x = [self.cached_x[i] for i in indices] if self.cached_x is not None else [self._read_txt(self.samples[i][0]) for i in indices]
        all_x = torch.stack(selected_x, dim=0)
        if self.normalize_method == "minmax":
            mean = all_x.amin(dim=(0, 2), keepdim=False).view(4, 1)
            std = (all_x.amax(dim=(0, 2), keepdim=False).view(4, 1) - mean)
        else:
            mean = all_x.mean(dim=(0, 2), keepdim=False).view(4, 1)
            std = all_x.std(dim=(0, 2), keepdim=False).view(4, 1)
        std = torch.where(std == 0, torch.ones_like(std), std)
        self.sensor_mean = mean
        self.sensor_std = std
        return mean, std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = self.cached_x[idx].clone() if self.cached_x is not None else self._read_txt(path)
        if self.normalize_sensor:
            x = (x - self.sensor_mean) / self.sensor_std
        y = label / self.label_scale
        return x, y

    def denormalize_labels(self, y):
        scale = self.label_scale.to(y.device if torch.is_tensor(y) else "cpu")
        return y * scale
