import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import math
from data import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerGRUEncoder(nn.Module):
    def __init__(
        self,
        in_chans=1,
        hidden=32,
        num_layers=2,
        dropout=0.2,
        output_dim=32,
        ty=False,
        seq_len=200,
        if_test=False,
        rnn_type="lstm",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.rnn_type = rnn_type.lower()
        if self.rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        self.gru_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # 输入归一化 - 通道维度
        self.input_norm = nn.BatchNorm1d(in_chans)

        # 逐层初始化循环层和对应的归一化层
        for i in range(num_layers):
            input_size = in_chans if i == 0 else hidden
            rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
            gru = rnn_cls(
                input_size=input_size,
                hidden_size=hidden,
                num_layers=1,
                batch_first=True,
                dropout=0,
            )
            self.gru_layers.append(gru)

            # GRU层归一化 - 通道维度
            norm = nn.BatchNorm1d(hidden)
            self.norm_layers.append(norm)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

        self.ty = ty
        self.if_test = if_test
        if ty == True:
            # 使用分组卷积替代循环
            self.conv_norm_first = nn.BatchNorm1d(4)

            self.conv1 = nn.Conv1d(4, 4 * 4, kernel_size=9, stride=1, padding=4, groups=4)
            self.conv_norm1 = nn.BatchNorm1d(4 * 4)

            self.conv2 = nn.Conv1d(4 * 4, 4, kernel_size=9, stride=1, padding=4, groups=4)

            self.conv3 = nn.Conv1d(4, 1, 9, 1, 4)

    def forward(self, x):
        # x: [B, 8, 240]
        all_layer_outputs = []
        channel_indices = []

        if self.ty == True:
            temp_output_1 = x[:, -1].unsqueeze(1)
            temp_output_0 = x[:, :-1]

            temp_output_0 = self.conv_norm_first(temp_output_0)
            temp_output_0 = F.gelu(self.conv_norm1(self.conv1(temp_output_0)))
            temp_output_0 = self.conv2(temp_output_0)

            if self.if_test == False:
                temp_output_0, channel_indices = shuffle_all_with_replacement_and_concat(temp_output_0)

            temp_output_0 = self.conv3(temp_output_0)
            temp_output = torch.cat([temp_output_0, temp_output_1], 1)

            x = temp_output

        # 准备循环层输入
        x = self.input_norm(x)
        x = x.transpose(1, 2)  # [B, 240, features]

        for i, (gru_layer, norm_layer) in enumerate(zip(self.gru_layers, self.norm_layers)):
            output, hidden = gru_layer(x)
            output = norm_layer(output.transpose(1, 2)).transpose(1, 2)
            all_layer_outputs.append(output)
            x = self.dropout(output)

        last_hidden = all_layer_outputs[-1][:, -1, :]
        output = self.fc(last_hidden)
        output = self.output_norm(output)

        return output, all_layer_outputs, channel_indices

class ClassificationDecoder(nn.Module):
    """分类解码器（含多层感知机）[6,11](@ref)"""

    def __init__(self, d_model=32, output_dim=2, dropout=0.1, ty=False):
        super().__init__()

        self.mlp = nn.Sequential(
            #nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
      

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.mlp(x)

class InformerClassifier(nn.Module):
    """完整分类模型"""

    def __init__(
        self,
        input_dim=1,
        d_model=32,
        num_layers=2,
        output_dim=32,
        target_dim=2,
        ty=False,
        dropout=0.1,
        if_test=False,
        rnn_type="lstm",
    ):
        super().__init__()
        self.encoder = MultiLayerGRUEncoder(
            in_chans=input_dim,
            hidden=d_model,
            output_dim=output_dim,
            dropout=dropout,
            num_layers=num_layers,
            ty=ty,
            if_test=if_test,
            rnn_type=rnn_type,
        )
        self.decoder = ClassificationDecoder(d_model=output_dim, output_dim=target_dim, dropout=dropout, ty=ty)

    def forward(self, x):

        features, all_layer_outputs, channel_indices = self.encoder(x)  # 提取全局特征
        logits = self.decoder(features)  # 生成分类结果
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F




class ComparativeModel(nn.Module):
    """完整分类模型（关系蒸馏：教师样本间关系指导学生样本间关系）"""

    def __init__(self, similarity_threshold=0.05, teacher_output_dim=32, student_output_dim=32, temperature=4):
        super().__init__()
        # 这里暂时没用到 embbed，如果你后面要对特征再做投影可以继续用
        self.embbed = nn.Sequential(
            nn.GELU(),
            nn.Linear(teacher_output_dim, (teacher_output_dim + student_output_dim) // 2),
            nn.GELU(),
            nn.Linear((teacher_output_dim + student_output_dim) // 2, student_output_dim),
            nn.LayerNorm(student_output_dim),
        )
        self.similarity_threshold = similarity_threshold  # 现在不再参与 loss，可按需删掉
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def pairwise_angle(self, x):
        """
        计算三元角

        返回:
        angle[i,j,k] = (i->j) 和 (i->k) 的夹角
        shape = [N,N,N]
        """

        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N,N,D]
        norm = F.normalize(diff, dim=2)

        angle = torch.einsum("ijd,ikd->ijk", norm, norm)  # [N,N,N]

        return angle

    def remove_diag(self, angle):
        """
        把 [N,N,N] 变成 [N,N-1,N-2]
        """

        device = angle.device
        N = angle.size(0)

        idx = torch.arange(N, device=device)

        i = idx.view(N, 1, 1)
        j = idx.view(1, N, 1)
        k = idx.view(1, 1, N)

        mask = (i != j) & (i != k) & (j != k)

        angle = angle[mask].view(N, N - 1, N - 2)

        return angle

    def forward(self, x_teacher, x_student, correct_mask):

        batch_size = x_teacher.size(0)

        x_teacher = x_teacher.detach()

        # flatten
        x_teacher = x_teacher.view(batch_size, -1)[correct_mask]
        x_student = x_student.view(batch_size, -1)[correct_mask]

        N = x_teacher.size(0)

        if N < 3:
            return torch.tensor(0.0, device=x_teacher.device)

        # ----------------------
        # 计算角度
        # ----------------------

        angle_t = self.pairwise_angle(x_teacher)
        angle_s = self.pairwise_angle(x_student)

        # ----------------------
        # 删除 i=j / i=k / j=k
        # ----------------------

        angle_t = self.remove_diag(angle_t)
        angle_s = self.remove_diag(angle_s)

        # shape
        # [N, N-1, N-2]

        # ----------------------
        # softmax on k
        # ----------------------

        T = self.temperature

        teacher_prob = F.softmax(angle_t / T, dim=2)
        student_log_prob = F.log_softmax(angle_s / T, dim=2)

        loss = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="batchmean"
        ) * (T ** 2)

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherComparativeModel(nn.Module):
    def __init__(self, temperature=0.1, feat_dim=32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
        )
        self.temperature = temperature

    def forward(self, x_teacher, x_student):
        """
        x_teacher, x_student: [B, D]
        """
        B = x_teacher.size(0)

        # flatten（如果本来就是 [B, D]，这一步其实是冗余的，但安全）
        x_teacher = x_teacher.view(B, -1)
        x_student = x_student.view(B, -1)

        # embedding
        t = self.embed(x_teacher)
        s = self.embed(x_student)

        # L2 normalize
        t = F.normalize(t, dim=1)
        s = F.normalize(s, dim=1)

        # -----------------------------
        # similarity matrix: [B, B]
        # -----------------------------
        logits1 = torch.matmul(s, t.T) / self.temperature
        logits2 = torch.matmul(t, s.T) / self.temperature

        # 正样本：对角线 (i, i)
        labels = torch.arange(B, device=logits1.device)

        # InfoNCE loss
        loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) / 2

        return loss
