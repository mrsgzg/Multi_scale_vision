import torch
import torch.nn as nn


class MultiScaleVisualEncoder(nn.Module):
    """多尺度视觉编码器，输出每帧的多尺度全局特征。"""

    def __init__(self, cnn_layers=3, cnn_channels=[64, 128, 256], input_channels=3):
        super().__init__()

        self.cnn_layers = cnn_layers
        self.cnn_channels = cnn_channels[:cnn_layers]

        self.conv_blocks = nn.ModuleList()
        self.global_pools = nn.ModuleList()

        in_channels = input_channels
        for out_channels in self.cnn_channels:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
            )
            self.global_pools.append(nn.AdaptiveAvgPool2d(1))
            in_channels = out_channels

        self.feature_dims = self.cnn_channels
        self.total_feature_dim = sum(self.feature_dims)

    def forward(self, x):
        multi_scale_feats = []
        current = x

        for conv_block, pool in zip(self.conv_blocks, self.global_pools):
            current = conv_block(current)
            multi_scale_feats.append(pool(current).flatten(1))

        return torch.cat(multi_scale_feats, dim=1)


class EmbodimentEncoder(nn.Module):
    """具身编码器，将每步关节状态映射到统一特征空间。"""

    def __init__(self, joint_dim=2, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )

    def forward(self, joint_positions):
        return self.encoder(joint_positions)


class StepwiseGatedFusion(nn.Module):
    """每个时间步的轻量门控融合。"""

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()

        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.joint_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, visual_hidden, joint_hidden):
        visual_state = self.visual_proj(visual_hidden)
        joint_state = self.joint_proj(joint_hidden)

        fusion_gate = self.gate(torch.cat([visual_state, joint_state], dim=-1))
        fused = fusion_gate * visual_state + (1.0 - fusion_gate) * joint_state

        return self.output_proj(fused)


class CountingDecoder(nn.Module):
    """计数分类头。"""

    def __init__(self, input_dim=256, hidden_dim=128, num_classes=11, dropout=0.1):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.decoder(x)


class UniversalEmbodiedCountingModel(nn.Module):
    """简洁版具身计数模型：双流LSTM + 每步融合。"""

    def __init__(self,
                 cnn_layers=3,
                 cnn_channels=[64, 128, 256],
                 visual_hidden_dim=256,
                 joint_hidden_dim=256,
                 lstm_layers=2,
                 joint_dim=2,
                 input_channels=3,
                 dropout=0.1,
                 num_classes=11,
                 **kwargs):
        super().__init__()

        self.joint_dim = joint_dim
        self.visual_hidden_dim = visual_hidden_dim
        self.joint_hidden_dim = joint_hidden_dim
        self.lstm_layers = lstm_layers

        if visual_hidden_dim != joint_hidden_dim:
            raise ValueError("visual_hidden_dim 和 joint_hidden_dim 必须一致，便于逐步融合")

        self.visual_encoder = MultiScaleVisualEncoder(
            cnn_layers=cnn_layers,
            cnn_channels=cnn_channels,
            input_channels=input_channels
        )
        self.embodiment_encoder = EmbodimentEncoder(
            joint_dim=joint_dim,
            hidden_dim=joint_hidden_dim,
            dropout=dropout
        )

        self.visual_projection = nn.Sequential(
            nn.Linear(self.visual_encoder.total_feature_dim, visual_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.visual_lstm = nn.LSTM(
            input_size=visual_hidden_dim,
            hidden_size=visual_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.joint_lstm = nn.LSTM(
            input_size=joint_hidden_dim,
            hidden_size=joint_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.step_fusion = StepwiseGatedFusion(
            hidden_dim=visual_hidden_dim,
            dropout=dropout
        )
        self.counting_decoder = CountingDecoder(
            input_dim=visual_hidden_dim,
            hidden_dim=visual_hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout
        )

        print("UniversalEmbodiedCountingModel 初始化:")
        print(f"  视觉特征维度: {self.visual_encoder.total_feature_dim}")
        print(f"  时序隐藏维度: {visual_hidden_dim}")
        print(f"  Joint维度: {joint_dim}")
        print("  结构: Dual-stream LSTM + Stepwise Gated Fusion")

    def forward(self, sequence_data):
        images = sequence_data['images']
        joints = sequence_data['joints']

        batch_size, seq_len = images.shape[:2]

        visual_features = []
        joint_features = []

        for t in range(seq_len):
            visual_features.append(self.visual_encoder(images[:, t]))
            joint_features.append(self.embodiment_encoder(joints[:, t]))

        visual_sequence = torch.stack(visual_features, dim=1)
        joint_sequence = torch.stack(joint_features, dim=1)

        visual_sequence = self.visual_projection(visual_sequence)

        visual_hidden_seq, _ = self.visual_lstm(visual_sequence)
        joint_hidden_seq, _ = self.joint_lstm(joint_sequence)

        fused_steps = []
        for t in range(seq_len):
            fused_steps.append(
                self.step_fusion(visual_hidden_seq[:, t], joint_hidden_seq[:, t])
            )

        fused_sequence = torch.stack(fused_steps, dim=1)
        sequence_logits = self.counting_decoder(fused_sequence)
        logits = sequence_logits[:, -1]

        return {
            'logits': logits,
            'sequence_logits': sequence_logits,
            'counts': sequence_logits,
            'visual_hidden_sequence': visual_hidden_seq,
            'joint_hidden_sequence': joint_hidden_seq,
            'fused_sequence': fused_sequence
        }

    def get_model_info(self):
        return {
            'model_type': 'UniversalEmbodiedCountingModel',
            'architecture': 'dual_stream_lstm_with_stepwise_gated_fusion',
            'has_multiscale_visual_encoder': True,
            'has_embodiment_encoder': True,
            'has_stepwise_fusion': True,
            'main_output': 'final_step_logits',
            'auxiliary_output': 'full_sequence_logits'
        }


def create_model(config, model_type='baseline'):
    """创建简洁版具身计数模型。"""
    if model_type != 'baseline':
        print(f"忽略 model_type={model_type}，当前仅保留简洁版 baseline 模型")

    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1

    model_config = config['model_config'].copy()
    model_config['input_channels'] = input_channels

    model = UniversalEmbodiedCountingModel(**model_config)

    print("创建简洁版具身计数模型")
    return model


if __name__ == "__main__":
    print("=== 简洁版具身计数模型测试 ===")

    config = {
        'image_mode': 'rgb',
        'model_config': {
            'cnn_layers': 3,
            'cnn_channels': [64, 128, 256],
            'visual_hidden_dim': 256,
            'joint_hidden_dim': 256,
            'lstm_layers': 2,
            'joint_dim': 2,
            'dropout': 0.1,
            'num_classes': 11
        }
    }

    device = torch.device('cpu')
    model = create_model(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")

    batch_size = 4
    seq_len = 11
    sequence_data = {
        'images': torch.randn(batch_size, seq_len, 3, 224, 224, device=device),
        'joints': torch.randn(batch_size, seq_len, 2, device=device)
    }

    model.eval()
    with torch.no_grad():
        outputs = model(sequence_data)

    print("输出形状:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print(f"模型信息: {model.get_model_info()}")
    print("=== 测试完成 ===")