import torch
import torch.nn as nn

from Model_alexnet_embodiment import (  # type: ignore
    CountingDecoder,
    EmbodimentEncoder,
    MultiScaleVisualEncoder,
)


class VisualOnlyModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        hidden_dim = model_config.get("visual_hidden_dim", 256)
        lstm_layers = model_config.get("lstm_layers", 2)
        dropout = model_config.get("dropout", 0.1)
        num_classes = model_config.get("num_classes", 11)

        self.visual_encoder = MultiScaleVisualEncoder(
            cnn_layers=model_config.get("cnn_layers", 3),
            cnn_channels=model_config.get("cnn_channels", [64, 128, 256]),
            input_channels=model_config.get("input_channels", 3),
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(self.visual_encoder.total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.decoder = CountingDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, sequence_data):
        images = sequence_data["images"]
        _, seq_len = images.shape[:2]
        visual_seq = []
        for t in range(seq_len):
            visual_seq.append(self.visual_encoder(images[:, t]))
        visual_seq = self.visual_projection(torch.stack(visual_seq, dim=1))
        hidden_seq, _ = self.lstm(visual_seq)
        sequence_logits = self.decoder(hidden_seq)
        return {
            "logits": sequence_logits[:, -1],
            "sequence_logits": sequence_logits,
            "counts": sequence_logits,
        }


class JointOnlyModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        hidden_dim = model_config.get("joint_hidden_dim", 256)
        lstm_layers = model_config.get("lstm_layers", 2)
        dropout = model_config.get("dropout", 0.1)
        num_classes = model_config.get("num_classes", 11)

        self.encoder = EmbodimentEncoder(
            joint_dim=model_config.get("joint_dim", 2),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.decoder = CountingDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, sequence_data):
        joints = sequence_data["joints"]
        _, seq_len = joints.shape[:2]
        joint_seq = []
        for t in range(seq_len):
            joint_seq.append(self.encoder(joints[:, t]))
        joint_seq = torch.stack(joint_seq, dim=1)
        hidden_seq, _ = self.lstm(joint_seq)
        sequence_logits = self.decoder(hidden_seq)
        return {
            "logits": sequence_logits[:, -1],
            "sequence_logits": sequence_logits,
            "counts": sequence_logits,
        }


class EarlyFusionSingleStreamModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        visual_hidden_dim = model_config.get("visual_hidden_dim", 256)
        joint_hidden_dim = model_config.get("joint_hidden_dim", 256)
        fused_hidden_dim = model_config.get("fused_hidden_dim", 256)
        lstm_layers = model_config.get("lstm_layers", 2)
        dropout = model_config.get("dropout", 0.1)
        num_classes = model_config.get("num_classes", 11)

        self.visual_encoder = MultiScaleVisualEncoder(
            cnn_layers=model_config.get("cnn_layers", 3),
            cnn_channels=model_config.get("cnn_channels", [64, 128, 256]),
            input_channels=model_config.get("input_channels", 3),
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(self.visual_encoder.total_feature_dim, visual_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.joint_encoder = EmbodimentEncoder(
            joint_dim=model_config.get("joint_dim", 2),
            hidden_dim=joint_hidden_dim,
            dropout=dropout,
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(visual_hidden_dim + joint_hidden_dim, fused_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=fused_hidden_dim,
            hidden_size=fused_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.decoder = CountingDecoder(
            input_dim=fused_hidden_dim,
            hidden_dim=fused_hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, sequence_data):
        images = sequence_data["images"]
        joints = sequence_data["joints"]
        _, seq_len = images.shape[:2]

        fused_steps = []
        for t in range(seq_len):
            vis = self.visual_projection(self.visual_encoder(images[:, t]))
            emb = self.joint_encoder(joints[:, t])
            fused_steps.append(self.fusion_projection(torch.cat([vis, emb], dim=-1)))

        fused_seq = torch.stack(fused_steps, dim=1)
        hidden_seq, _ = self.lstm(fused_seq)
        sequence_logits = self.decoder(hidden_seq)
        return {
            "logits": sequence_logits[:, -1],
            "sequence_logits": sequence_logits,
            "counts": sequence_logits,
        }


class DualStreamLateFusionModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        hidden_dim = model_config.get("visual_hidden_dim", 256)
        lstm_layers = model_config.get("lstm_layers", 2)
        dropout = model_config.get("dropout", 0.1)
        num_classes = model_config.get("num_classes", 11)

        self.visual_encoder = MultiScaleVisualEncoder(
            cnn_layers=model_config.get("cnn_layers", 3),
            cnn_channels=model_config.get("cnn_channels", [64, 128, 256]),
            input_channels=model_config.get("input_channels", 3),
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(self.visual_encoder.total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.joint_encoder = EmbodimentEncoder(
            joint_dim=model_config.get("joint_dim", 2),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.visual_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.joint_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = CountingDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, sequence_data):
        images = sequence_data["images"]
        joints = sequence_data["joints"]
        _, seq_len = images.shape[:2]

        visual_seq = []
        joint_seq = []
        for t in range(seq_len):
            visual_seq.append(self.visual_projection(self.visual_encoder(images[:, t])))
            joint_seq.append(self.joint_encoder(joints[:, t]))

        visual_seq = torch.stack(visual_seq, dim=1)
        joint_seq = torch.stack(joint_seq, dim=1)

        vis_hidden, _ = self.visual_lstm(visual_seq)
        joint_hidden, _ = self.joint_lstm(joint_seq)

        fused_seq = self.fusion(torch.cat([vis_hidden, joint_hidden], dim=-1))
        sequence_logits = self.decoder(fused_seq)

        return {
            "logits": sequence_logits[:, -1],
            "sequence_logits": sequence_logits,
            "counts": sequence_logits,
            "visual_hidden_sequence": vis_hidden,
            "joint_hidden_sequence": joint_hidden,
        }


def create_baseline_model(model_name: str, model_config):
    if model_name == "joint_only":
        return JointOnlyModel(model_config)
    if model_name == "visual_only":
        return VisualOnlyModel(model_config)
    if model_name == "early_fusion_single_stream":
        return EarlyFusionSingleStreamModel(model_config)
    if model_name == "dual_stream_late_fusion":
        return DualStreamLateFusionModel(model_config)

    raise ValueError(
        f"Unsupported baseline model_name={model_name}. "
        "Expected one of: joint_only, visual_only, early_fusion_single_stream, dual_stream_late_fusion"
    )
