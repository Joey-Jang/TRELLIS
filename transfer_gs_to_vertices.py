import os

from trellis.representations.gaussian.general_utils import PILtoTorch

os.environ['ATTN_BACKEND'] = 'xformers'  # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

from typing import *
import torch
import torch.nn as nn
from trellis.modules.sparse.basic import SparseTensor
import trellis.models as models
from dataloader import create_dataloader


###################################################
# 2. 간단 VertexDecoder: SLAT -> Vertices
###################################################
class VertexDecoder(nn.Module):
    """
    단순 MLP 디코더 예시:
    SLAT(SparseTensor)을 입력받아 [N, 3]의 정점(Vertices)로 매핑
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # (x, y, z)
            nn.Tanh()
        )

    def forward(self, slat: SparseTensor):
        """
        Args:
            slat.feats: [N, in_dim]
        Returns:
            pred_vertices: [N, 3]
        """
        feats = slat.feats  # [N, C]
        pred_verts = self.mlp(feats)  # [N, 3]
        return pred_verts


###################################################
# 3. 부분 동결(Freeze) / 전이학습용 GsToVertexDecoder
###################################################
class GsToVertexDecoder(nn.Module):
    """
    Pretrained SLatGaussianDecoder를 래핑:
    - pos_embedder, input_layer, blocks는 그대로 재사용
    - out_layer만 정점(3차원)으로 교체
    - freeze_backbone나 freeze_until_layer로 부분 동결 가능
    """

    def __init__(self,
                 gs_decoder: nn.Module,
                 out_dim: int = 3,
                 freeze_backbone: bool = False,
                 freeze_until_layer: int = -1):
        """
        Args:
            gs_decoder: 이미 학습된 SLatGaussianDecoder
            out_dim: 최종 출력 차원 (기본: 3 -> (x,y,z))
            freeze_backbone: True면 backbone 전부 동결
            freeze_until_layer: 0 <= freeze_until_layer <= #blocks (12)
                                block 인덱스 미만만 동결, 나머지 학습
                                freeze_backbone=True이면 이 값 무시
        """
        super().__init__()

        # SLatGaussianDecoder의 주요 구성 복사
        self.pos_embedder = gs_decoder.pos_embedder
        self.input_layer = gs_decoder.input_layer
        self.blocks = gs_decoder.blocks  # ModuleList(12개의 SparseTransformerBlock)

        # out_layer 교체 (기존 out_features=448 -> out_dim=3)
        in_features = gs_decoder.out_layer.in_features  # 보통 768
        self.new_out_layer = nn.Linear(in_features, out_dim)

        # 전체 backbone 동결 시
        if freeze_backbone:
            for param in self.pos_embedder.parameters():
                param.requires_grad = False
            for param in self.input_layer.parameters():
                param.requires_grad = False
            for block in self.blocks:
                for p in block.parameters():
                    p.requires_grad = False
        else:
            # 부분 동결
            if freeze_until_layer >= 0:
                for i, block in enumerate(self.blocks):
                    if i < freeze_until_layer:
                        for p in block.parameters():
                            p.requires_grad = False
                    else:
                        for p in block.parameters():
                            p.requires_grad = True

    def forward(self, slat: SparseTensor) -> torch.Tensor:
        """
        Args:
            slat: SparseTensor (feats=[N,8], coords=[N,4], 등)
        Returns:
            pred_vertices: [N, out_dim=3]
        """
        # 1) 위치 임베딩
        slat = self.pos_embedder(slat)
        # 2) input_layer로 확장 [N,8] -> [N,768]
        slat.feats = self.input_layer(slat.feats)
        # 3) 12개 블록 통과
        for block in self.blocks:
            slat.feats = block(slat.feats, slat.coords)
        # 4) 최종 out_layer (3차원)
        pred_vertices = self.new_out_layer(slat.feats)
        return pred_vertices


###################################################
# 4. 간단 Dataset & 로스, 학습 루프 예시
###################################################
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class My3DDataset(Dataset):
    """
    (이미지, 3D 정점GT)을 반환하는 간단 예시
    실제로는 별도 전처리 / 정점 개수 가변 등에 맞춰 수정 필요
    """

    def __init__(self, img_list: List[Image.Image], vertices_list: List[torch.Tensor]):
        """
        img_list: 이미지 목록
        vertices_list: 각 이미지에 대응하는 [L,3] Tensor
        """
        self.img_list = img_list
        self.vertices_list = vertices_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.img_list[idx], self.vertices_list[idx]


def vertex_loss(pred_vertices: torch.Tensor, gt_vertices_batch: List[torch.Tensor],
                layout: List[slice]):
    """
    pred_vertices: [N, 3] (배치 전체)
    gt_vertices_batch: 길이 B, 각 원소 [L_b, 3]
    layout: slat.layout
    """
    total_loss = 0.0
    for i, slice_i in enumerate(layout):
        preds_i = pred_vertices[slice_i]  # [L_i, 3]
        gts_i = gt_vertices_batch[i]  # [L_i, 3]이라고 가정
        total_loss += nn.functional.mse_loss(preds_i, gts_i)
    return total_loss / len(gt_vertices_batch)


def chamfer_distance(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """
    pred_points: [N, 3]
    gt_points:   [M, 3]
    Returns:
        scalar (Chamfer Distance)
    """
    # 1) (N, M) 거리 행렬 (유클리드)
    dist_mat = torch.cdist(pred_points, gt_points, p=2)  # [N, M]

    # 2) 각각의 점이 "상대 집합에서 가장 가까운 점"까지 거리
    #    - 예측->GT
    dist_pred = dist_mat.min(dim=1)[0].mean()  # shape: []
    #    - GT->예측
    dist_gt = dist_mat.min(dim=0)[0].mean()  # shape: []

    return dist_pred + dist_gt


def vertex_loss_chamfer(pred_vertices: torch.Tensor,
                        gt_vertices_batch: List[torch.Tensor],
                        layout: List[slice]) -> torch.Tensor:
    """
    pred_vertices: [N, 3] (배치 전체를 펼친 정점)
    gt_vertices_batch: 길이 B의 리스트(각 [L_b, 3]), 샘플마다 정점 수 다를 수 있음
    layout: slat.layout, 배치별 [slice] 범위
    """
    total_loss = 0.0
    for i, slice_i in enumerate(layout):
        pred_b = pred_vertices[slice_i]  # shape: [L_pred_i, 3]
        gt_b = gt_vertices_batch[i]  # shape: [L_gt_i,   3]
        # Chamfer Distance
        cd_i = chamfer_distance(pred_b, gt_b)
        total_loss += cd_i
    return total_loss / len(layout)


###################################################
# 5. 실제 학습/사용 예시 (전체 스크립트 메인)
###################################################
if __name__ == "__main__":
    import os

    # 1) 사전학습된 인코더 (SLAT Encoder) 로드
    #    (TRELLIS: 'slat_enc_swin8_B_64l8_fp16')
    encoder = models.from_pretrained('JeffreyXiang/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16')
    encoder.eval()

    # 2) 사전학습된 가우시안 디코더 로드
    #    (TRELLIS: 'slat_dec_gs_swin8_B_64l8gs32_fp16')
    decoder_gs = models.from_pretrained('JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16')

    # 3) 전이학습용 Vertex 디코더 생성
    #    - 예: 12개 중 앞 8개 블록은 동결, 나머지 학습
    vertex_decoder = GsToVertexDecoder(
        gs_decoder=decoder_gs,
        out_dim=3,
        freeze_backbone=False,
        freeze_until_layer=8
    ).cuda()

    # 4) Optimizer (vertex_decoder만)
    optimizer = torch.optim.Adam(vertex_decoder.parameters(), lr=1e-4)

    # (인코더까지 학습하려면 아래처럼)
    # optimizer = torch.optim.Adam(
    #     list(vertex_decoder.parameters()) + list(encoder.parameters()),
    #     lr=1e-5
    # )

    # 5) 예시 데이터셋 구성
    train_loader = create_dataloader(
        img_dir="assets/training_data/image",
        vertex_dir="assets/training_data/vertices",
        batch_size=3
    )

    # 6) 학습 루프
    epochs = 2
    for epoch in range(epochs):
        vertex_decoder.train()
        for images, gt_vertices_batch in train_loader:
            # images: B개
            # gt_vertices_batch: 길이 B 리스트
            # 1) 이미지 -> SLAT
            #    encoder(...)가 SparseTensor 반환한다고 가정
            slat = encoder(images)  # type: SparseTensor
            slat = slat.cuda()

            # 2) 디코더 -> 예측 정점
            pred_vertices = vertex_decoder(slat)  # [N, 3]

            # 3) Chamfer Distance 로스
            loss = vertex_loss_chamfer(pred_vertices, gt_vertices_batch, slat.layout)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch + 1}/{epochs}] Loss={loss.item():.4f}")

    print("Training complete!")
