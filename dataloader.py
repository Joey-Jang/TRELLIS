from PIL.Image import open as openPILImage
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from trellis.representations.gaussian.general_utils import PILtoTorch


class VertexDataset(Dataset):
    def __init__(self, img_dir, vertex_dir):
        """
        Args:
            img_dir (str): 이미지 데이터가 있는 디렉토리 경로
            vertex_dir (str): 정점 데이터가 있는 디렉토리 경로
        """
        self.img_dir = img_dir
        self.vertex_dir = vertex_dir

        # 파일 리스트 가져오기
        self.file_list = [f for f in os.listdir(vertex_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        base_name = os.path.splitext(file_name)[0]

        # 이미지 데이터 로드
        img_path = os.path.join(self.img_dir, f"{base_name}.png")
        image = openPILImage(img_path).convert('RGB')
        image_torch = PILtoTorch(image, (518, 518))

        # 정점 데이터 로드
        vertex_path = os.path.join(self.vertex_dir, f"{base_name}.npy")
        vertices = np.load(vertex_path)  # (N, 3) 형태

        # 정점 수 제한 (만약 5000개보다 많다면)
        if vertices.shape[0] > 5000:
            indices = np.random.choice(vertices.shape[0], 5000, replace=False)
            vertices = vertices[indices]

        # numpy array를 torch tensor로 변환
        vertices = torch.FloatTensor(vertices)

        return image_torch, vertices


# 데이터 크기를 맞추기 위한 커스텀 collate 함수
def collate_fn(batch):
    vertices, point_clouds = zip(*batch)

    # 정점 수가 다른 경우를 처리
    max_vertices = max(v.size(0) for v in vertices)
    padded_vertices = torch.stack([
        torch.cat([v, v.new_zeros(max_vertices - v.size(0), 3)])
        if v.size(0) < max_vertices else v
        for v in vertices
    ])

    # 포인트 클라우드 수가 다른 경우를 처리
    max_points = max(pc.size(0) for pc in point_clouds)
    padded_point_clouds = torch.stack([
        torch.cat([pc, pc.new_zeros(max_points - pc.size(0), 3)])
        if pc.size(0) < max_points else pc
        for pc in point_clouds
    ])

    return padded_vertices, padded_point_clouds


def create_dataloader(img_dir, vertex_dir, batch_size=32, num_workers=4):
    """
    DataLoader 생성 함수

    Args:
        vertex_dir (str): 정점 데이터 디렉토리
        pointcloud_dir (str): 포인트 클라우드 데이터 디렉토리
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩에 사용할 워커 수
    """
    dataset = VertexDataset(img_dir, vertex_dir)
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=num_workers,
        # collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader


# 사용 예시
if __name__ == "__main__":
    # 데이터 디렉토리 설정
    img_dir = "assets/training_data/image"
    vertex_dir = "assets/training_data/vertices"

    # DataLoader 생성
    train_loader = create_dataloader(
        img_dir=img_dir,
        vertex_dir=vertex_dir,
        batch_size=3
    )
