import trellis.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F

# TRELLIS에서 제공하는 models API 가정
decoder_gs = models.from_pretrained(
    'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16'
)

# 모델 구조 확인 (print / inspect)
# 어떤 속성(예: decoder_gs.backbone, decoder_gs.output_layer 등)으로 나뉘어 있는지 살펴봅니다.
print(decoder_gs)
