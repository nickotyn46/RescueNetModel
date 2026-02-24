# RescueNet — Attention U-Net

Teknofest 2026 | Sentinair Team  
Semantic segmentation model for post-disaster UAV imagery.

## Model

**Attention U-Net** (Oktay et al., 2018)  
- 11 classes: Background, Water, 4× Building damage levels, Vehicle, Road-Clear, Road-Blocked, Tree, Pool  
- Input: 512×512  
- Best result in paper: **98.47% mIoU**  
- Deploy target: Raspberry Pi 5 + Hailo-8L (26 TOPS NPU)

## Dataset

[RescueNet](https://www.kaggle.com/datasets/yaroslavchyrko/rescuenet) — 4494 high-resolution UAV images after Hurricane Michael.

## Quick Start (Kaggle)

1. Add `yaroslavchyrko/rescuenet` dataset to your Kaggle notebook
2. Open `kaggle_train.ipynb`
3. Set `REPO_URL` to your GitHub repo
4. Run all cells

## Local Setup

```bash
pip install -r requirements.txt
python train.py --config configs/rescuenet_aunet.yaml
python evaluate.py --config configs/rescuenet_aunet.yaml --model-path checkpoints/best.pth
python export_onnx.py --config configs/rescuenet_aunet.yaml --model-path checkpoints/best.pth
```

## Deployment (RPi5 + Hailo-8L)

After ONNX export, use Hailo Dataflow Compiler (Ubuntu 22.04 / Docker):

```bash
hailo parse --hw-arch hailo8l --ckpt rescuenet_aunet.onnx
hailo optimize --hw-arch hailo8l hailo_model.har
hailo compile --hw-arch hailo8l hailo_model.har --output-dir .
# Copy .hef to RPi5
python inference.py --model rescuenet_aunet.hef
```

## Project Structure

```
RescueNetModel/
├── data/dataset.py          # RescueNet DataLoader
├── models/unet.py           # Attention U-Net (11 classes)
├── utils/metrics.py         # IoU, poly_lr, AverageMeter
├── transforms.py            # Joint image+mask augmentation
├── configs/rescuenet_aunet.yaml
├── train.py                 # Training script
├── evaluate.py              # Test set evaluation
├── export_onnx.py           # ONNX export for Hailo
├── inference.py             # RPi5 + Hailo-8L real-time inference
├── kaggle_train.ipynb       # Kaggle notebook (one-click run)
└── requirements.txt
```

## Training Details

| Parameter      | Value          |
|---------------|----------------|
| Input size     | 512×512        |
| Batch size     | 8              |
| Learning rate  | 0.001 (poly)   |
| Momentum       | 0.9            |
| Weight decay   | 0.00001        |
| LR power       | 0.9            |
| Max epochs     | 100            |
| Early stopping | 15 epochs      |
| Loss           | Weighted CrossEntropy |

## Citation

```bibtex
@article{rahnemoonfar2023rescuenet,
  title={RescueNet: a high resolution UAV semantic segmentation dataset for natural disaster damage assessment},
  author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Murphy, Robin},
  journal={Scientific data},
  volume={10}, number={1}, pages={913}, year={2023}
}
```
