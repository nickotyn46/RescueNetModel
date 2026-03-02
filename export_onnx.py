"""
Export trained segmentation model (Attention U-Net or PSPNet) to ONNX format
for Hailo-8 deployment.

IMPORTANT for Hailo:
  - Input shape must be FIXED (no dynamic axes)
  - opset_version=11 for maximum compatibility
  - After export, convert with Hailo Dataflow Compiler:
      hailo parse --hw-arch hailo8 --ckpt model.onnx
      hailo optimize --hw-arch hailo8 hailo_model.har
      hailo compile --hw-arch hailo8 hailo_model.har --output-dir .

Usage:
    python export_onnx.py --config configs/rescuenet_pspnet101.yaml \
                          --model-path /kaggle/working/checkpoints_pspnet/best.pth \
                          --output rescuenet_pspnet101.onnx
"""

import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort

from models.unet import AttU_Net


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Export Attention U-Net to ONNX')
    parser.add_argument('--config',     default='configs/rescuenet_pspnet101.yaml')
    parser.add_argument('--model-path', required=True, help='Path to best.pth')
    parser.add_argument('--output',     default='rescuenet_aunet.onnx')
    parser.add_argument('--input-size', type=int, default=713,
                        help='Input image size (square, default 713)')
    return parser.parse_args()


def export(args, cfg):
    num_classes = cfg['DATA']['num_classes']
    train_cfg   = cfg.get('TRAIN', {})
    arch        = train_cfg.get('arch', 'aunet')
    h = w = args.input_size

    if arch == 'aunet':
        model = AttU_Net(img_ch=3, output_ch=num_classes)
        model_desc = 'Attention U-Net'
    elif arch in ('pspnet', 'pspnet_resnet101'):
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "segmentation_models_pytorch is required for PSPNet export. "
                "Install it with `pip install segmentation-models-pytorch`."
            ) from e

        encoder_name = 'resnet101' if '101' in arch else 'resnet50'

        class PSPNetSMPWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = smp.PSPNet(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=3,
                    classes=num_classes,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, h_in, w_in = x.shape
                new_h = (h_in + 7) // 8 * 8
                new_w = (w_in + 7) // 8 * 8
                pad_bottom = new_h - h_in
                pad_right = new_w - w_in
                if pad_bottom > 0 or pad_right > 0:
                    x = F.pad(x, (0, pad_right, 0, pad_bottom))
                y = self.model(x)
                if pad_bottom > 0 or pad_right > 0:
                    y = y[..., :h_in, :w_in]
                return y

        model = PSPNetSMPWrapper()
        model_desc = f'PSPNet-{encoder_name}'
    else:
        raise ValueError(f"Unknown architecture '{arch}' in config. Supported: 'aunet', 'pspnet_resnet101'.")

    ckpt  = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Fixed input — NO dynamic axes for Hailo compatibility
    dummy_input = torch.randn(1, 3, h, w)

    print(f'Exporting {model_desc} to ONNX: {args.output}')
    print(f'  Input shape : (1, 3, {h}, {w})')
    print(f'  Output shape: (1, {num_classes}, {h}, {w})')
    print(f'  ONNX opset  : 11')

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # NO dynamic_axes — Hailo requires fixed shapes
    )

    # Validate ONNX model
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print('ONNX model validation: PASSED')

    # Verify with ONNX Runtime
    sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
    dummy_np = dummy_input.numpy()
    out = sess.run(None, {'input': dummy_np})
    print(f'ONNX Runtime output shape: {out[0].shape}  (expected (1, {num_classes}, {h}, {w}))')

    # Check PyTorch vs ONNX output match
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
    max_diff = np.abs(torch_out - out[0]).max()
    print(f'Max difference PyTorch vs ONNX: {max_diff:.6f}  (should be < 1e-4)')

    print(f'\nExport complete: {args.output}')
    print('\nNext steps for Hailo-8 deployment (RPi AI HAT+ 26 TOPS):')
    print('  1. Install Hailo Dataflow Compiler (Ubuntu 22.04 / Docker)')
    print('  2. hailo parse --hw-arch hailo8 --ckpt rescuenet_aunet.onnx')
    print('  3. hailo optimize --hw-arch hailo8 hailo_model.har')
    print('  4. hailo compile --hw-arch hailo8 hailo_model.har --output-dir .')
    print('  5. Copy rescuenet_aunet.hef to Raspberry Pi 5')


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    export(args, cfg)


if __name__ == '__main__':
    main()
