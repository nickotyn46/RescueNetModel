"""
Export trained Attention U-Net to ONNX format for Hailo-8 deployment.

IMPORTANT for Hailo:
  - Input shape must be FIXED (no dynamic axes)
  - opset_version=11 for maximum compatibility
  - After export, convert with Hailo Dataflow Compiler:
      hailo parse --hw-arch hailo8 --ckpt model.onnx
      hailo optimize --hw-arch hailo8 hailo_model.har
      hailo compile --hw-arch hailo8 hailo_model.har --output-dir .

Usage:
    python export_onnx.py --config configs/rescuenet_aunet.yaml \
                          --model-path /kaggle/working/checkpoints/best.pth \
                          --output rescuenet_aunet.onnx
"""

import argparse
import yaml
import numpy as np
import torch
import onnx
import onnxruntime as ort

from models.unet import AttU_Net


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Export Attention U-Net to ONNX')
    parser.add_argument('--config',     default='configs/rescuenet_aunet.yaml')
    parser.add_argument('--model-path', required=True, help='Path to best.pth')
    parser.add_argument('--output',     default='rescuenet_aunet.onnx')
    parser.add_argument('--input-size', type=int, default=713,
                        help='Input image size (square, default 713)')
    return parser.parse_args()


def export(args, cfg):
    num_classes = cfg['DATA']['num_classes']
    h = w = args.input_size

    model = AttU_Net(img_ch=3, output_ch=num_classes)
    ckpt  = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Fixed input — NO dynamic axes for Hailo compatibility
    dummy_input = torch.randn(1, 3, h, w)

    print(f'Exporting to ONNX: {args.output}')
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
