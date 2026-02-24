"""
Real-time inference on Raspberry Pi 5 + Hailo-8L (AI HAT+).

Reads frames from Raspberry Pi Camera Module V3 Wide, runs segmentation,
and reports building damage levels and road status per frame.

Requirements on RPi:
    pip install hailo-platform opencv-python numpy Pillow

Usage:
    python inference.py --model rescuenet_aunet.hef [--save-output]
"""

import argparse
import time
import numpy as np
from collections import Counter

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface,
        InferVStreams, ConfigureParams, InputVStreamParams,
        OutputVStreamParams, FormatType
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────────────────────

NUM_CLASSES  = 11
INPUT_HEIGHT = 512
INPUT_WIDTH  = 512

CLASS_NAMES = [
    'Background',
    'Water',
    'Building-No-Damage',
    'Building-Minor-Damage',
    'Building-Major-Damage',
    'Building-Total-Destruction',
    'Vehicle',
    'Road-Clear',
    'Road-Blocked',
    'Tree',
    'Pool',
]

COLOR_MAP = np.array([
    [0,   0,   0],    # 0 Background
    [61,  230, 250],  # 1 Water
    [180, 120, 120],  # 2 Building-No-Damage
    [235, 255, 7],    # 3 Building-Minor-Damage
    [255, 184, 6],    # 4 Building-Major-Damage
    [255, 0,   0],    # 5 Building-Total-Destruction
    [255, 0,   245],  # 6 Vehicle
    [140, 140, 140],  # 7 Road-Clear
    [160, 150, 20],   # 8 Road-Blocked
    [4,   250, 7],    # 9 Tree
    [255, 235, 0],    # 10 Pool
], dtype=np.uint8)

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Classes reported in final output
BUILDING_CLASSES = {2, 3, 4, 5}
ROAD_CLASSES     = {7, 8}


# ─── Preprocessing / Postprocessing ──────────────────────────────────────────

def preprocess(frame_bgr):
    """BGR frame → normalized float32 array (1, 3, H, W) for Hailo."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_WIDTH, INPUT_HEIGHT),
                         interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))           # HWC → CHW
    img = np.expand_dims(img, axis=0)             # 1CHW
    return img.astype(np.float32)


def postprocess(output):
    """Hailo output (1, num_classes, H, W) → predicted class map (H, W)."""
    pred = np.argmax(output[0], axis=0).astype(np.uint8)
    return pred


def colorize(pred):
    """Convert class-index map to RGB image for visualization."""
    return COLOR_MAP[pred]


def analyze_scene(pred):
    """Analyze segmentation mask and return damage/road report."""
    total_pixels = pred.size
    counts = Counter(pred.flatten().tolist())

    # Building damage breakdown
    building_report = {}
    for cls_id in BUILDING_CLASSES:
        pct = counts.get(cls_id, 0) / total_pixels * 100
        building_report[CLASS_NAMES[cls_id]] = round(pct, 2)

    # Road status
    road_clear   = counts.get(7, 0) / total_pixels * 100
    road_blocked = counts.get(8, 0) / total_pixels * 100
    road_report = {
        'Road-Clear':   round(road_clear, 2),
        'Road-Blocked': round(road_blocked, 2),
    }

    return building_report, road_report


def print_report(building, road, fps):
    print(f'\n[FPS: {fps:.1f}]')
    print('  Buildings:')
    for name, pct in building.items():
        bar = '█' * int(pct / 2)
        print(f'    {name:<30} {pct:>5.1f}%  {bar}')
    print('  Roads:')
    for name, pct in road.items():
        bar = '█' * int(pct / 2)
        print(f'    {name:<30} {pct:>5.1f}%  {bar}')


# ─── Hailo inference ──────────────────────────────────────────────────────────

class HailoInference:
    def __init__(self, hef_path):
        assert HAILO_AVAILABLE, 'hailo_platform not installed. Run: pip install hailo-platform'
        self.hef    = HEF(hef_path)
        self.device = VDevice()
        net_groups  = self.device.configure(self.hef)
        self.network_group = net_groups[0]
        self.ng_params = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

    def infer(self, input_data):
        with InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params
        ) as infer_pipeline:
            input_dict = {
                self.hef.get_input_vstream_infos()[0].name: input_data
            }
            with self.network_group.activate(self.ng_params):
                output = infer_pipeline.infer(input_dict)
        output_name = self.hef.get_output_vstream_infos()[0].name
        return output[output_name]

    def close(self):
        self.device.release()


# ─── Fallback ONNX inference (for testing without Hailo) ─────────────────────

class ONNXInference:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider']
        )
        self.input_name = self.sess.get_inputs()[0].name

    def infer(self, input_data):
        return self.sess.run(None, {self.input_name: input_data})[0]


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='RescueNet inference on RPi5 + Hailo-8L')
    parser.add_argument('--model', required=True,
                        help='Path to .hef file (Hailo) or .onnx file (fallback)')
    parser.add_argument('--source', default='camera',
                        help='Input source: "camera" or path to video/image file')
    parser.add_argument('--save-output', action='store_true',
                        help='Save annotated frames to output.mp4')
    parser.add_argument('--report-interval', type=int, default=30,
                        help='Print scene report every N frames (default 30)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load inference engine
    if args.model.endswith('.hef'):
        print(f'Loading Hailo HEF: {args.model}')
        engine = HailoInference(args.model)
    else:
        print(f'Loading ONNX (fallback): {args.model}')
        engine = ONNXInference(args.model)

    # Input source
    if args.source == 'camera':
        assert PICAMERA_AVAILABLE, 'picamera2 not available'
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(
            main={'size': (1920, 1080), 'format': 'RGB888'}
        ))
        cam.start()
        use_camera = True
    else:
        assert CV2_AVAILABLE, 'opencv-python not installed'
        cap = cv2.VideoCapture(args.source)
        use_camera = False

    writer = None
    if args.save_output and CV2_AVAILABLE:
        writer = cv2.VideoWriter(
            'output.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            10, (INPUT_WIDTH, INPUT_HEIGHT)
        )

    frame_idx = 0
    t_start   = time.time()

    print('Starting inference. Press Ctrl+C to stop.')
    try:
        while True:
            # Capture frame
            if use_camera:
                frame_rgb = cam.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

            t0 = time.time()

            # Run inference
            inp  = preprocess(frame_bgr)
            out  = engine.infer(inp)
            pred = postprocess(out)

            fps = 1.0 / (time.time() - t0)
            frame_idx += 1

            # Analyze and report
            if frame_idx % args.report_interval == 0:
                building, road = analyze_scene(pred)
                print_report(building, road, fps)

            # Visualize
            if CV2_AVAILABLE:
                color_mask = colorize(pred)
                overlay = cv2.addWeighted(
                    cv2.resize(frame_bgr, (INPUT_WIDTH, INPUT_HEIGHT)),
                    0.5,
                    cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR),
                    0.5,
                    0
                )
                if writer:
                    writer.write(overlay)

    except KeyboardInterrupt:
        pass

    total_time = time.time() - t_start
    print(f'\nProcessed {frame_idx} frames in {total_time:.1f}s '
          f'({frame_idx/total_time:.1f} FPS average)')

    if use_camera:
        cam.stop()
    elif not use_camera:
        cap.release()
    if writer:
        writer.release()
    if hasattr(engine, 'close'):
        engine.close()


if __name__ == '__main__':
    main()
