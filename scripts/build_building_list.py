"""
Bina etiketi olan görüntülerin listesini üretir (train ve val).
Eğitimde use_building_only için kullanılır.

Kullanım:
  python scripts/build_building_list.py --data_root /kaggle/input/.../RescueNet --out_dir /kaggle/working/building_lists
"""
import os
import sys
import argparse

# Proje kökü (RescueNetModel) path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_building_list


def main():
    p = argparse.ArgumentParser(description='Bina içeren görüntü listelerini üret')
    p.add_argument('--data_root', required=True, help='RescueNet dataset kökü (train/, val/ içeren)')
    p.add_argument('--out_dir', required=True, help='train.txt ve val.txt yazılacak klasör')
    args = p.parse_args()

    train_txt = os.path.join(args.out_dir, 'train.txt')
    val_txt = os.path.join(args.out_dir, 'val.txt')

    n_train = build_building_list(args.data_root, 'train', train_txt)
    n_val = build_building_list(args.data_root, 'val', val_txt)

    print(f'Bina içeren görüntüler: train={n_train}, val={n_val}')
    print(f'Yazıldı: {train_txt}, {val_txt}')


if __name__ == '__main__':
    main()
