"""Munneke源码直接运行入口；无原生库或MATLAB运行依赖。"""
from .runtime import configure
ROOT = configure()

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='Munneke20Patch+原ArtVen/Tube传播循环+自主呼吸直接联合积分')
    parser.add_argument('--beats', type=int, default=2)
    parser.add_argument('--dt-us', type=int, choices=(250, 500), default=500)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='核对清单和源码初态，不加载CircAdapt')
    args = parser.parse_args()
    if args.check:
        manifest_path = ROOT/'manifest.json'
        count = 0
        if manifest_path.is_file():
            for relative, expected in json.loads(manifest_path.read_text(encoding='utf-8')).items():
                path = (ROOT/relative).resolve()
                if not path.is_relative_to(ROOT.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('文件身份不符：'+relative)
                count += 1
        from .model import CardiorespiratoryModel
        model = CardiorespiratoryModel(args.dt_us*1e-6)
        _, cardiac, _, _ = model.evaluate(0., model.initial_state())
        print(json.dumps(dict(status='通过', heart_backend='Munneke2023_python_source',
            manifest_files=count, circadapt_module_loaded='circadapt' in sys.modules,
            dll_required=False, matlab_required=False, patch_count=model.heart.patch_count,
            physical_states=model.physical_states,coronary_network=False,coronary_supply_demand=False,
            initial_cardiac_volumes_ml=cardiac['volume_ml'].tolist()),
            ensure_ascii=False, indent=2))
        return
    out = args.out or ROOT/'runs'/datetime.now().strftime('munneke_%Y%m%d_%H%M%S_%f')
    if not out.is_absolute():
        out = ROOT/out
    if not out.resolve().is_relative_to((ROOT/'runs').resolve()):
        parser.error('结果须保存在本项目runs下')
    if args.resume and args.out is None:
        parser.error('续算须用--out指定已有源码运行目录')
    from .model import simulate
    summary = simulate(out, beats=args.beats, dt_us=args.dt_us, resume=args.resume)
    from .plot import plot_results
    plot_results(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('结果目录：'+str(out))


if __name__ == '__main__':
    main()
