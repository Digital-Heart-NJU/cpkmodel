"""Munneke基础心脏源码心肺模型的本地配置。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
PA = 133.322368


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_parameters():
    return read_json(DATA/'config.json')


def load_initial():
    return read_json(DATA/'initial_state.json')
