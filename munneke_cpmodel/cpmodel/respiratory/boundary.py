"""独立心肺自主呼吸输入。"""
from bisect import bisect_right
from dataclasses import dataclass
from ..circulation.parameters import finite_real


@dataclass(frozen=True)
class RespiratoryBoundary:
    respiratory_rate: float = 12.
    muscle_pressure_min: float = -5.

    def __post_init__(self):
        finite_real(self.respiratory_rate,'呼吸频率',positive=True)
        finite_real(self.muscle_pressure_min,'肌压力幅度')
        if self.muscle_pressure_min>0:raise ValueError('自主吸气肌压力须非正；0表示关闭肌驱动')

