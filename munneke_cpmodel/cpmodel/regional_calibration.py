"""五变量区域负荷匹配：总血量和四个系统ArtVen的原p0AV。

目标、差分步幅、参数边界和回溯序列均预先固定；肺p0AV、心肌、激活、
血管壁及几何不参与求根。每次候选独立恢复同一77态和完整波历史检查点。
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from .calibration import alter_total_blood, load_checkpoint, odeca_beat, settle
from .model import CardiorespiratoryModel, save_json
from .settings import PA


TARGET_PRESSURE_PA = 12200.
TARGET_SYSTEMIC_FLOW = np.array([1.275e-5, 1.020e-5, 4.845e-5, 1.360e-5])
FD_STEPS = np.array([.002, .01, .01, .01, .01])
LOG_BOUNDS = np.array([.7, 8., 8., 8., 8.])
STEP_LIMITS = np.array([.1, 1., 1., 1., 1.])
DAMPINGS = (1., .5, .25, .125)
TARGET_TOLERANCE = .0035
RAMP_BEATS = 20
MAX_ITERATIONS = 12


class TrialInfeasible(RuntimeError):
    """物理状态或预定周期收敛不满足；允许回溯到更小参数步幅。"""


def target_residual(metric):
    """固定顺序：log(BrAr/12200Pa)，log(qCa/q0Ca)...log(qFe/q0Fe)。"""
    flows = np.asarray(metric['artven_flow_m3_s'], dtype=float)
    if flows.shape != (5,):
        raise ValueError('指标中的ArtVen平均流必须包含Ca/Br/Ce/Fe/Pu五项')
    ratios = np.r_[float(metric['baroreceptor_pressure_mmhg'])*PA/TARGET_PRESSURE_PA,
                   flows[:4]/TARGET_SYSTEMIC_FLOW]
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise TrialInfeasible('区域平均压力或流量非正/非有限，无法形成预定对数残差')
    return np.log(ratios)


def apply_ramp_parameters(model, state, base_blood, base_p0av, x, fraction):
    """仅按全部静脉腔/Tube分配血量，并修改四系统p0AV；不裁剪物理状态。"""
    x, base_p0av = np.asarray(x, dtype=float), np.asarray(base_p0av, dtype=float)
    if x.shape != (5,) or not np.isfinite(x).all() or base_p0av.shape != (5,):
        raise ValueError('区域匹配参数维度或有限值错误')
    if not 0. <= fraction <= 1.:
        raise ValueError('负荷渐变比例必须处于0至1')
    if not np.isfinite(base_blood) or base_blood <= 0 or not np.isfinite(base_p0av).all() or np.any(base_p0av <= 0):
        raise ValueError('基准血量和p0AV必须为有限正数')
    if not np.isclose(base_p0av[4], 750., rtol=0., atol=1e-10):
        raise ValueError('原肺p0AV必须保持750Pa')
    result = alter_total_blood(state, float(base_blood*np.exp(x[0]*fraction)))
    model.vascular.p0av[:4] = base_p0av[:4]*np.exp(x[1:]*fraction)
    model.vascular.p0av[4] = base_p0av[4]
    model.facp_control, model.pleural_fraction = 1., 1.
    return result


def _known_infeasible(exc):
    """只分类明确的数值物理失败；不吞掉I/O、配置错误或未知程序异常。"""
    if isinstance(exc, (TrialInfeasible, FloatingPointError, ZeroDivisionError)):
        return True
    physical_messages = (
        '拟议血量校正会导致非正容积', '联合模型需要有限心血管状态',
        '血管和心腔容积须正', 'TriSeg交界半径须正', 'C须非负', 'Lsi须正',
        '心腔容积、交界半径和Lsi须正，C须非负',
        '18个血管物理容积必须为有限正值，不裁剪ODE状态',
        '联合模型需要77个有限状态及正血管/心腔容积',
        '预设共同周期轨迹收敛标准未达到，未输出正式标定初态',
    )
    return isinstance(exc, (ValueError, RuntimeError)) and str(exc) in physical_messages


@dataclass(frozen=True)
class CandidateJob:
    ident: int
    x: tuple
    base_path: str
    folder: str


def _candidate_worker(job):
    """Windows spawn可调用的顶层worker；只写此候选自己的目录。"""
    folder = Path(job.folder)
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(id=job.ident, log_parameters=list(job.x), directory=str(folder))
    started = time.perf_counter()
    with (folder/'trial.log').open('w', encoding='utf-8', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        # 检查点/模型装载错误不属于候选不可行，直接传播给主进程。
        model = CardiorespiratoryModel(.0005, use_calibrated=False)
        state = load_checkpoint(job.base_path, model)
        if not np.allclose(model.vascular.q0av[:4], TARGET_SYSTEMIC_FLOW, rtol=0., atol=1e-16):
            raise ValueError('模型系统q0AV与预先固定的原区域目标不符')
        base_blood, base_p0av = float(state[:22].sum()), model.vascular.p0av.copy()
        record.update(blood_target_ml=float(base_blood*np.exp(job.x[0])*1e6),
                      p0av_target_pa=np.r_[base_p0av[:4]*np.exp(job.x[1:]), base_p0av[4]].tolist())
        ramp_history = []
        try:
            for beat in range(1, RAMP_BEATS+1):
                state = apply_ramp_parameters(model, state, base_blood, base_p0av,
                                              job.x, beat/RAMP_BEATS)
                _, trajectory = odeca_beat(model, state)
                state = trajectory[-1].copy()
                ramp_history.append(dict(beat=beat, fraction=beat/RAMP_BEATS,
                    blood_ml=float(state[:22].sum()*1e6), p0av_pa=model.vascular.p0av.tolist()))
            save_json(folder/'ramp.json', ramp_history)
            state, _, metric, periods = settle(model, state, folder, 'candidate',
                                               min_periods=3, max_periods=8, save_periods=False)
            residual = target_residual(metric)
        except Exception as exc:
            save_json(folder/'ramp.json', ramp_history)
            if not _known_infeasible(exc):
                traceback.print_exc(file=log)
                raise
            record.update(rejected=True, reason=str(exc), error_type=type(exc).__name__,
                          elapsed_wall_s=time.perf_counter()-started)
            save_json(folder/'result.json', record)
            return record
    record.update(rejected=False, metrics=metric, residual=residual.tolist(),
                  accepted_periods=len(periods), checkpoint=str(folder/'candidate_checkpoint.npz'),
                  elapsed_wall_s=time.perf_counter()-started)
    save_json(folder/'result.json', record)
    return record


class _Candidates:
    """按固定输入次序分配编号、保存结果；并行只发生于独立候选之间。"""
    def __init__(self, folder, base_path, executor=None):
        self.folder, self.base_path, self.executor = Path(folder), str(Path(base_path).resolve()), executor
        self.records, self.cache = [], {}

    def batch(self, points):
        keys = [tuple(float(v) for v in point) for point in points]
        jobs = []
        for key in keys:
            if key not in self.cache:
                ident = len(self.records)+1
                self.cache[key] = None
                self.records.append(dict(id=ident, log_parameters=list(key), status='running'))
                jobs.append(CandidateJob(ident, key, self.base_path,
                                         str(self.folder/f'trial_{ident:03d}')))
        save_json(self.folder/'evaluations.json', self.records)
        iterator = map(_candidate_worker, jobs) if self.executor is None else self.executor.map(_candidate_worker, jobs)
        for job, record in zip(jobs, iterator):
            self.cache[job.x] = record
            self.records[job.ident-1] = record
            save_json(self.folder/'evaluations.json', self.records)
        return [self.cache[key] for key in keys]

    def __call__(self, point):
        return self.batch([point])[0]


def _backtrack(x, step, residual, evaluate):
    """固定四级回溯；生理不可行与边界越界均显式记录，继续更小步幅。"""
    attempts = []
    for damping in DAMPINGS:
        candidate = x+damping*step
        row = dict(damping=damping, log_parameters=candidate.tolist())
        if np.any(abs(candidate) > LOG_BOUNDS):
            row.update(rejected=True, reason='超过预先固定的对数参数边界')
            attempts.append(row)
            continue
        result = evaluate(candidate)
        row['trial_id'] = result['id']
        if result['rejected']:
            row.update(rejected=True, reason=result['reason'])
        elif np.linalg.norm(result['residual']) < np.linalg.norm(residual):
            row.update(rejected=False, accepted=True, residual=result['residual'])
            attempts.append(row)
            return candidate, result, attempts
        else:
            row.update(rejected=True, reason='残差二范数未下降', residual=result['residual'])
        attempts.append(row)
    return None, None, attempts


def _solve(folder, candidates):
    x, iterations = np.zeros(5), []
    current = candidates(x)
    for iteration in range(MAX_ITERATIONS+1):
        if current['rejected']:
            raise RuntimeError('共同基准候选不可行：'+current['reason'])
        residual = np.asarray(current['residual'], dtype=float)
        if np.max(abs(np.expm1(residual))) <= TARGET_TOLERANCE:
            destination = folder/'matched_checkpoint.npz'
            shutil.copyfile(current['checkpoint'], destination)
            save_json(folder/'matched.json', dict(log_parameters=x.tolist(), metrics=current['metrics'],
                residual=residual.tolist(), maximum_target_relative_error=float(np.max(abs(np.expm1(residual)))),
                matched_trial_id=current['id'], iterations=iterations))
            return destination
        if iteration == MAX_ITERATIONS:
            break
        epsilon = FD_STEPS.copy()
        points = []
        for j in range(5):
            if x[j]+epsilon[j] > LOG_BOUNDS[j]:
                epsilon[j] *= -1
            point = x.copy()
            point[j] += epsilon[j]
            points.append(point)
        neighbors = candidates.batch(points)
        fd_records = []
        jacobian = np.empty((5, 5))
        for j, neighbor in enumerate(neighbors):
            row = dict(parameter_index=j, initial_trial_id=neighbor['id'])
            if neighbor['rejected']:
                row['initial_rejection'] = neighbor['reason']
                epsilon[j] *= -1
                point = x.copy()
                point[j] += epsilon[j]
                if abs(point[j]) > LOG_BOUNDS[j]:
                    raise RuntimeError('差分邻点不可行且反向邻点越界，未生成区域匹配参考')
                neighbor = candidates(point)
                row['opposite_trial_id'] = neighbor['id']
                if neighbor['rejected']:
                    raise RuntimeError('同参数双向差分均不可行，未生成区域匹配参考')
            row['used_step'] = float(epsilon[j])
            fd_records.append(row)
            jacobian[:, j] = (np.asarray(neighbor['residual'])-residual)/epsilon[j]
        record = dict(iteration=iteration, log_parameters=x.tolist(), residual=residual.tolist(),
                      finite_difference_steps=epsilon.tolist(), finite_difference_trials=fd_records,
                      jacobian=jacobian.tolist())
        iterations.append(record)
        save_json(folder/'newton_history.json', iterations)
        try:
            raw_step = -np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError as exc:
            record['failure'] = '有限差分Jacobian奇异'
            save_json(folder/'newton_history.json', iterations)
            raise RuntimeError('区域负荷Jacobian奇异，未生成匹配参考') from exc
        step = np.clip(raw_step, -STEP_LIMITS, STEP_LIMITS)
        record.update(unclipped_step=raw_step.tolist(), step=step.tolist())
        next_x, next_result, attempts = _backtrack(x, step, residual, candidates)
        record['backtracking'] = attempts
        save_json(folder/'newton_history.json', iterations)
        if next_x is None:
            raise RuntimeError('固定回溯序列未找到可行下降方向，未生成区域匹配参考')
        x, current = next_x, next_result
    raise RuntimeError('区域负荷匹配未达到预设五项目标，未生成匹配参考')


def match(folder, base_path, workers=4):
    """match(folder,base_path,workers=4) -> 通过五项目标的检查点Path。"""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError('并行worker数必须是1至4的整数')
    base_path, folder = Path(base_path).resolve(), Path(folder).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    folder.mkdir(parents=True, exist_ok=False)
    # 保存不可变的共同基准，避免外部任务续算原路径时改变后续候选的初态。
    frozen_base = folder/'base_checkpoint.npz'
    shutil.copyfile(base_path, frozen_base)
    save_json(folder/'protocol.json', dict(method='五变量原区域负荷确定性Newton求根',
        base_checkpoint=str(base_path), frozen_base_checkpoint=str(frozen_base),
        base_sha256=hashlib.sha256(frozen_base.read_bytes()).hexdigest(),
        adjusted_parameters=['log总血量倍率', 'logCa_p0AV倍率',
            'logBr_p0AV倍率', 'logCe_p0AV倍率', 'logFe_p0AV倍率'],
        target_pressure_pa=TARGET_PRESSURE_PA, target_systemic_flow_m3_s=TARGET_SYSTEMIC_FLOW.tolist(),
        finite_difference_log_steps=FD_STEPS.tolist(), log_parameter_bounds=LOG_BOUNDS.tolist(),
        maximum_log_steps=STEP_LIMITS.tolist(), backtracking_dampings=list(DAMPINGS),
        target_max_relative_error=TARGET_TOLERANCE, max_newton_iterations=MAX_ITERATIONS,
        ramp_beats=RAMP_BEATS, min_common_periods=3, max_common_periods=8,
        trajectory_tolerance=.002, required_consecutive_passes=2, workers=workers,
        candidate_initialization='各候选从同一77态/波缓存检查点起步；20搏对数负荷渐变',
        fixed_parameters='肺p0AV=750Pa、全部q0AV、心肌、激活、血管壁及几何',
        controller='FacpControl=1，自主呼吸胸膜压完整接入；无自动评分调参'))
    if workers == 1:
        return _solve(folder, _Candidates(folder, frozen_base))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        return _solve(folder, _Candidates(folder, frozen_base, executor))


def main():
    parser = argparse.ArgumentParser(description='原系统四区域与血压的五变量负荷匹配')
    parser.add_argument('--from-checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    print(match(args.out, args.from_checkpoint, args.workers), flush=True)


if __name__ == '__main__':
    main()
