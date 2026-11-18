"""自主呼吸线性子系统的分段矩阵指数精确传播。

保持 kernel.py 原方程。吸气驱动是累计相位的二次式，呼气驱动是
指数式；分别增广到8维和7维后求矩阵指数，不近似刚性卷积核。
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy.linalg import expm
from .boundary import RespiratoryBoundary
from .kernel import muscle_pressure
from .parameters import CMH2O_TO_MMHG


@dataclass
class FastRespiratoryEvaluation:
    derivative: np.ndarray
    values: dict


def evaluate_respiration_fast(parameters, state, boundary):
    """原呼吸values/derivative接口，省略未使用的Jacobian和partials。"""
    x = np.asarray(state, dtype=float)
    if x.shape != (6,) or not np.isfinite(x).all() or np.any(x[:4] <= 0):
        raise ValueError('呼吸需六有限状态与四正气室容积')
    if not isinstance(boundary, RespiratoryBoundary):
        raise ValueError('需RespiratoryBoundary')
    p, b = parameters, boundary
    pmus, _ = muscle_pressure(p, x[5], b.muscle_pressure_min)
    ppl = x[4]/p.chest_wall_compliance+pmus+p.pleural_end_expiratory
    pl = (x[0]-p.laryngeal_unstressed)/p.laryngeal_compliance
    ptr = (x[1]-p.tracheal_unstressed)/p.tracheal_compliance+ppl
    pbr = (x[2]-p.bronchial_unstressed)/p.bronchial_compliance+ppl
    pa = (x[3]-p.alveolar_unstressed)/p.alveolar_compliance+ppl
    qm = -pl/p.mouth_resistance
    qlt = (pl-ptr)/p.larynx_trachea_resistance
    qtb = (ptr-pbr)/p.trachea_bronchi_resistance
    qba = (pbr-pa)/p.bronchi_alveoli_resistance
    derivative = np.array([qm-qlt, qlt-qtb, qtb-qba, qba, qlt, b.respiratory_rate/60])
    values = dict(p_l=pl, p_tr=ptr, p_br=pbr, p_A=pa, q_mouth=qm,
                  q_lt=qlt, q_tb=qtb, q_bA=qba, p_mus=pmus,
                  ppl_cmh2o=ppl, ppl_mmhg=CMH2O_TO_MMHG*ppl,
                  V_gas=float(x[:4].sum()), V_dead=float(x[:3].sum()),
                  chest_invariant=float(x[4]-x[1:4].sum()),
                  p_abd=float(-x[4]/p.abdominal_compliance-pmus+p.abdominal_end_expiratory))
    if not np.isfinite(derivative).all() or not np.isfinite(list(values.values())).all():
        raise ValueError('呼吸代数非有限，不裁剪气室')
    return FastRespiratoryEvaluation(derivative, values)


@njit(cache=True)
def _regular_steps(states, start, count, initial_phase, increment, matrix,
                   inspiration, alpha, scale):
    for k in range(start, start+count):
        phase = initial_phase+k*increment
        q = phase-np.floor(phase)
        if 1-q < 1e-13:
            q = 0.
        driver = np.empty(matrix.shape[0])
        driver[:5] = states[k, :5]
        driver[5] = 1.
        if inspiration:
            driver[6], driver[7] = q, q*q
        else:
            driver[6] = np.exp(-(q-alpha)/scale)
        for i in range(5):
            value = 0.
            for j in range(len(driver)):
                value += matrix[i, j]*driver[j]
            states[k+1, i] = value
        states[k+1, 5] = initial_phase+(k+1)*increment


class RespiratoryPropagator:
    """固定边界/步长的精确线性呼吸传播，不依赖心血管状态。"""

    def __init__(self, parameters, boundary, dt):
        if not isinstance(boundary, RespiratoryBoundary):
            raise ValueError('需RespiratoryBoundary')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('呼吸传播步长须为有限正数')
        self.parameters, self.boundary, self.dt = parameters, boundary, float(dt)
        p = parameters
        self.rate = boundary.respiratory_rate/60
        beta = 1/(1+p.inspiratory_expiratory_ratio)
        self.alpha = 1-beta
        self.scale = p.expiration_tau_fraction*beta
        c = np.array([p.laryngeal_compliance, p.tracheal_compliance,
                      p.bronchial_compliance, p.alveolar_compliance])
        vu = np.array([p.laryngeal_unstressed, p.tracheal_unstressed,
                       p.bronchial_unstressed, p.alveolar_unstressed])
        r = np.array([p.mouth_resistance, p.larynx_trachea_resistance,
                      p.trachea_bronchi_resistance, p.bronchi_alveoli_resistance])
        s = np.array([0., 1., 1., 1.])
        pressure_matrix = np.zeros((4, 5))
        pressure_matrix[:, :4] = np.diag(1/c)
        pressure_matrix[:, 4] = s/p.chest_wall_compliance
        flow_matrix = (-np.eye(4)+np.eye(4, k=-1))/r[:, None]
        balance = np.array([[1., -1., 0., 0.], [0., 1., -1., 0.],
                            [0., 0., 1., -1.], [0., 0., 0., 1.],
                            [0., 1., 0., 0.]])
        self.A = balance@flow_matrix@pressure_matrix
        self.b0 = balance@flow_matrix@(-vu/c+s*p.pleural_end_expiratory)
        self.b1 = balance@flow_matrix@s
        amplitude = boundary.muscle_pressure_min
        self._inspiration = np.zeros((8, 8))
        self._inspiration[:5, :5] = self.A
        self._inspiration[:5, 5] = self.b0
        self._inspiration[:5, 6] = self.b1*amplitude/(self.alpha*beta)
        self._inspiration[:5, 7] = -self._inspiration[:5, 6]
        self._inspiration[6, 5] = self.rate
        self._inspiration[7, 6] = 2*self.rate
        end = np.exp(-1/p.expiration_tau_fraction)
        denominator = -np.expm1(-1/p.expiration_tau_fraction)
        self._expiration = np.zeros((7, 7))
        self._expiration[:5, :5] = self.A
        self._expiration[:5, 5] = self.b0-self.b1*amplitude*end/denominator
        self._expiration[:5, 6] = self.b1*amplitude/denominator
        self._expiration[6, 6] = -self.rate/self.scale
        self._full = (expm(self._inspiration*self.dt), expm(self._expiration*self.dt))

    def _advance_partial(self, state, phase, duration):
        """仅相位分界落在采样步内部时调用，逐分段精确传播。"""
        x = state.copy()
        remaining = duration
        while remaining > 1e-15:
            q = phase-np.floor(phase)
            # 分界是连续驱动；向前传播选取右侧的解析分支。
            if abs(q-self.alpha) < 1e-13:
                q = self.alpha
            if q < self.alpha:
                h = min(remaining, (self.alpha-q)/self.rate)
                vector = np.r_[x, 1., q, q*q]
                x = (expm(self._inspiration*h)@vector)[:5]
            else:
                h = min(remaining, (1-q)/self.rate)
                vector = np.r_[x, 1., np.exp(-(q-self.alpha)/self.scale)]
                x = (expm(self._expiration*h)@vector)[:5]
            phase += h*self.rate
            remaining -= h
            # 消除恰好跨周期时舍入造成的零长度尾段。
            if abs(phase-round(phase)) < 1e-13:
                phase = float(round(phase))
        return x

    def trajectory(self, initial6, steps):
        initial = np.asarray(initial6, dtype=float)
        if initial.shape != (6,) or not np.isfinite(initial).all() or np.any(initial[:4] <= 0):
            raise ValueError('呼吸需六有限状态与四正气室容积')
        if isinstance(steps, bool) or int(steps) != steps or steps < 0:
            raise ValueError('传播步数须为非负整数')
        steps = int(steps)
        states = np.empty((steps+1, 6))
        states[0] = initial
        initial_phase, increment = initial[5], self.rate*self.dt
        k = 0
        while k < steps:
            phase = initial_phase+k*increment
            q = phase-np.floor(phase)
            if abs(q-self.alpha) < 1e-13:
                q = self.alpha
            if 1-q < 1e-13:
                q = 0.
            inspiration = q < self.alpha
            endpoint = self.alpha if inspiration else 1.
            n = min(steps-k, int(np.floor((endpoint-q)/increment+1e-9)))
            if n:
                _regular_steps(states, k, n, initial_phase, increment,
                               self._full[0 if inspiration else 1], inspiration,
                               self.alpha, self.scale)
                k += n
            else:
                states[k+1, :5] = self._advance_partial(states[k, :5], phase, self.dt)
                states[k+1, 5] = initial_phase+(k+1)*increment
                k += 1
        if not np.isfinite(states).all() or np.any(states[:, :4] <= 0):
            raise FloatingPointError('呼吸精确传播产生非有限或非正气室，不裁剪状态')
        return states
