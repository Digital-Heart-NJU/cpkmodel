"""Munneke2023 原非冠脉 ArtVen/Tube 网络，全部量使用 SI。

逐式对应 ArtVenV2p、TubeV2p、pNodeVDot、TubeDelays。传播缓冲是独立
动力学状态；evaluate 只读，只有推进器调用 update_delay 才更新当前及下一格。
胸膜压仅添加到原 Thorax 覆盖的血管和 Tube；心腔外压由心肌模块计算。
"""
import json
from pathlib import Path

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(**kwargs):
        return lambda function: function


VESSEL_NAMES = ('CaAr', 'CaVe', 'BrAr', 'BrVe', 'CeAr', 'CeVe', 'FeAr',
                'FeVe', 'PuAr', 'PuVe', 'AoBrAr', 'BrArCaAr', 'BrArCeAr',
                'CeArFeAr', 'VcBrVe', 'BrVeCaVe', 'BrVeCeVe', 'CeVeFeVe')
NODE_NAMES = VESSEL_NAMES[:10]+('Ra', 'La', 'Lv', 'Rv', 'Ao', 'Vc')
VALVE_PROX = np.array([15, 9, 12, 13, 10, 11])
VALVE_DIST = np.array([10, 11, 14, 8, 13, 12])


def waterfall_drop(p_node, p_external, p_source, y_source, y_node):
    """原 dpWf：外压以下的出流加 Starling waterfall 压降。"""
    ps, pn = p_source-p_external, p_node-p_external
    drop = (ps > pn)*np.maximum(0., np.minimum(0., ps)-pn)
    x = 1-y_source/y_node
    return drop/(x+.2*np.exp(-(x/.2)**2))


def artven_mechanics(volume, length, p0, a0, stiffness, rho):
    """原 ArtVenV2p，源代码自身的防塌陷正则保持原值。"""
    area = np.maximum(1e-10, volume/length)
    normalized = area/a0
    exponent = 1+(stiffness/3-2)/1.02
    positive, negative = 1.02*normalized**exponent, -.02/normalized
    pressure = (positive+negative)*p0
    speed = np.sqrt(.25+(positive*exponent-negative)*p0/rho)
    return pressure, rho*speed/area, area


def tube_mechanics(volume, length, p0, a0, wall_area, stiffness, rho,
                   viscosity, frequency, flow, u_prox, u_dist):
    """原 TubeV2p：壁力学、流速依赖阻抗及 Womersley 修正。"""
    normalized = np.maximum(volume/length/wall_area, .1)
    area = normalized*wall_area
    half_poiseuille = 4*np.pi*viscosity*length/area**2
    exponent = stiffness/3-1
    low = np.maximum(0., 1/normalized-1)
    high_pressure = ((normalized+.5)/(a0/wall_area+.5))**exponent
    pressure = p0*(high_pressure-low**2)
    derivative = p0*(high_pressure*exponent/(normalized+.5)+2*low/normalized**2)/wall_area
    speed = np.sqrt(.25+derivative*area/rho)
    womersley = np.sqrt(area*(2*frequency*rho/viscosity))
    speed = speed/(1+.71/womersley+.71/womersley**4)
    attenuation = frequency*6.3*np.sqrt(1+.03125*womersley**2)/(1+.25*womersley**2)
    impedance = rho*speed/area
    half_velocity = .5*flow/area/speed
    base = np.sqrt(1+half_velocity**2)
    right, left = base+half_velocity, base-half_velocity
    zr, zl = impedance/right, impedance/left
    cr, cl = np.sqrt(.25+(speed*right)**2), np.sqrt(.25+(speed*left)**2)
    return {'pTrans': pressure, 'A': area, 'ZR': zr, 'ZL': zl, 'cR': cr,
            'cL': cl, 'Att': attenuation, 'pSProx': u_prox*np.sqrt(zr)+flow*half_poiseuille,
            'pSDist': u_dist*np.sqrt(zl)-flow*half_poiseuille}


@njit(cache=True, fastmath=False)
def _network_core(volumes, heart_p, heart_z, heart_a, valveq, valveareas,
                  ppl, fac, rho, viscosity, frequency, av, tb, p0av, q0av,
                  thorax_cavities, thorax_tubes, tube_prox, tube_dist,
                  flow, u_prox, u_dist):
    """与上方可读NumPy公式一致的逐元素核；不省略任何网络或物理分支。"""
    for value in volumes:
        if not np.isfinite(value) or value <= 0.:
            raise ValueError('18个血管物理容积必须为有限正值，不裁剪ODE状态')
    if not np.isfinite(fac) or fac <= 0.:
        raise ValueError('FacpControl必须为有限正值')
    for j in range(4):
        if not np.isfinite(heart_p[j]) or not np.isfinite(heart_z[j]) or heart_z[j] <= 0. or not np.isfinite(heart_a[j]):
            raise ValueError('心腔端口必须有限且阻抗为正')
    # av行: Len,p0Ar,p0Ve,A0Ar,A0Ve,kAr,kVe,kAV。
    artven = np.empty((3, 10))
    cavity = np.empty((3, 14))  # pressure, impedance, area
    for j in range(10):
        pair, side = j//2, j % 2
        area = max(1e-10, volumes[j]/av[0, pair])
        normalized = area/av[3+side, pair]
        exponent = 1+(av[5+side, pair]/3-2)/1.02
        positive, negative = 1.02*normalized**exponent, -.02/normalized
        pressure = (positive+negative)*av[1+side, pair]
        speed = np.sqrt(.25+(positive*exponent-negative)*av[1+side, pair]/rho)
        artven[0, j], artven[1, j], artven[2, j] = pressure, rho*speed/area, area
        cavity[0, j], cavity[1, j], cavity[2, j] = pressure, rho*speed/area, area
    for j in thorax_cavities:
        if j < 10:
            cavity[0, j] += ppl
    for j in range(4):
        cavity[0, 10+j], cavity[1, 10+j], cavity[2, 10+j] = heart_p[j], heart_z[j], heart_a[j]
    tube = np.empty((13, 8))
    # tube行: pTrans,A,ZR,ZL,cR,cL,Att,pSProx,pSDist,p,qProx,qDist,VDot。
    for j in range(8):
        length, p0, a0, aw, stiffness = tb[0, j], tb[1, j], tb[2, j], tb[3, j], tb[4, j]
        normalized = max(volumes[10+j]/length/aw, .1)
        area = normalized*aw
        half_poiseuille = 4*np.pi*viscosity*length/area**2
        exponent = stiffness/3-1
        low = max(0., 1/normalized-1)
        high = ((normalized+.5)/(a0/aw+.5))**exponent
        pressure = p0*(high-low**2)
        derivative = p0*(high*exponent/(normalized+.5)+2*low/normalized**2)/aw
        speed = np.sqrt(.25+derivative*area/rho)
        womersley = np.sqrt(area*(2*frequency*rho/viscosity))
        speed /= 1+.71/womersley+.71/womersley**4
        attenuation = frequency*6.3*np.sqrt(1+.03125*womersley**2)/(1+.25*womersley**2)
        impedance = rho*speed/area
        half_velocity = .5*flow[j]/area/speed
        base = np.sqrt(1+half_velocity**2)
        right, left = base+half_velocity, base-half_velocity
        zr, zl = impedance/right, impedance/left
        tube[0, j], tube[1, j], tube[2, j], tube[3, j] = pressure, area, zr, zl
        tube[4, j], tube[5, j] = np.sqrt(.25+(speed*right)**2), np.sqrt(.25+(speed*left)**2)
        tube[6, j] = attenuation
        tube[7, j] = u_prox[j]*np.sqrt(zr)+flow[j]*half_poiseuille
        tube[8, j] = u_dist[j]*np.sqrt(zl)-flow[j]*half_poiseuille
        tube[9, j] = pressure
    for j in thorax_tubes:
        tube[7, j] += ppl
        tube[8, j] += ppl
        tube[9, j] += ppl
    nodes = np.zeros((4, 16))  # pressure, area, conductance, source_flow
    for j in range(14):
        nodes[1, j], nodes[2, j], nodes[3, j] = cavity[2, j], 1/cavity[1, j], cavity[0, j]/cavity[1, j]
    qav = np.empty(5)
    volume_source = 0.
    for j in range(5):
        dp = cavity[0, 2*j]-cavity[0, 2*j+1]
        dpv20 = p0av[j]*av[0, j]**2/(av[3, j]**-2+av[4, j]**-2)
        dpv2 = abs(dp)/(volumes[2*j]**-2+volumes[2*j+1]**-2)
        qav[j] = (dpv2/dpv20)**av[7, j]*np.sign(dp)*q0av[j]
        nodes[3, 2*j] -= qav[j]*fac
        nodes[3, 2*j+1] += qav[j]/fac
        volume_source += qav[j]
    # 与NumPy参考相同，先遍历全部prox，再遍历全部dist，以保留累加顺序。
    for j in range(6):
        nodes[3, VALVE_PROX[j]] -= valveq[j]
        nodes[1, VALVE_PROX[j]] += valveareas[j]
    for j in range(6):
        nodes[3, VALVE_DIST[j]] += valveq[j]
        nodes[1, VALVE_DIST[j]] += valveareas[j]
    for side in range(2):
        for j in range(8):
            node = tube_prox[j] if side == 0 else tube_dist[j]
            admittance = 1/tube[2+side, j]
            nodes[2, node] += admittance
            nodes[3, node] += tube[7+side, j]*admittance
            nodes[1, node] += tube[1, j]
    for j in range(16):
        nodes[0, j] = nodes[3, j]/nodes[2, j]
    waterfall = np.empty(16)
    for side in range(2):
        for j in range(8):
            node = tube_prox[j] if side == 0 else tube_dist[j]
            pext = tube[9, j]-tube[0, j]
            ps, pn = tube[7+side, j]-pext, nodes[0, node]-pext
            drop = max(0., min(0., ps)-pn) if ps > pn else 0.
            ys = 1/tube[2+side, j]
            x = 1-ys/nodes[2, node]
            waterfall[8*side+j] = drop/(x+.2*np.exp(-(x/.2)**2))
    for side in range(2):
        for j in range(8):
            node = tube_prox[j] if side == 0 else tube_dist[j]
            nodes[3, node] -= waterfall[8*side+j]/tube[2+side, j]
    for j in range(16):
        nodes[0, j] = nodes[3, j]/nodes[2, j]
    vessels = np.empty((2, 18))  # derivative, pressure
    source_cavity_vdot = np.empty(14)
    for j in range(10):
        source_cavity_vdot[j] = (nodes[0, j]-cavity[0, j])/cavity[1, j]
        vessels[1, j] = cavity[0, j]
    for j in range(4):
        source_cavity_vdot[10+j] = (nodes[0, 10+j]-cavity[0, 10+j])/cavity[1, 10+j]
    for j in range(8):
        qp = (nodes[0, tube_prox[j]]-tube[7, j]+waterfall[j])/tube[2, j]
        qd = (tube[8, j]-waterfall[8+j]-nodes[0, tube_dist[j]])/tube[3, j]
        tube[10, j], tube[11, j], tube[12, j] = qp, qd, qp-qd
        vessels[0, 10+j], vessels[1, 10+j] = qp-qd, tube[9, j]
    # 同一节点守恒方程的稳定求值，避免极小Z时(pNode-pCavity)/Z的消减。
    # 心腔只有瓣膜连接，因而精确等价于6瓣净流；不改压力/阻抗/流量物理。
    cavity_vdot = np.zeros(14)
    for j in range(6):
        if VALVE_PROX[j] < 14:
            cavity_vdot[VALVE_PROX[j]] -= valveq[j]
    for j in range(6):
        if VALVE_DIST[j] < 14:
            cavity_vdot[VALVE_DIST[j]] += valveq[j]
    for j in range(5):
        cavity_vdot[2*j] -= qav[j]*fac
        cavity_vdot[2*j+1] += qav[j]/fac
    for j in range(8):
        if tube_prox[j] < 14:
            cavity_vdot[tube_prox[j]] -= tube[10, j]
    for j in range(8):
        if tube_dist[j] < 14:
            cavity_vdot[tube_dist[j]] += tube[11, j]
    for j in range(10):
        vessels[0, j] = cavity_vdot[j]
    heart_vdot = cavity_vdot[10:].copy()
    valve_dp = np.empty(6)
    for j in range(6):
        valve_dp[j] = nodes[0, VALVE_PROX[j]]-nodes[0, VALVE_DIST[j]]
    return nodes, vessels, heart_vdot, valve_dp, cavity, qav, artven, tube, waterfall, volume_source*(1/fac-fac), source_cavity_vdot


@njit(cache=True, fastmath=False)
def _delay_core(t, dt, period, length, tube, q, pleft, pright, uprox, udist, tau_l, tau_r, transmural):
    """原TubeDelays和Fu，按Tube列独立更新；当前/下一格保持双写。"""
    nt = len(q)
    slot = int(np.floor(t/dt+.5)) % nt
    next_slot = (slot+1) % nt
    for j in range(8):
        tau_l[j] = max(dt, min(.9*period, tau_l[j]+dt*(1-tau_l[j]*tube[5, j]/length[j])))
        tau_r[j] = max(dt, min(.9*period, tau_r[j]+dt*(1-tau_r[j]*tube[4, j]/length[j])))
        pos_l = ((t-tau_l[j]) % period/dt) % (nt-1)
        pos_r = ((t-tau_r[j]) % period/dt) % (nt-1)
        il, ir = int(np.floor(pos_l)), int(np.floor(pos_r))
        al, ar = pos_l-il, pos_r-ir
        pp = (1-al)*pleft[il, j]+al*pleft[(il+1) % (nt-1), j]
        pd = (1-ar)*pright[ir, j]+ar*pright[(ir+1) % (nt-1), j]
        flow = q[slot, j]
        zr, zl = tube[2, j], tube[3, j]
        total_z = zr+zl
        mean_pressure = max(0., tube[0, j])
        pp0, pd0 = mean_pressure-flow*zr, mean_pressure+flow*zl
        external = tube[9, j]-tube[0, j] if transmural else 0.
        left = (tube[8, j]-external)-tube[11, j]*total_z
        right = (tube[7, j]-external)+tube[10, j]*total_z
        dq = .5*dt*((-tube[7, j]+tube[8, j])/total_z-flow)/(tau_l[j]+tau_r[j])
        up = (np.exp(-tau_l[j]*tube[6, j])*(pp-pp0)+pp0)/np.sqrt(zr)
        ud = (np.exp(-tau_r[j]*tube[6, j])*(pd-pd0)+pd0)/np.sqrt(zl)
        for target in (slot, next_slot):
            q[target, j], pleft[target, j], pright[target, j] = flow+dq, left, right
            uprox[target, j], udist[target, j] = up, ud


class OriginalCirculation:
    """5 ArtVen、8 Tube、16 节点的独立网络。

    evaluate 的压力、容积、流量单位依次为 Pa、m³、m³/s。FacpControl
    仅作为显式初始化标定输入；冻结阶段必须为1。p0av/q0av 是可标定原数组。
    容积导数按节点连接流平衡求值，避免强屈曲极小阻抗放大压力消减误差；
    同方程的原始压力差求值保留在 source_cavity_vdot 诊断中。
    默认启用不使用fastmath的Numba核；use_numba=False保留逐式NumPy实现。
    wave_pressure_mode默认native_absolute以复现源代码；transmural将入射反射波
    历史改用跨壁基准，防止pCavity已加的外压在下一次传播时再次加入。
    后一模式借用PRef波历史作初猜，需重新达到周期稳态，不能视为精确初始化。
    """
    state_size = 18
    history_fields = ('q', 'pL', 'pR', 'uP', 'uD')

    def __init__(self, parameters=None, dt=.0005, use_numba=True,
                 wave_pressure_mode='native_absolute'):
        if wave_pressure_mode not in ('native_absolute', 'transmural'):
            raise ValueError('wave_pressure_mode必须为native_absolute或transmural')
        self.wave_pressure_mode = wave_pressure_mode
        self.history_initialization = ('PRef_native_history_as_initial_guess' if
            wave_pressure_mode == 'transmural' else 'PRef_native_history')
        if parameters is None:
            parameters = Path(__file__).parent.parent/'data'/'original_circulation.json'
        if isinstance(parameters, (str, Path)):
            parameters = json.loads(Path(parameters).read_text(encoding='utf-8'))
        self.parameters = parameters
        general = parameters['General']
        self.period_s = float(general['tCycle'])
        self.dt = float(dt)
        self.rho, self.viscosity = float(general['RhoB']), float(general['EtaB'])
        self.frequency = 2.5/float(general['tCycleRest'])
        av, tb = parameters['ArtVen'], parameters['Tube']
        self.av = {key: np.asarray(av[key], dtype=float) for key in
                   ('Len', 'p0', 'A0', 'k', 'p0AV', 'q0AV', 'kAV')}
        self.p0av, self.q0av = self.av['p0AV'], self.av['q0AV']
        self.tb = {key: np.asarray(tb[key], dtype=float) for key in ('Len', 'p0', 'A0', 'AWall', 'k')}
        self.use_numba = bool(use_numba and NUMBA_AVAILABLE)
        self._av_coefficients = np.vstack((self.av['Len'], self.av['p0'], self.av['A0'], self.av['k'], self.av['kAV']))
        self._tube_coefficients = np.vstack([self.tb[key] for key in ('Len', 'p0', 'A0', 'AWall', 'k')])
        self.tube_prox = np.asarray(tb['iNodeProx'], dtype=int)
        self.tube_dist = np.asarray(tb['iNodeDist'], dtype=int)
        self.valve_area = np.maximum(parameters['Valve']['AOpen'], parameters['Valve']['ALeak'])
        thorax = parameters['Thorax']
        self.thorax_cavities = np.asarray(thorax.get('cavity_indices0', thorax.get('iCavity', [])), dtype=int)
        self.thorax_tubes = np.asarray(thorax.get('tube_indices0', thorax.get('iTube', [])), dtype=int)
        self._initial = np.r_[parameters['initial']['cavity_V'][:10], parameters['initial']['tube_V']]
        history = parameters['history']
        self.TauL, self.TauR = np.asarray(history['TauL'], dtype=float).copy(), np.asarray(history['TauR'], dtype=float).copy()
        self._resample_history(history, np.asarray(history['t'], dtype=float))
        self.beat_index = 0

    def set_reference_geometry(self, *, av_p0, av_a0, tube_p0, tube_a0):
        """初始化适应原参考压力/截面积；先验证全部输入，再同步两种内核。

        仅更新ArtVen和Tube的p0/A0，不改变微循环p0AV/q0AV、材料、长度、
        壁截面积或传播历史。检查点应独立保存这四个参考几何数组。
        """
        values = {}
        for name, value, shape in (
            ('av_p0', av_p0, (2, 5)), ('av_a0', av_a0, (2, 5)),
            ('tube_p0', tube_p0, (8,)), ('tube_a0', tube_a0, (8,))):
            array = np.asarray(value, dtype=float)
            if array.shape != shape or not np.isfinite(array).all() or np.any(array <= 0):
                raise ValueError(f'参考几何{name}须为形状{shape}的有限正值数组')
            # 输入可能就是当前内部数组的视图；全部复制后才允许任何写入。
            values[name] = array.copy()
        self.av['p0'][:] = values['av_p0']
        self.av['A0'][:] = values['av_a0']
        self.tb['p0'][:] = values['tube_p0']
        self.tb['A0'][:] = values['tube_a0']
        self._av_coefficients[1:3] = values['av_p0']
        self._av_coefficients[3:5] = values['av_a0']
        self._tube_coefficients[1] = values['tube_p0']
        self._tube_coefficients[2] = values['tube_a0']

    def export_reference_geometry(self):
        """导出四个独立副本，SI：压力Pa，截面积m²。"""
        return dict(av_p0=self.av['p0'].copy(), av_a0=self.av['A0'].copy(),
                    tube_p0=self.tb['p0'].copy(), tube_a0=self.tb['A0'].copy())

    def initial_state(self):
        return self._initial.copy()

    def _grid(self):
        count = int(np.floor(self.period_s/self.dt+.5))
        if not np.isclose(count*self.dt, self.period_s, rtol=0., atol=1e-11):
            raise ValueError('心搏周期必须是固定积分步长的整数倍')
        return np.arange(count+1)*self.dt

    def _resample_history(self, history, old_time):
        self.history_time = self._grid()
        # TimingInit/interp0 的上次周期终点平移；端点外采用最近端点。
        old_time = old_time.copy()
        if old_time[-1] != self.period_s:
            old_time += self.period_s-old_time[-1]
        old_time[0] = min(0., old_time[0])
        for name in self.history_fields:
            signal = np.asarray(history[name], dtype=float)
            if signal.shape != (len(old_time), 8) or not np.isfinite(signal).all():
                raise ValueError('Tube历史格式或有限值错误: '+name)
            value = signal.copy() if np.array_equal(self.history_time, old_time) else np.column_stack(
                [np.interp(self.history_time, old_time, signal[:, j]) for j in range(8)])
            setattr(self, name, value)

    def start_beat(self, period_s=None, dt=None):
        """下一搏保留上一搏历史；周期或步长改变时依 TimingInit 重采样。"""
        old_time = self.history_time.copy()
        previous = {name: getattr(self, name) for name in self.history_fields}
        if period_s is not None:
            self.period_s = float(period_s)
        if dt is not None:
            self.dt = float(dt)
        if self.period_s != old_time[-1] or not np.isclose(self.dt, old_time[1]-old_time[0], rtol=0., atol=1e-15):
            self._resample_history(previous, old_time)
        self.beat_index += 1

    def export_delay_state(self):
        """完整可独立保存的波传播状态（数组均为副本）。"""
        return {**{name: getattr(self, name).copy() for name in self.history_fields},
                'TauL': self.TauL.copy(), 'TauR': self.TauR.copy(),
                't': self.history_time.copy(), 'dt': self.dt,
                'period_s': self.period_s, 'beat_index': self.beat_index,
                'wave_pressure_mode': self.wave_pressure_mode,
                'history_initialization': self.history_initialization}

    def restore_delay_state(self, state):
        """恢复检查点；若目标实例步长不同，则保留周期并重采样历史。"""
        # 缺省标记只代表旧版native检查点，不能悄悄解释为跨壁历史。
        if state.get('wave_pressure_mode', 'native_absolute') != self.wave_pressure_mode:
            raise ValueError('检查点波压力基准与目标模式不同，禁止隐式转换')
        if not np.isclose(float(state.get('period_s', self.period_s)), self.period_s):
            raise ValueError('检查点心搏周期与目标实例不同')
        self._resample_history(state, np.asarray(state['t'], dtype=float))
        for name in ('TauL', 'TauR'):
            value = np.asarray(state[name], dtype=float)
            if value.shape != (8,) or not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError('传播延迟无效: '+name)
            setattr(self, name, value.copy())
        self.beat_index = int(state.get('beat_index', 0))
        self.history_initialization = state.get('history_initialization', 'checkpoint')

    def _slot(self, t_local):
        # MATLAB round 对非负时间为半格向上，保留周期终点而非提前取模。
        return int(np.floor(float(t_local)/self.dt+.5)) % len(self.history_time)

    def evaluate(self, t_local, vessel_volumes, heart_pressure, source_impedance,
                 cavity_area, valveq, valveareas=None, ppl=0., facp_control=1.):
        """只读计算；heart_pressure 已含心包及胸膜外压，避免重复叠加。"""
        if self.use_numba:
            return self._evaluate_compiled(t_local, vessel_volumes, heart_pressure,
                source_impedance, cavity_area, valveq, valveareas, ppl, facp_control)
        volumes = np.asarray(vessel_volumes, dtype=float)
        if volumes.shape != (18,) or not np.isfinite(volumes).all() or np.any(volumes <= 0):
            raise ValueError('18个血管物理容积必须为有限正值，不裁剪ODE状态')
        if not np.isfinite(facp_control) or facp_control <= 0:
            raise ValueError('FacpControl必须为有限正值')
        av = self.av
        flat = lambda value: np.asarray(value).ravel(order='F')
        pa, za, aa = artven_mechanics(volumes[:10], np.repeat(av['Len'], 2),
                                    flat(av['p0']), flat(av['A0']), flat(av['k']), self.rho)
        pc = np.r_[pa, np.asarray(heart_pressure, dtype=float)]
        vascular_thorax = self.thorax_cavities[self.thorax_cavities < 10]
        pc[vascular_thorax] += ppl
        zc = np.r_[za, np.asarray(source_impedance, dtype=float)]
        ac = np.r_[aa, np.asarray(cavity_area, dtype=float)]
        if pc.shape != (14,) or zc.shape != (14,) or ac.shape != (14,) or np.any(zc <= 0):
            raise ValueError('心腔端口维度或阻抗错误')
        it = self._slot(t_local)
        tb = self.tb
        tube = tube_mechanics(volumes[10:], tb['Len'], tb['p0'], tb['A0'], tb['AWall'],
            tb['k'], self.rho, self.viscosity, self.frequency, self.q[it], self.uP[it], self.uD[it])
        pext = np.zeros(8)
        pext[self.thorax_tubes] = ppl
        tube['p'] = tube['pTrans']+pext
        tube['pSProx'] += pext
        tube['pSDist'] += pext
        dp_av = pc[0:10:2]-pc[1:10:2]
        dpv20 = self.p0av*av['Len']**2/np.sum(av['A0']**-2, axis=0)
        dpv2 = np.abs(dp_av)/(volumes[0:10:2]**-2+volumes[1:10:2]**-2)
        qav = (dpv2/dpv20)**av['kAV']*np.sign(dp_av)*self.q0av
        yn, qn, an = np.zeros(16), np.zeros(16), np.zeros(16)
        yc = 1/zc
        yn[:14], qn[:14], an[:14] = yc, pc*yc, ac
        qn[0:10:2] -= qav*facp_control
        qn[1:10:2] += qav/facp_control
        areas = self.valve_area if valveareas is None else np.asarray(valveareas, dtype=float)
        np.add.at(qn, VALVE_PROX, -np.asarray(valveq))
        np.add.at(qn, VALVE_DIST, np.asarray(valveq))
        np.add.at(an, VALVE_PROX, areas)
        np.add.at(an, VALVE_DIST, areas)
        nodes = np.r_[self.tube_prox, self.tube_dist]
        yt = 1/np.r_[tube['ZR'], tube['ZL']]
        ps = np.r_[tube['pSProx'], tube['pSDist']]
        np.add.at(yn, nodes, yt)
        np.add.at(qn, nodes, ps*yt)
        np.add.at(an, nodes, np.tile(tube['A'], 2))
        pn = qn/yn
        correction = waterfall_drop(pn[nodes], np.tile(pext, 2), ps, yt, yn[nodes])
        np.add.at(qn, nodes, -yt*correction)
        corrected = ps-correction
        pn = qn/yn
        source_cavity_vdot = (pn[:14]-pc)*yc
        qp = (pn[self.tube_prox]-corrected[:8])/tube['ZR']
        qd = (corrected[8:]-pn[self.tube_dist])/tube['ZL']
        # 从同一节点方程累计已解出的连接流，消除强屈曲心房极小Z的数值消减。
        vdot_cav = np.zeros(16)
        np.add.at(vdot_cav, VALVE_PROX, -np.asarray(valveq))
        np.add.at(vdot_cav, VALVE_DIST, np.asarray(valveq))
        vdot_cav[0:10:2] -= qav*facp_control
        vdot_cav[1:10:2] += qav/facp_control
        np.add.at(vdot_cav, self.tube_prox, -qp)
        np.add.at(vdot_cav, self.tube_dist, qd)
        tube['qProx'], tube['qDist'], tube['VDot'] = qp, qd, qp-qd
        tube['waterfall_dp'] = correction
        return {'node_pressure': pn, 'node_area': an, 'node_conductance': yn,
                'node_source_flow': qn, 'vessel_derivative': np.r_[vdot_cav[:10], qp-qd],
                'heart_vdot': vdot_cav[10:14], 'valve_dp': pn[VALVE_PROX]-pn[VALVE_DIST],
                'source_cavity_vdot': source_cavity_vdot,
                'cavity_pressure': pc, 'cavity_impedance': zc, 'cavity_area': ac,
                'vessel_pressure': np.r_[pc[:10], tube['p']], 'artven_flow': qav,
                'qAv': qav, 'facp_volume_source': float(np.sum(qav)*(1/facp_control-facp_control)),
                'artven': {'pTrans': pa, 'Z': za, 'A': aa}, 'tube': tube}

    def _evaluate_compiled(self, t_local, vessel_volumes, heart_pressure,
                           source_impedance, cavity_area, valveq, valveareas,
                           ppl, facp_control):
        arrays = (np.asarray(vessel_volumes, dtype=float), np.asarray(heart_pressure, dtype=float),
                  np.asarray(source_impedance, dtype=float), np.asarray(cavity_area, dtype=float),
                  np.asarray(valveq, dtype=float), self.valve_area if valveareas is None else np.asarray(valveareas, dtype=float))
        if any(array.shape != (length,) for array, length in zip(arrays, (18, 4, 4, 4, 6, 6))):
            raise ValueError('18血管/4心腔/6瓣膜网络端口维度错误')
        it = self._slot(t_local)
        node, vessel, heart_vdot, valve_dp, cavity, qav, artven, tube_array, waterfall, volume_source, source_cavity_vdot = _network_core(
            *arrays, float(ppl), float(facp_control), self.rho, self.viscosity, self.frequency,
            self._av_coefficients, self._tube_coefficients, self.p0av, self.q0av,
            self.thorax_cavities, self.thorax_tubes, self.tube_prox, self.tube_dist,
            self.q[it], self.uP[it], self.uD[it])
        tube_keys = ('pTrans', 'A', 'ZR', 'ZL', 'cR', 'cL', 'Att', 'pSProx',
                     'pSDist', 'p', 'qProx', 'qDist', 'VDot')
        tube = {key: tube_array[j] for j, key in enumerate(tube_keys)}
        tube['waterfall_dp'] = waterfall
        return {'node_pressure': node[0], 'node_area': node[1], 'node_conductance': node[2],
                'node_source_flow': node[3], 'vessel_derivative': vessel[0],
                'heart_vdot': heart_vdot, 'valve_dp': valve_dp, 'cavity_pressure': cavity[0],
                'source_cavity_vdot': source_cavity_vdot,
                'cavity_impedance': cavity[1], 'cavity_area': cavity[2],
                'vessel_pressure': vessel[1], 'artven_flow': qav, 'qAv': qav,
                'facp_volume_source': volume_source,
                'artven': {'pTrans': artven[0], 'Z': artven[1], 'A': artven[2]},
                'tube': tube, '_tube_core': tube_array}

    def _delayed(self, signal, time):
        # 原 Fu 把最后一行临时当作首行；不改写存储缓冲。
        position = np.mod(time/self.dt, len(signal)-1)
        index = np.floor(position).astype(int)
        fraction = position-index
        columns = np.arange(8)
        following = (index+1) % (len(signal)-1)
        return (1-fraction)*signal[index, columns]+fraction*signal[following, columns]

    def update_delay(self, t_local, result):
        """原 TubeDelays；每次推进RHS调用一次，evaluate和输出采样不能调用。"""
        if self.use_numba and '_tube_core' in result:
            _delay_core(float(t_local), self.dt, self.period_s, self.tb['Len'],
                        result['_tube_core'], self.q, self.pL, self.pR, self.uP, self.uD,
                        self.TauL, self.TauR, self.wave_pressure_mode == 'transmural')
            return
        tube = result['tube']
        self.TauL = np.maximum(self.dt, np.minimum(.9*self.period_s,
            self.TauL+self.dt*(1-self.TauL*tube['cL']/self.tb['Len'])))
        self.TauR = np.maximum(self.dt, np.minimum(.9*self.period_s,
            self.TauR+self.dt*(1-self.TauR*tube['cR']/self.tb['Len'])))
        pp = self._delayed(self.pL, np.mod(t_local-self.TauL, self.period_s))
        pd = self._delayed(self.pR, np.mod(t_local-self.TauR, self.period_s))
        it = self._slot(t_local)
        q = self.q[it].copy()
        zr, zl = tube['ZR'], tube['ZL']
        total_z = zr+zl
        mean_pressure = np.maximum(0., tube['pTrans'])
        pp0, pd0 = mean_pressure-q*zr, mean_pressure+q*zl
        external = tube['p']-tube['pTrans'] if self.wave_pressure_mode == 'transmural' else 0.
        left = (tube['pSDist']-external)-tube['qDist']*total_z
        right = (tube['pSProx']-external)+tube['qProx']*total_z
        dq = .5*self.dt*((-tube['pSProx']+tube['pSDist'])/total_z-q)/(self.TauL+self.TauR)
        slots = [it, (it+1) % len(self.q)]
        self.q[slots] = q+dq
        self.pL[slots], self.pR[slots] = left, right
        self.uP[slots] = (np.exp(-self.TauL*tube['Att'])*(pp-pp0)+pp0)/np.sqrt(zr)
        self.uD[slots] = (np.exp(-self.TauR*tube['Att'])*(pd-pd0)+pd0)/np.sqrt(zl)
