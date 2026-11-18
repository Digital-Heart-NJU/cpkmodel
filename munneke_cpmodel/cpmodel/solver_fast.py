"""保留原OdeCA操作顺序和波历史更新的Numba融合心肺推进器。"""
import numpy as np
from numba import njit
from .heart import valve_derivative
from .heart_fast import _evaluate_core
from .original_circulation import _network_core, _delay_core, VALVE_PROX, VALVE_DIST
from .respiratory.propagator import evaluate_respiration_fast
from .settings import PA

HEART_NODES = np.array([10, 11, 12, 13, 15, 9, 14, 8])


@njit(cache=True)
def _valve_acceleration(q, nodes, valve, rho, dt):
    out = np.empty(6)
    for j in range(6):
        drop = nodes[0, VALVE_PROX[j]]-nodes[0, VALVE_DIST[j]]
        aopen, aleak = valve[0, j], valve[1, j]
        opening = max(aopen, aleak)
        aprox = nodes[1, VALVE_PROX[j]]-opening
        adist = nodes[1, VALVE_DIST[j]]-opening
        bwq = 1. if q[j] < 0. else 0.
        forward = 0. if q[j] < 0. and drop < 0. else 1.
        area = aleak+forward*(aopen-aleak)
        resistance = 1.+rho*abs(q[j])*abs(area**-2-1/((1-bwq)*aprox**2+bwq*adist**2))
        dpb = .5*q[j]*resistance
        inertia = 1.5*rho*(valve[2, j]/area+.5*(1/np.sqrt(aprox)+1/np.sqrt(adist)))
        tau = inertia/resistance
        out[j] = (1-1/(1+dt/tau))*(drop-dpb)/(dt*resistance)
    return out


@njit(cache=True)
def _rhs(t, state, ppl, dt, fac, heart_parameters, network_parameters, buffers,
         length, period, transmural, update):
    for i in range(70):
        if not np.isfinite(state[i]):
            raise ValueError('联合模型需要有限心血管状态')
    for i in range(22):
        if state[i] <= 0.:
            raise ValueError('血管和心腔容积须正')
    if state[23] <= 0.:
        raise ValueError('TriSeg交界半径须正')
    for i in range(24, 44):
        if state[i] < 0.:
            raise ValueError('C须非负')
    for i in range(44, 64):
        if state[i] <= 0.:
            raise ValueError('Lsi须正')
    p, ptw, dead, vw, delay, hp, hrho, p0, bag, valve, ports = heart_parameters
    heart = _evaluate_core(t, state[18:70], np.zeros(4), ppl, dt, p, ptw,
                           dead, vw, delay, hp, hrho, p0, bag, valve, ports)
    rho, viscosity, frequency, av, tb, p0av, q0av, tc, tt, prox, dist = network_parameters
    q, pl, pr, up, ud, tl, tr = buffers
    slot = int(np.floor(t/dt+.5)) % len(q)
    areas = np.maximum(valve[0], valve[1])
    # Python公开API在Pa->mmHg->Pa间往返；保留同一浮点操作顺序。
    network = _network_core(state[:18], (heart[1]/PA)*PA, heart[9], heart[10],
                            state[64:70], areas, ppl*PA, fac, rho, viscosity,
                            frequency, av, tb, p0av, q0av, tc, tt, prox, dist,
                            q[slot], up[slot], ud[slot])
    nodes, vessels, hv, dp, cavity, qav, artven, tube, waterfall, source, source_vdot = network
    derivative = np.empty(70)
    derivative[:18] = vessels[0]
    derivative[18:70] = heart[0]
    derivative[18:22] = hv
    derivative[64:70] = _valve_acceleration(state[64:70], nodes, valve, hrho, dt)
    if not np.isfinite(derivative).all():
        raise FloatingPointError('联合导数产生非有限值，未裁剪状态')
    if update:
        _delay_core(t, dt, period, length, tube, q, pl, pr, up, ud, tl, tr, transmural)
    return derivative, nodes[:2].copy(), qav, tube


@njit(cache=True)
def _beat_core(start, respiratory, ledger, ppl, dt, count, stride, fac, hp, np_,
               buffers, length, period, transmural):
    samples = count//stride+1+(1 if count % stride else 0)
    time, states = np.empty(samples), np.empty((samples, 77))
    node_samples, qav_samples = np.empty((samples, 2, 16)), np.empty((samples, 5))
    tube_samples = np.empty((samples, 13, 8))
    h = start[:70].copy()
    _, nodes, qav, tube = _rhs(0., h, ppl[0], dt, fac, hp, np_, buffers, length,
                         period, transmural, False)
    time[0], states[0] = 0., start
    node_samples[0], qav_samples[0] = nodes, qav
    tube_samples[0] = tube
    sample = 1
    first, _, _, _ = _rhs(0., h, ppl[0], dt, fac, hp, np_, buffers, length,
                        period, transmural, True)
    derivative, _, _, _ = _rhs(dt, h+first*dt, ppl[1], dt, fac, hp, np_, buffers,
                             length, period, transmural, True)
    g = .5*(derivative+first)
    h = h+g*dt
    if stride == 1:
        _, nodes, qav, tube = _rhs(dt, h, ppl[1], dt, fac, hp, np_, buffers, length,
                             period, transmural, False)
        time[sample], states[sample, :70] = dt, h
        states[sample, 70:76], states[sample, 76] = respiratory[1], ledger[1]
        node_samples[sample], qav_samples[sample] = nodes, qav
        tube_samples[sample] = tube
        sample += 1
    for step in range(2, count+1):
        t = step*dt
        derivative, _, _, _ = _rhs(t, h+g*dt, ppl[step], dt, fac, hp, np_, buffers,
                                 length, period, transmural, True)
        g = (g+2*derivative)/3
        h = h+g*dt
        if step % stride == 0 or step == count:
            _, nodes, qav, tube = _rhs(t, h, ppl[step], dt, fac, hp, np_, buffers,
                                 length, period, transmural, False)
            time[sample], states[sample, :70] = t, h
            states[sample, 70:76], states[sample, 76] = respiratory[step], ledger[step]
            node_samples[sample], qav_samples[sample] = nodes, qav
            tube_samples[sample] = tube
            sample += 1
    return time, states, node_samples, qav_samples, tube_samples


def fused_fast_beat(model, start77):
    """返回(time,states,observations)，原位更新完整波缓冲与TauL/R。"""
    start = np.asarray(start77, dtype=float)
    if start.shape != (77,) or not np.isfinite(start).all():
        raise ValueError('联合初态须为77维有限数组')
    dt, period = float(model.dt), float(model.period)
    count = round(period/dt)
    stride = round(model.config['output_interval_s']/dt)
    if count < 1 or stride < 1:
        raise ValueError('心搏与输出间隔须覆盖至少一个积分步')
    v, h = model.vascular, model.heart
    p, bag, valve = h._packed_parameters()
    hp = (p, h.patch_to_wall, h.am_dead, h.wall_volume, h.activation_delay_s,
          h.period_s, h.rho, h.p0, bag, valve, h.port_area)
    np_ = (v.rho, v.viscosity, v.frequency, v._av_coefficients, v._tube_coefficients,
           v.p0av, v.q0av, v.thorax_cavities, v.thorax_tubes, v.tube_prox, v.tube_dist)
    respiratory = model.respiratory_propagator.trajectory(start[70:76], count)
    ledger = start[76]+respiratory[:, :4].sum(axis=1)-start[70:74].sum()
    # 与原肌压函数相同的分段表达，矢量化仅用于预计算独立驱动。
    rp, rb = model.respiratory, model.respiratory_boundary
    q = respiratory[:, 5]-np.floor(respiratory[:, 5])
    beta = 1/(1+rp.inspiratory_expiratory_ratio)
    alpha, amplitude = 1-beta, rb.muscle_pressure_min
    pmus = np.empty(len(q))
    inspiratory = q <= alpha
    pmus[inspiratory] = amplitude*q[inspiratory]*(1-q[inspiratory])/(alpha*beta)
    end = np.exp(-1/rp.expiration_tau_fraction)
    pmus[~inspiratory] = amplitude*(np.exp(-(q[~inspiratory]-alpha)/
        (rp.expiration_tau_fraction*beta))-end)/(-np.expm1(-1/rp.expiration_tau_fraction))
    fraction = model.pleural_fraction
    ppl = fraction*((respiratory[:, 4]/rp.chest_wall_compliance+pmus+
                     rp.pleural_end_expiratory)*.73556)+(1-fraction)*.03340482192495265
    v.start_beat()
    buffers = (v.q, v.pL, v.pR, v.uP, v.uD, v.TauL, v.TauR)
    time, states, nodes, qav, tubes = _beat_core(start, respiratory, ledger, ppl, dt,
        count, stride, float(model.facp_control), hp, np_, buffers, v.tb['Len'],
        v.period_s, v.wave_pressure_mode == 'transmural')
    observations = []
    tube_keys = ('pTrans', 'A', 'ZR', 'ZL', 'cR', 'cL', 'Att', 'pSProx',
                 'pSDist', 'p', 'qProx', 'qDist', 'VDot')
    for i, (t, state) in enumerate(zip(time, states)):
        resp = evaluate_respiration_fast(rp, state[70:76], rb)
        fraction_ppl = fraction*resp.values['ppl_mmhg']+(1-fraction)*.03340482192495265
        cardiac = h.evaluate(float(t), state[18:70], np.zeros(4), fraction_ppl, dt)
        node, area = nodes[i, 0, HEART_NODES], nodes[i, 1, HEART_NODES]
        qdot = valve_derivative(state[64:70], node, area, h.valve['AOpen'],
                                h.valve['ALeak'], h.valve['Len'], h.rho, dt)
        cardiac['derivative'][46:52] = qdot
        cardiac['node_pressure_mmhg'], cardiac['node_area_m2'] = node/PA, area
        cardiac['valve_acceleration_m3_s2'] = qdot
        cardiac['valve_open_fraction'] = 1.-((state[64:70] < 0)&
            (nodes[i, 0, VALVE_PROX]-nodes[i, 0, VALVE_DIST] < 0)).astype(float)
        observations.append((None, cardiac, resp,
                             {'node_pressure': nodes[i, 0], 'artven_flow': qav[i],
                              'tube': {key: tubes[i, j] for j, key in enumerate(tube_keys)}}))
    model._last_beat_samples = (time, states, observations)
    return time, states, observations
