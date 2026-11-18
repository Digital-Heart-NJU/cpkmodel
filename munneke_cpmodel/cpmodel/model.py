"""Munneke20Patch、原ArtVen/Tube传播循环与自主呼吸直接联合积分。"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time as wall_time
import numpy as np
from .heart import MunnekeHeart, valve_derivative
from .original_circulation import OriginalCirculation
from .respiratory.boundary import RespiratoryBoundary
from .respiratory.propagator import RespiratoryPropagator, evaluate_respiration_fast
from .respiratory.parameters import RespiratoryParameters
from .settings import DATA, ROOT, PA, load_parameters, read_json
BACKEND='Munneke2023_python_source'
NETWORK='Munneke_original_ArtVen_Tube_no_coronary'
HEART_NODES=np.array([10,11,12,13,15,9,14,8])


def save_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # Windows/OneDrive可能短暂持有目标文件；只重试文件占用，不吞掉最终I/O错误。
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==19:raise
            wall_time.sleep(min(.02*(attempt+1),.2))


class CardiorespiratoryModel:
    """77态；血液容积统一m³，波传播记忆显式保存在vascular中。"""
    heart_slice=slice(18,70)
    respiratory_slice=slice(70,76)
    bookkeeping_index=76
    state_size=77
    physical_states=76

    def __init__(self,dt,use_calibrated=True,fast=True):
        self.config=load_parameters();self.period=float(self.config['period_s'])
        if self.config['heart_backend']!=BACKEND:raise ValueError('配置必须明确使用Munneke2023源码心脏')
        if not np.isfinite(dt) or dt<=0 or not np.isclose(round(self.period/dt)*dt,self.period,atol=1e-12,rtol=0):
            raise ValueError('步长须为有限正数并整分心周期')
        if not np.isclose(round(self.config['output_interval_s']/dt)*dt,self.config['output_interval_s'],atol=1e-12,rtol=0):
            raise ValueError('输出间隔须为积分步长的整数倍')
        self.dt=float(dt)
        if fast:
            from .heart_fast import FastMunnekeHeart
            self.heart=FastMunnekeHeart()
        else:self.heart=MunnekeHeart()
        self.vascular=OriginalCirculation(dt=self.dt,wave_pressure_mode='transmural')
        self._native_geometry=self.vascular.export_reference_geometry()
        self.respiratory=RespiratoryParameters(**read_json(DATA/'respiratory_parameters.json')['mechanics'])
        self.respiratory_boundary=RespiratoryBoundary(respiratory_rate=self.config['respiratory_rate_min'],
                             muscle_pressure_min=self.config['muscle_pressure_min_cmh2o'])
        self.respiratory_propagator=RespiratoryPropagator(self.respiratory,self.respiratory_boundary,self.dt)
        self.facp_control=1.;self.pleural_fraction=1.;self.use_calibrated=bool(use_calibrated)
        self.calibrated_reference_loaded=False
        if 'circadapt' in sys.modules:raise RuntimeError('直接源码运行不应加载circadapt模块')

    def initial_state(self):
        suffix='_250us' if np.isclose(self.dt,.00025,rtol=0,atol=1e-14) else ''
        path=DATA/('calibrated_reference'+suffix+'.npz')
        if self.use_calibrated and (DATA/'calibrated_reference.npz').exists():
            if not any(np.isclose(self.dt,x,rtol=0,atol=1e-14) for x in (.0005,.00025)) or not path.is_file():
                raise ValueError('当前步长没有独立验收的冻结参考；可显式使用未标定原初猜')
        if self.use_calibrated and path.is_file():
            metadata=read_json(path.with_suffix('.json'))
            if metadata['physics_signature']!=physical_signature(include_reference=False):
                raise ValueError('标定参考的方程或参数身份已改变，必须重新标定')
            if metadata['reference_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('标定参考文件身份不符')
            if metadata.get('accepted') is not True:
                raise ValueError('标定参考未通过预定验收')
            if not np.isclose(metadata['dt_s'],self.dt,rtol=0,atol=1e-14):
                raise ValueError('冻结参考的积分步长不一致')
            with np.load(path,allow_pickle=False) as z:
                self.restore_auxiliary(z)
                if self.facp_control!=1. or self.pleural_fraction!=1.:
                    raise ValueError('正式初态必须冻结校正并使用完整自主呼吸')
                self.calibrated_reference_loaded=True
                return z['state'].copy()
        respiration=np.array(read_json(DATA/'initial_state.json')['respiratory_state'],dtype=float)
        respiration[5]=0.
        heart=self.heart.initial_state()
        heart[46:52]=self.vascular.parameters['initial']['valve_q']
        return np.r_[self.vascular.initial_state(),heart,respiration,0.]

    def evaluate_full(self,time,state):
        state=np.asarray(state,dtype=float)
        if state.shape!=(77,) or not np.isfinite(state).all() or np.any(state[:22]<=0):
            raise ValueError('联合模型需要77个有限状态及正血管/心腔容积')
        respiration=evaluate_respiration_fast(self.respiratory,state[70:76],self.respiratory_boundary)
        ppl=self.pleural_fraction*respiration.values['ppl_mmhg']+(1-self.pleural_fraction)*.03340482192495265
        cardiac=self.heart.evaluate(time,state[18:70],np.zeros(4),ppl,self.dt)
        network=self.vascular.evaluate(time,state[:18],cardiac['pressure_mmhg']*PA,
            cardiac['source_impedance_pa_s_m3'],cardiac['cavity_area_m2'],state[64:70],
            np.maximum(self.heart.valve['AOpen'],self.heart.valve['ALeak']),ppl=ppl*PA,
            facp_control=self.facp_control)
        node=network['node_pressure'][HEART_NODES];area=network['node_area'][HEART_NODES]
        qdot=valve_derivative(state[64:70],node,area,self.heart.valve['AOpen'],
                             self.heart.valve['ALeak'],self.heart.valve['Len'],self.heart.rho,self.dt)
        dh=cardiac['derivative'].copy();dh[:4]=network['heart_vdot'];dh[46:52]=qdot
        cardiac['derivative']=dh;cardiac['node_pressure_mmhg']=node/PA;cardiac['node_area_m2']=area
        cardiac['valve_acceleration_m3_s2']=qdot
        cardiac['valve_open_fraction']=1.-((state[64:70]<0)&(network['valve_dp']<0)).astype(float)
        derivative=np.r_[network['vessel_derivative'],dh,respiration.derivative,respiration.values['q_mouth']]
        if not np.isfinite(derivative).all():raise FloatingPointError('联合导数产生非有限值，未裁剪状态')
        return derivative,cardiac,respiration,network

    def evaluate(self,time,state):
        d,c,r,n=self.evaluate_full(time,state)
        return d,c,r,n['node_pressure']/PA

    def rhs(self,time,state):
        d,_,_,n=self.evaluate_full(time,state)
        self.vascular.update_delay(time,n)
        return d

    def auxiliary(self):
        result={'p0av':np.asarray(self.vascular.p0av).copy(),'facp_control':np.array(self.facp_control),
                'pleural_fraction':np.array(self.pleural_fraction)}
        result.update({'delay__'+k:np.asarray(v) for k,v in self.vascular.export_delay_state().items()})
        result.update({'geometry__'+k:v for k,v in self.vascular.export_reference_geometry().items()})
        return result

    def restore_auxiliary(self,values):
        geometry={k[10:]:values[k].copy() for k in values.keys() if k.startswith('geometry__')}
        self.vascular.set_reference_geometry(**(geometry or self._native_geometry))
        self.vascular.p0av[:]=values['p0av'];self.facp_control=float(values['facp_control'])
        self.pleural_fraction=float(values['pleural_fraction']) if 'pleural_fraction' in values else 1.
        self.vascular.restore_delay_state({k[7:]:values[k].copy() for k in values.keys() if k.startswith('delay__')})


def _odeca_beat_python(model,start):
    """原OdeCA二阶更新；只有推进RHS更新波历史。"""
    model.vascular.start_beat()
    dt=model.dt;count=round(model.period/dt);stride=round(model.config['output_interval_s']/dt)
    respiratory=model.respiratory_propagator.trajectory(start[70:76],count)
    ledger=start[76]+respiratory[:,:4].sum(axis=1)-start[70:74].sum()
    def set_resp(x,i):
        x[70:76]=respiratory[i];x[76]=ledger[i]
        return x
    samples=[(0.,np.array(start,copy=True))];h=np.array(start,copy=True)
    observations=[model.evaluate_full(0.,h)]
    first=model.rhs(0.,h);derivative=model.rhs(dt,set_resp(h+first*dt,1))
    g=.5*(derivative+first);h=h+g*dt
    set_resp(h,1)
    if stride==1:
        samples.append((dt,h.copy()));observations.append(model.evaluate_full(dt,h))
    for step in range(2,count+1):
        t=step*dt;derivative=model.rhs(t,set_resp(h+g*dt,step));g=(g+2*derivative)/3;h=h+g*dt
        set_resp(h,step)
        if step%stride==0 or step==count:
            samples.append((t,h.copy()));observations.append(model.evaluate_full(t,h))
    time=np.array([x[0] for x in samples]);states=np.array([x[1] for x in samples])
    model._last_beat_samples=(time,states,observations)
    return time,states


def odeca_beat(model,start):
    """融合加速与Python参考使用同一方程和更新次序。"""
    if not hasattr(model.heart,'_packed_parameters'):
        return _odeca_beat_python(model,start)
    from .solver_fast import fused_fast_beat
    time,states,_=fused_fast_beat(model,start)
    return time,states


def beat_data(model,time,states):
    stored=getattr(model,'_last_beat_samples',None)
    if stored is None or stored[1] is not states or not np.array_equal(time,stored[0]):
        raise ValueError('带波历史系统的输出必须使用推进时保存的同次因果采样')
    evaluated=stored[2]
    cardiac=[x[1] for x in evaluated]
    return dict(time=time,states=states,vessels=states[:,:18]*1e6,
        vascular_pressure=np.array([x[3]['node_pressure']/PA for x in evaluated]),
        artven_flow=np.array([x[3]['artven_flow']*1e6 for x in evaluated]),
        tube_q_prox=np.array([x[3]['tube']['qProx'] for x in evaluated]),
        tube_q_dist=np.array([x[3]['tube']['qDist'] for x in evaluated]),
        tube_area=np.array([x[3]['tube']['A'] for x in evaluated]),
        tube_ptrans=np.array([x[3]['tube']['pTrans'] for x in evaluated]),
        tube_impedance=np.array([.5*(x[3]['tube']['ZR']+x[3]['tube']['ZL']) for x in evaluated]),
        cardiac_volume=states[:,18:22]*1e6,
        cardiac_pressure=np.array([x['pressure_mmhg'] for x in cardiac]),
        flow=states[:,64:70]*1e6,ppl=np.array([x[2].values['ppl_mmhg'] for x in evaluated]),
        gas=states[:,70:76],blood_volume=states[:,:22].sum(axis=1)*1e6,
        C=states[:,24:44],Lsi=states[:,44:64],
        node_pressure=np.array([x['node_pressure_mmhg'][:4] for x in cardiac]),
        sarcomere_length=np.array([x['sarcomere_length'] for x in cardiac]),
        wall_tension=np.array([x['wall_tension_N_m'] for x in cardiac]),
        pericardial_pressure=np.array([x['pericardial_pressure_mmhg'] for x in cardiac]))


def beat_metrics(data,period=.85):
    mean=lambda v:np.trapezoid(v,data['time'],axis=0)/period
    co=mean(data['flow'][:,[2,3]])*.06
    return dict(LV_RV_CO_L_min=co.tolist(),MAP_mmhg=float(mean(data['vascular_pressure'][:,14])),
        baroreceptor_pressure_mmhg=float(mean(data['vascular_pressure'][:,2])),
        mean_PAP_mmhg=float(mean(data['vascular_pressure'][:,8])),
        mean_RAP_mmhg=float(mean(data['cardiac_pressure'][:,0])),
        artven_flow_m3_s=(mean(data['artven_flow'])*1e-6).tolist(),
        min_cardiac_volume_ml=float(data['cardiac_volume'].min()),
        blood_volume_ml=float(data['blood_volume'][-1]),max_C=float(data['C'].max()))


def physical_signature(include_reference=True):
    paths=[DATA/'config.json',DATA/'initial_state.json',DATA/'munneke_parameters.json',
        DATA/'original_circulation.json',DATA/'respiratory_parameters.json',
        ROOT/'cpmodel/circulation/reference_parameters.json',*sorted((ROOT/'cpmodel').rglob('*.py'))]
    if (DATA/'vascular_adaptation_parameters.json').exists():paths.append(DATA/'vascular_adaptation_parameters.json')
    if include_reference:
        paths.extend(sorted(DATA.glob('calibrated_reference*.npz')))
        paths.extend(sorted(DATA.glob('calibrated_reference*.json')))
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def simulate(out,*,beats=2,dt_us=500,resume=False):
    if isinstance(beats,bool) or not isinstance(beats,int) or beats<1:raise ValueError('本次心搏数须为正整数')
    out=Path(out);model=CardiorespiratoryModel(float(dt_us)*1e-6);signature=physical_signature()
    if resume:
        previous=read_json(out/'checkpoint.json');protocol=read_json(out/'protocol.json')
        if previous.get('heart_backend')!=BACKEND or previous.get('circulation_backend')!=NETWORK:
            raise ValueError('只能续算同一Munneke原ArtVen/Tube网络检查点，不能使用旧16池或Arts2024状态')
        if protocol['signature']!=signature or protocol['dt_s']!=model.dt:raise ValueError('续算方程、参数或步长已改变')
        delay=out/'delay_checkpoint.npz'
        if hashlib.sha256(delay.read_bytes()).hexdigest()!=previous['auxiliary_sha256']:raise ValueError('波传播检查点身份不符')
        with np.load(delay,allow_pickle=False) as z:
            state=z['state'].copy();model.restore_auxiliary(z)
        completed=previous['completed_beats'];history=read_json(out/'beats.json')
    else:
        out.mkdir(parents=True,exist_ok=False);state=model.initial_state();completed=0;history=[]
        protocol=dict(heart_backend=BACKEND,circulation_backend=NETWORK,dt_s=model.dt,period_s=model.period,
            physical_states=76,bookkeeping_states=1,patch_count=20,signature=signature,
            source_commit='85d9d8bdaddb6b09902ba3578cba1721caa6e145',
            initial_total_blood_volume_ml=float(state[:22].sum()*1e6),initial_air_volume_L=float(state[70:74].sum()),
            initial_mouth_ledger_L=float(state[76]),
            formal_steady_reference=model.calibrated_reference_loaded,
            initialization='已冻结标定参考' if model.calibrated_reference_loaded else '原PRef去冠脉初态，尚未标定',
            numerical_method='心血管OdeCA二阶物理状态更新；呼吸分段线性系统精确矩阵传播；Tube保留波历史',
            scientific_scope='20Patch、5ArtVen/8Tube、自主呼吸，无冠脉网络/供需/肾脏')
        save_json(out/'protocol.json',protocol)
    for beat in range(completed+1,completed+beats+1):
        time,states=odeca_beat(model,state);data=beat_data(model,time,states);row=beat_metrics(data)
        blood_error=float(np.max(abs(data['blood_volume']-protocol['initial_total_blood_volume_ml'])))
        air_error=float(np.max(abs(states[:,70:74].sum(axis=1)-protocol['initial_air_volume_L']
                                  -(states[:,76]-protocol['initial_mouth_ledger_L']))))
        if blood_error>1e-6 or air_error>1e-8:raise ValueError(f'冻结期守恒超差：血量{blood_error:g}mL，气量{air_error:g}L')
        row.update(beat=beat,blood_volume_error_ml=blood_error,air_volume_error_L=air_error)
        data['time']=time+(beat-1)*model.period
        np.savez_compressed(out/f'beat_{beat:04d}.npz',**data)
        history.append(row);state=states[-1].copy()
        temporary=out/'delay_checkpoint.tmp.npz'
        np.savez_compressed(temporary,state=state,**model.auxiliary());temporary.replace(out/'delay_checkpoint.npz')
        save_json(out/'checkpoint.json',dict(heart_backend=BACKEND,circulation_backend=NETWORK,
            completed_beats=beat,time_s=beat*model.period,state=state.tolist(),
            auxiliary_sha256=hashlib.sha256((out/'delay_checkpoint.npz').read_bytes()).hexdigest()))
        save_json(out/f'beat_{beat:04d}.json',row);save_json(out/'beats.json',history)
        summary=dict(heart_backend=BACKEND,circulation_backend=NETWORK,beats=beat,end_time_s=beat*model.period,
            dt_s=model.dt,physical_states=76,patch_count=20,coronary_network=False,coronary_supply_demand=False,
            max_blood_volume_error_ml=max(x['blood_volume_error_ml'] for x in history),
            max_air_volume_error_L=max(x['air_volume_error_L'] for x in history),last_beat=row,
            circadapt_module_loaded='circadapt' in sys.modules,dll_required=False,matlab_required=False,
            formal_steady_reference=protocol['formal_steady_reference'],clinical_validation=False)
        save_json(out/'summary.json',summary)
        print(f'原ArtVen/Tube第{beat}搏：MAP={row["MAP_mmhg"]:.3f} mmHg，血量误差{blood_error:.3g}mL',flush=True)
    export_csv(out);return summary


def export_csv(out):
    names=['time_s','P_RA_mmHg','P_LA_mmHg','P_LV_mmHg','P_RV_mmHg',
        'V_RA_mL','V_LA_mL','V_LV_mL','V_RV_mL','Q_Vc_RA_mL_s','Q_PV_LA_mL_s',
        'Q_LV_Ao_mL_s','Q_RV_PA_mL_s','Q_RA_RV_mL_s','Q_LA_LV_mL_s',
        'P_Ao_mmHg','P_PA_mmHg','Ppl_mmHg','V_alveolar_L','V_blood_mL']
    completed=read_json(out/'checkpoint.json')['completed_beats']
    with (out/'waveforms.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(names)
        for beat in range(1,completed+1):
            with np.load(out/f'beat_{beat:04d}.npz',allow_pickle=False) as d:
                values=np.column_stack((d['time'],d['cardiac_pressure'],d['cardiac_volume'],d['flow'],
                    d['vascular_pressure'][:,[14,8]],d['ppl'],d['gas'][:,3],d['blood_volume']))
                writer.writerows(values[int(beat>1):])
