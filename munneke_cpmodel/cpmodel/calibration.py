"""原负荷反馈、冻结后85s共同周期验证；所有目标在运行前固定。"""
import argparse
from pathlib import Path
import time as timer
import numpy as np
from .model import (CardiorespiratoryModel,odeca_beat,beat_data,beat_metrics,save_json)
from .settings import DATA,PA

COMMON_BEATS=100
TARGET_P=12200/PA
TARGET_Q=5.1
TOL=.002


def checkpoint(path,model,state):
    np.savez_compressed(path,state=state,**model.auxiliary())


def load_checkpoint(path,model):
    with np.load(path,allow_pickle=False) as z:
        model.restore_auxiliary(z)
        return z['state'].copy()


def feedback(model,state,beats,folder,log_relaxation=1.):
    """原Adapt逐搏控制；均值按完整积分网格的样本算术均值。"""
    original_interval=model.config['output_interval_s'];model.config['output_interval_s']=model.dt
    history=[];p0history=[];bloodhistory=[]
    gain=.3*log_relaxation
    for beat in range(1,beats+1):
        model.pleural_fraction=min(1.,max(0.,(beat-20)/80))
        t,s=odeca_beat(model,state);d=beat_data(model,t,s);row=beat_metrics(d)
        mean_p=float(np.mean(d['vascular_pressure'][:,2]))
        mean_q=np.mean(d['artven_flow'],axis=0)*1e-6
        fac=(mean_p/TARGET_P)**gain
        if not np.isfinite(fac) or fac<=0:raise ValueError('校正阶段压力非正，不能执行原Adapt')
        before=model.vascular.p0av.copy()
        model.vascular.p0av[:4]*=np.exp(gain*(mean_q[:4]/model.vascular.q0av[:4]-1)/model.vascular.av['kAV'][:4])/fac
        model.facp_control=fac
        state=s[-1].copy()
        row.update(beat=beat,control_mean_pressure_mmhg=mean_p,control_mean_q_m3_s=mean_q.tolist(),
            pleural_fraction=model.pleural_fraction,
            next_facp_control=fac,p0av_before=before.tolist(),p0av_after=model.vascular.p0av.tolist())
        history.append(row);p0history.append(before);bloodhistory.append(float(state[:22].sum()))
        if beat%10==0 or beat==beats:
            save_json(folder/'feedback_history.json',history);checkpoint(folder/'feedback_checkpoint.npz',model,state)
            print(f'标定反馈 {beat}/{beats}：BrAr {mean_p:.3f}，CO {row["LV_RV_CO_L_min"]}，血量 {row["blood_volume_ml"]:.3f}mL',flush=True)
    model.config['output_interval_s']=original_interval
    save_json(folder/'feedback_window_comparison.json',dict(
        last_100_mean_blood_m3=float(np.mean(bloodhistory[-100:])),
        previous_100_mean_blood_m3=float(np.mean(bloodhistory[-200:-100])) if beats>=200 else None,
        last_100_geometric_mean_p0av=np.exp(np.mean(np.log(p0history[-100:]),axis=0)).tolist(),
        previous_100_geometric_mean_p0av=np.exp(np.mean(np.log(p0history[-200:-100]),axis=0)).tolist() if beats>=200 else None))
    return state,np.asarray(p0history),np.asarray(bloodhistory)


def alter_total_blood(state,target):
    """仅标定阶段加减血量；按全部静脉腔及静脉Tube容积分配。"""
    result=state.copy();indices=np.array([1,3,5,7,9,14,15,16,17]);delta=target-state[:22].sum()
    result[indices]+=delta*state[indices]/state[indices].sum()
    if np.any(result[:22]<=0):raise ValueError('拟议血量校正会导致非正容积')
    return result


def common_period(model,state):
    if model.facp_control!=1. or model.pleural_fraction!=1.:
        raise ValueError('冻结验收必须关闭校正并完整接入胸膜压')
    rows=[];start_total=float(state[:22].sum());max_error=0.
    air_start=float(state[70:74].sum());ledger_start=float(state[76])
    chest_start=float(state[74]-state[71:74].sum());air_error=0.;chest_error=0.
    for beat in range(COMMON_BEATS):
        t,s=odeca_beat(model,state);d=beat_data(model,t,s)
        max_error=max(max_error,float(np.max(abs(s[:,:22].sum(axis=1)-start_total))*1e6))
        air_error=max(air_error,float(np.max(abs(s[:,70:74].sum(axis=1)-air_start-(s[:,76]-ledger_start)))))
        chest_error=max(chest_error,float(np.max(abs(s[:,74]-s[:,71:74].sum(axis=1)-chest_start))))
        d['time']=t+beat*model.period
        rows.append({k:v[int(beat>0):].copy() for k,v in d.items()})
        state=s[-1].copy()
    joined={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    metric=beat_metrics(joined,model.period*COMMON_BEATS)
    metric['max_blood_volume_error_ml']=max_error
    metric['max_air_volume_error_L']=air_error
    metric['max_chest_invariant_error_L']=chest_error
    metric['minimum_blood_compartment_volume_ml']=float(joined['states'][:,:22].min()*1e6)
    metric['relative_LV_RV_output_difference']=abs(metric['LV_RV_CO_L_min'][0]-metric['LV_RV_CO_L_min'][1])/TARGET_Q
    metric['pressure_target_relative_error']=abs(metric['baroreceptor_pressure_mmhg']/TARGET_P-1)
    metric['flow_target_relative_error']=abs(np.mean(metric['LV_RV_CO_L_min'])/TARGET_Q-1)
    return state,joined,metric


def trajectory_difference(current,previous):
    result={}
    series={'pressure':np.column_stack([current['cardiac_pressure'],current['vascular_pressure'][:,[14,8]]]),
            'flow':current['flow'],'volume':current['states'][:,:22]*1e6}
    old={'pressure':np.column_stack([previous['cardiac_pressure'],previous['vascular_pressure'][:,[14,8]]]),
         'flow':previous['flow'],'volume':previous['states'][:,:22]*1e6}
    for key,a in series.items():
        b=old[key];scale=np.maximum.reduce([np.ptp(a,axis=0),np.mean(abs(a),axis=0),np.ones(a.shape[1])])
        result[key]=float(np.max(np.max(abs(a-b),axis=0)/scale))
    a,b=current['states'][:,22:70],previous['states'][:,22:70]
    floor=np.r_[1e-8,1e-5,np.full(20,.1),np.full(20,.1),np.full(6,1e-8)]
    scale=np.maximum(np.maximum(np.ptp(a,axis=0),np.mean(abs(a),axis=0)),floor)
    result['heart_states']=float(np.max(np.max(abs(a-b),axis=0)/scale))
    return result


def history_difference(current,previous):
    error=0.
    for key in ('q','pL','pR','uP','uD','TauL','TauR'):
        a,b=current['delay__'+key],previous['delay__'+key]
        floor=1. if key in ('pL','pR') else 1e-9
        scale=np.maximum(np.maximum(np.max(abs(a),axis=0),np.max(abs(b),axis=0)),floor)
        error=max(error,float(np.max(abs(a-b)/scale)))
    return error


def settle(model,state,folder,prefix,min_periods=3,max_periods=10,save_periods=True):
    model.facp_control=1.;previous=None;previous_aux=None;streak=0;records=[]
    for block in range(1,max_periods+1):
        started=timer.perf_counter()
        state,data,metric=common_period(model,state)
        difference=None if previous is None else trajectory_difference(data,previous)
        auxiliary=model.auxiliary()
        if difference is not None:difference['wave_history']=history_difference(auxiliary,previous_aux)
        if difference is not None and max(difference.values())<=TOL:streak+=1
        else:streak=0
        metric.update(block=block,trajectory_difference=difference,consecutive_passes=streak,
                      elapsed_wall_s=timer.perf_counter()-started)
        records.append(metric);save_json(folder/(prefix+'_history.json'),records)
        checkpoint(folder/(prefix+'_checkpoint.npz'),model,state)
        np.savez_compressed(folder/(prefix+'_last_period.npz'),**data)
        if save_periods:np.savez_compressed(folder/(prefix+f'_period_{block:02d}.npz'),**data)
        print(f'冻结共同周期 {prefix} {block}：BrAr {metric["baroreceptor_pressure_mmhg"]:.5f}，CO {metric["LV_RV_CO_L_min"]}，轨迹差 {difference}',flush=True)
        previous=data;previous_aux=auxiliary
        if block>=min_periods and streak>=2:return state,data,metric,records
    raise RuntimeError('预设共同周期轨迹收敛标准未达到，未输出正式标定初态')


def run_source_feedback(folder,dt=.0005,feedback_beats=300):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    model=CardiorespiratoryModel(dt,use_calibrated=False);state=model.initial_state()
    save_json(folder/'protocol.json',dict(target_BrAr_mmhg=TARGET_P,target_CO_L_min=TARGET_Q,period_s=.85,
        respiratory_period_s=5,common_period_s=85,feedback_beats=feedback_beats,dt_s=dt,
        criterion_target_relative_error=.01,criterion_trajectory_relative_error=.002,
        criterion_LV_RV_output_difference=.002,criterion_half_step_mean_difference=.005,
        feedback_mean='完整积分网格含首尾点的算术均值；冻结验收为85s时间积分均值',
        controller='原Adapt逐搏血量校正与4器官p0AV更新；仅诊断，不自动生成正式参考',
        source_feedback_gain=.3,initialization_log_relaxation=1.,effective_initialization_feedback_gain=.3,
        allowed_final_adjustment='必要时仅总血量与系统p0AV共同乘子两参数确定性求根',
        startup='前20搏原参考胸压，第21至100搏线性引入自主呼吸胸膜压；其后完整自主呼吸',
        wave_pressure='波历史用跨壁压力；节点/瀑布计算用绝对压力；修正原非零外压历史混用',
        sampling='在接受的状态处保存因果输出，不在整搏结束后重算旧节点压力'))
    state,p0,blood=feedback(model,state,feedback_beats,folder)
    n=min(COMMON_BEATS,feedback_beats)
    model.vascular.p0av[:]=np.exp(np.mean(np.log(p0[-n:]),axis=0))
    state=alter_total_blood(state,float(np.mean(blood[-n:])))
    model.facp_control=1.
    checkpoint(folder/'freeze_start.npz',model,state)
    return settle(model,state,folder,'frozen')


def run(folder):
    """正式流程：固定参数周期基准→仅血量/系统负荷求根→独立半步长验收。"""
    from .validation import validate
    from .regional_calibration import match as regional_match
    from .rest_initialization import initialize
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    model=CardiorespiratoryModel(.0005,use_calibrated=False)
    save_json(folder/'protocol.json',dict(method='原AdaptRest血管Diameter与PFC初始化后冻结，再求根并独立验收',
        targets=dict(BrAr_mmhg=TARGET_P,CO_L_min=TARGET_Q,heart_period_s=.85),
        common_period_s=85,adjusted_parameters=['总模型血量','系统4个p0AV独立倍率'],
        fixed_parameters='心肌/激活/阀口/壁量/心包/血管本构形式/肺p0AV均固定',feedback_active_during_formal_run=False,
        stages=['原Rest血管直径适应与PFC初始化','冻结参考几何','各ArtVen区域负荷微调','独立半步长验收'],
        finite_difference_log_steps=[.002,.01,.01,.01,.01],parameter_ramp_beats=20,
        artven_target_m3_s=model.vascular.q0av.tolist()))
    path=initialize(folder/'native_rest_initialization')
    path=regional_match(folder/'regional_load_matching',path,workers=4)
    return validate(folder/'verification',path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--feedback-beats',type=int,default=300)
    parser.add_argument('--diagnostic-source-feedback',action='store_true',help='仅复现原PFC诊断，不发布标定参考')
    args=parser.parse_args()
    if args.diagnostic_source_feedback:run_source_feedback(args.out,feedback_beats=args.feedback_beats)
    else:run(args.out)


if __name__=='__main__':main()
