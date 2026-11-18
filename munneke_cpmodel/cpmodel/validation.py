"""固定参数两变量负荷匹配与独立共同周期、半步长验收。"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from .model import CardiorespiratoryModel, physical_signature, save_json, odeca_beat
from .settings import DATA
from .calibration import (TARGET_P,TARGET_Q,checkpoint,load_checkpoint,
                          alter_total_blood,settle,trajectory_difference)


def target_residual(metric):
    return np.log([metric['baroreceptor_pressure_mmhg']/TARGET_P,
                   metric['LV_RV_CO_L_min'][0]/TARGET_Q])


class TrialInfeasible(RuntimeError):
    """已记录的生理状态/周期收敛失败，回溯可尝试更小步幅。"""


def match(folder,base_path):
    """固定同相位初态，预先固定差分步幅，只调整血量和系统p0AV共同倍率。"""
    folder=Path(folder);folder.mkdir(exist_ok=False)
    base_model=CardiorespiratoryModel(.0005,use_calibrated=False)
    base_state=load_checkpoint(base_path,base_model)
    base_aux=base_model.auxiliary();blood=float(base_state[:22].sum())
    evaluations=[];cache={}
    def trial(x):
        key=tuple(float(v) for v in x)
        if key in cache:return cache[key]
        ident=len(evaluations)+1;sub=folder/f'trial_{ident:03d}';sub.mkdir()
        model=CardiorespiratoryModel(.0005,use_calibrated=False);model.restore_auxiliary(base_aux)
        model.facp_control=1.;model.pleural_fraction=1.
        record=dict(id=ident,log_parameters=list(key),blood_target_ml=float(blood*np.exp(x[0])*1e6),
                    systemic_p0av_multiplier=float(np.exp(x[1])))
        evaluations.append(record)
        state=base_state.copy()
        try:
            for ramp in range(1,21):
                fraction=ramp/20
                model.vascular.p0av[:4]=base_aux['p0av'][:4]*np.exp(x[1]*fraction)
                state=alter_total_blood(state,blood*np.exp(x[0]*fraction))
                _,trajectory=odeca_beat(model,state);state=trajectory[-1].copy()
            state,data,metric,periods=settle(model,state,sub,'candidate',max_periods=8)
        except (ValueError,FloatingPointError,ZeroDivisionError,RuntimeError) as exc:
            record.update(rejected=True,reason=str(exc));save_json(folder/'evaluations.json',evaluations)
            raise TrialInfeasible(str(exc)) from exc
        residual=target_residual(metric)
        record.update(metrics=metric,residual=residual.tolist(),accepted_periods=len(periods))
        save_json(folder/'evaluations.json',evaluations)
        value=(residual,model,state,data,metric,sub);cache[key]=value
        return value
    x=np.zeros(2);iterations=[]
    for iteration in range(9):
        result=trial(x);r=result[0]
        if np.max(abs(np.expm1(r)))<=.0035:
            checkpoint(folder/'matched_checkpoint.npz',result[1],result[2])
            save_json(folder/'matched.json',dict(log_parameters=x.tolist(),metrics=result[4],iterations=iterations))
            return folder/'matched_checkpoint.npz'
        if iteration==8:break
        epsilon=np.array([.002,.005]);jacobian=np.empty((2,2))
        for j in range(2):
            perturbed=x.copy();perturbed[j]+=epsilon[j]
            try:neighbor=trial(perturbed)
            except TrialInfeasible:
                epsilon[j]=-epsilon[j];perturbed=x.copy();perturbed[j]+=epsilon[j]
                neighbor=trial(perturbed)
            jacobian[:,j]=(neighbor[0]-r)/epsilon[j]
        step=-np.linalg.solve(jacobian,r);step=np.clip(step,[-.15,-.5],[.15,.5])
        record=dict(iteration=iteration,log_parameters=x.tolist(),residual=r.tolist(),
                    finite_difference_steps=epsilon.tolist(),jacobian=jacobian.tolist(),step=step.tolist())
        iterations.append(record);save_json(folder/'newton_history.json',iterations)
        for damping in (1.,.5,.25,.125):
            candidate=x+damping*step
            if np.any(abs(candidate)>np.array([.7,2.5])):continue
            try:next_result=trial(candidate)
            except TrialInfeasible:continue
            if np.linalg.norm(next_result[0])<np.linalg.norm(r):
                x=candidate;break
        else:raise RuntimeError('两变量负荷匹配未找到下降方向，未生成正式参考')
    raise RuntimeError('两变量负荷匹配未达到预设目标，未生成正式参考')


def _assert_periods(records,artven_targets):
    if len(records)<3 or records[-1]['consecutive_passes']<2:raise ValueError('缺少连续两次共同周期轨迹验收')
    for row in records[-2:]:
        if row['pressure_target_relative_error']>.01 or row['flow_target_relative_error']>.01:
            raise ValueError('冻结后的平均负荷未匹配预定目标')
        if max(abs(np.asarray(row['LV_RV_CO_L_min'])/TARGET_Q-1))>.01:
            raise ValueError('左/右心输出量未匹配')
        if row['relative_LV_RV_output_difference']>.002:raise ValueError('左/右排量尚未平衡')
        if row['max_blood_volume_error_ml']>1e-6:raise ValueError('冻结总血量不守恒')
        if row['max_air_volume_error_L']>1e-8 or row['max_chest_invariant_error_L']>1e-8:
            raise ValueError('气量或胸壁不变量不守恒')
        if row['minimum_blood_compartment_volume_ml']<=0:raise ValueError('存在非正血液容积')
        if np.max(abs(np.asarray(row['artven_flow_m3_s'])/np.asarray(artven_targets)-1))>.01:
            raise ValueError('ArtVen各支路平均流量未匹配原参考目标')


def validate(folder,base_path,publish=True):
    folder=Path(folder);folder.mkdir(exist_ok=False)
    coarse=CardiorespiratoryModel(.0005,use_calibrated=False)
    state=load_checkpoint(base_path,coarse);coarse.facp_control=1.;coarse.pleural_fraction=1.
    state,coarse_data,coarse_metric,coarse_records=settle(coarse,state,folder,'formal_500us')
    _assert_periods(coarse_records,coarse.vascular.q0av)
    checkpoint(folder/'verified_coarse.npz',coarse,state)
    fine=CardiorespiratoryModel(.00025,use_calibrated=False)
    fine.restore_auxiliary(coarse.auxiliary());fine_state=state.copy()
    fine_state,fine_data,fine_metric,fine_records=settle(fine,fine_state,folder,'formal_250us')
    _assert_periods(fine_records,fine.vascular.q0av)
    differences={}
    for key in ('baroreceptor_pressure_mmhg','MAP_mmhg','mean_PAP_mmhg'):
        differences[key]=abs(fine_metric[key]-coarse_metric[key])/max(abs(fine_metric[key]),1.)
    differences['LV_RV_CO_L_min']=list(abs(np.array(fine_metric['LV_RV_CO_L_min'])-
                                                coarse_metric['LV_RV_CO_L_min'])/TARGET_Q)
    differences['artven_flow_m3_s']=list(abs(np.array(fine_metric['artven_flow_m3_s'])-
        coarse_metric['artven_flow_m3_s'])/coarse.vascular.q0av)
    maximum=max(max(v) if isinstance(v,list) else v for v in differences.values())
    report=dict(accepted=False,dt_s=.0005,half_step_dt_s=.00025,
        coarse=coarse_metric,fine=fine_metric,half_step_mean_relative_differences=differences,
        maximum_half_step_mean_relative_difference=float(maximum),
        half_step_waveform_difference=trajectory_difference(fine_data,coarse_data),
        pressure_target_BrAr_mmhg=TARGET_P,cardiac_output_target_L_min=TARGET_Q,
        artven_target_flow_m3_s=coarse.vascular.q0av.tolist(),
        common_period_s=85,source_feedback_frozen=True,clinical_validation=False)
    save_json(folder/'verification.json',report)
    if maximum>.005:raise ValueError('主要均值半步长差超过0.5%，未发布正式参考')
    report['accepted']=True;save_json(folder/'verification.json',report)
    if publish:
        for suffix,model,reference_state,metric in (('',coarse,state,coarse_metric),('_250us',fine,fine_state,fine_metric)):
            reference_state=reference_state.copy();reference_state[76]=0.
            if abs(reference_state[75]-round(reference_state[75]))<1e-8:reference_state[75]=0.
            reference=DATA/('calibrated_reference'+suffix+'.npz')
            checkpoint(reference,model,reference_state)
            metadata=dict(accepted=True,physics_signature=physical_signature(include_reference=False),
                reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                dt_s=model.dt,wave_pressure_mode='transmural',calibration_controller_active=False,
                targets=dict(baroreceptor_pressure_mmhg=TARGET_P,cardiac_output_L_min=TARGET_Q,period_s=.85,
                             artven_flow_m3_s=model.vascular.q0av.tolist()),
                metrics=metric,half_step_verification=report,
                total_blood_volume_ml=float(reference_state[:22].sum()*1e6),
                p0av_pa=model.vascular.p0av.tolist(),verification_result_folder=folder.name,
                reference_geometry={k:v.tolist() for k,v in model.vascular.export_reference_geometry().items()},
                numerical_method='心血管OdeCA物理状态、守恒节点流；呼吸分段矩阵指数；跨壁波缓存',
                caveat='仅原基准均值负荷及数值周期稳态验收，不构成临床外部验证')
            save_json(reference.with_suffix('.json'),metadata)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--from-checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--match',action='store_true')
    args=parser.parse_args()
    path=args.from_checkpoint
    if args.match:path=match(args.out,path)
    else:validate(args.out,path)


if __name__=='__main__':main()
