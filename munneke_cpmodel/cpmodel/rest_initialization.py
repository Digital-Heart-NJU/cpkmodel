"""原AdaptRest的血管Diameter与原PFC，仅在初始化阶段运行。"""
import argparse
from pathlib import Path
import numpy as np
from .model import CardiorespiratoryModel,odeca_beat,beat_data,beat_metrics,save_json
from .calibration import TARGET_P,checkpoint,settle
from .vascular_adaptation import adapt_diameter


def _window(rows):
    return dict(mean_pressure=float(np.mean([r['control_mean_pressure'] for r in rows])),
        mean_flow=np.mean([r['control_mean_flow_m3_s'] for r in rows],axis=0).tolist(),
        p0av=np.exp(np.mean(np.log([r['p0av_before'] for r in rows]),axis=0)).tolist(),
        geometry={key:np.exp(np.mean(np.log([r['geometry_before'][key] for r in rows]),axis=0)).tolist()
                  for key in rows[0]['geometry_before']})


def initialize(folder,*,max_beats=1000):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    model=CardiorespiratoryModel(.0005,use_calibrated=False);state=model.initial_state()
    rows=[];windows=[];previous=None;accepted=False
    save_json(folder/'protocol.json',dict(method='原Adapt压力/流量控制+原AdaptRest Diameter初始化',
        pressure_target_mmhg=TARGET_P,artven_target_flow_m3_s=model.vascular.q0av.tolist(),
        source_feedback_gain=.3,artven_diameter_gain=.15,tube_diameter_gain=.20,
        myocardial_adaptation=False,wall_volume_adaptation=False,bag_adaptation=False,valve_adaptation=False,
        source_period_s=.85,respiratory_period_s=5,common_period_beats=100,
        convergence_mean_parameter_relative_error=.005,maximum_beats=max_beats,
        pleural_ramp='前20搏原参考胸压，21至100搏引入完整自主呼吸'))
    for beat in range(1,max_beats+1):
        model.pleural_fraction=min(1.,max(0.,(beat-20)/80))
        geo=model.vascular.export_reference_geometry();p0av=model.vascular.p0av.copy()
        t,s=odeca_beat(model,state);d=beat_data(model,t,s);metric=beat_metrics(d)
        meanp=float(np.mean(d['vascular_pressure'][:,2]))
        meanq=np.mean(d['artven_flow'],axis=0)*1e-6
        if meanp<=0:raise ValueError('原初始化控制的平均系统压力非正')
        fac=(meanp/TARGET_P)**.3
        model.vascular.p0av[:4]*=np.exp(.3*(meanq[:4]/model.vascular.q0av[:4]-1)/model.vascular.av['kAV'][:4])/fac
        model.facp_control=fac
        state,adaptation=adapt_diameter(model,s[-1].copy(),d)
        model.vascular.set_reference_geometry(**adaptation['reference_geometry'])
        row=dict(beat=beat,pleural_fraction=model.pleural_fraction,metrics=metric,
            control_mean_pressure=meanp,control_mean_flow_m3_s=meanq.tolist(),
            p0av_before=p0av.tolist(),p0av_after=model.vascular.p0av.tolist(),
            geometry_before={k:v.tolist() for k,v in geo.items()},
            diameter=adaptation,total_blood_after_updates_ml=float(state[:22].sum()*1e6))
        rows.append(row)
        if beat%25==0:
            save_json(folder/'initialization_history.json',rows);checkpoint(folder/'initialization_checkpoint.npz',model,state)
            print(f'原Rest初始化{beat}搏：BrAr={meanp:.3f}，CO={metric["LV_RV_CO_L_min"]}，血量={state[:22].sum()*1e6:.3f}mL',flush=True)
        if beat%100==0:
            current=_window(rows[-100:]);difference=None
            if previous is not None:
                differences=[float(np.max(abs(np.log(np.array(current['p0av'])/previous['p0av']))))]
                differences.extend(float(np.max(abs(np.log(np.array(current['geometry'][k])/previous['geometry'][k]))))
                                   for k in current['geometry'])
                difference=max(differences)
            target_error=max(abs(current['mean_pressure']/TARGET_P-1),
                float(np.max(abs(np.array(current['mean_flow'])/model.vascular.q0av-1))))
            windows.append(dict(beat=beat,mean=current,parameter_log_difference=difference,target_relative_error=target_error))
            save_json(folder/'window_history.json',windows)
            if beat>=300 and difference is not None and difference<=.005 and target_error<=.02:
                accepted=True;break
            previous=current
    if not accepted:raise RuntimeError('原Rest初始化尚未达到参数统计稳定，未输出冻结参考')
    target_geometry={k:np.array(v) for k,v in current['geometry'].items()}
    old=model.vascular.export_reference_geometry()
    state[:10]*=(target_geometry['av_a0']/old['av_a0']).ravel(order='F')
    state[10:18]*=target_geometry['tube_a0']/old['tube_a0']
    model.vascular.set_reference_geometry(**target_geometry)
    model.vascular.p0av[:]=current['p0av'];model.facp_control=1.;model.pleural_fraction=1.
    checkpoint(folder/'diameter_frozen_start.npz',model,state)
    save_json(folder/'frozen_parameters.json',dict(geometry={k:v.tolist() for k,v in target_geometry.items()},
        p0av=model.vascular.p0av.tolist(),blood_ml=float(state[:22].sum()*1e6),beats=beat))
    state,data,metric,periods=settle(model,state,folder,'diameter_frozen',save_periods=False)
    return folder/'diameter_frozen_checkpoint.npz'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-beats',type=int,default=1000);args=parser.parse_args()
    print(initialize(args.out,max_beats=args.max_beats),flush=True)


if __name__=='__main__':main()
