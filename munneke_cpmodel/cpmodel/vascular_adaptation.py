"""原AdaptRest的血管Diameter适应：纯计算，不修改联合模型。

逐式对应ArtVenAdapt.m、TubeAdapt.m。仅用于初始化；几何重塑引起的血量
跳变单独记账。调用方应用reference_geometry并接续newstate后须重新达到稳态。
"""
import json
from pathlib import Path
import numpy as np

PARAMETERS=Path(__file__).resolve().parents[1]/'data'/'vascular_adaptation_parameters.json'


def _array(value,shape,name,positive=False):
    try:result=np.asarray(value,dtype=float)
    except (TypeError,ValueError) as exc:raise ValueError(name+'必须为有限数值数组') from exc
    if result.shape!=shape or not np.isfinite(result).all():
        raise ValueError(name+'维度或有限值错误，应为'+str(shape))
    if positive and np.any(result<=0):raise ValueError(name+'必须严格为正，不裁剪输入')
    return result


def _plain(value):
    if isinstance(value,dict):return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def adapt_diameter(model,state77,data):
    """返回(newstate77, diagnostic)，输入和model均不变。

    data.states为N×77的SI状态，artven_flow为N×5 mL/s；tube_q_prox、
    tube_q_dist为m³/s，tube_area为m²，tube_ptrans为Pa，tube_impedance为
    (ZL+ZR)/2，单位Pa·s/m³。源函数使用样本算术mean/max，含输入端点，
    不替换成时间积分、不额外约束静脉正压。diagnostic.reference_geometry
    可直接传给vascular.set_reference_geometry(**geometry)。
    """
    state=_array(state77,(77,),'state77')
    if np.any(state[:18]<=0):raise ValueError('18个血管容积须为正')
    try:states=np.asarray(data['states'],dtype=float)
    except (KeyError,TypeError,ValueError) as exc:raise ValueError('缺少有效states历史') from exc
    if states.ndim!=2 or states.shape[1]!=77 or len(states)<1:
        raise ValueError('states须为至少1行的N×77数组')
    states=_array(states,states.shape,'states')
    if np.any(states[:,:18]<=0):raise ValueError('历史血管容积须为正')
    n=len(states)
    def field(name,columns,positive=False):
        if name not in data:raise ValueError('缺少直径适应历史字段：'+name)
        return _array(data[name],(n,columns),name,positive)
    q_av=field('artven_flow',5)*1e-6
    qp=field('tube_q_prox',8);qd=field('tube_q_dist',8)
    tube_area=field('tube_area',8,True)
    tube_pressure=field('tube_ptrans',8)
    tube_z=field('tube_impedance',8,True)
    vascular=model.vascular;av=vascular.av;tb=vascular.tb
    av_p0=_array(av['p0'],(2,5),'ArtVen.p0',True)
    av_a0=_array(av['A0'],(2,5),'ArtVen.A0',True)
    av_k=_array(av['k'],(2,5),'ArtVen.k',True)
    av_length=_array(av['Len'],(5,),'ArtVen.Len',True)
    tb_p0=_array(tb['p0'],(8,),'Tube.p0',True)
    tb_a0=_array(tb['A0'],(8,),'Tube.A0',True)
    tb_k=_array(tb['k'],(8,),'Tube.k',True)
    rho=float(vascular.rho)
    if not np.isfinite(rho) or rho<=0:raise ValueError('血密度必须有限且为正')
    target=json.loads(PARAMETERS.read_text(encoding='utf-8'))
    av_target=target['ArtVen'];tube_target=target['Tube']
    def target_array(section,key,shape):return _array(section[key],shape,'适应目标'+key,True)
    av_velocity_target=target_array(av_target,'vFlowMean',(2,5))
    av_impact=target_array(av_target,'vImpact',(2,5))
    tube_velocity_target=target_array(tube_target,'vFlowMean',(8,))
    tube_impact=target_array(tube_target,'vImpact',(8,))
    flat=lambda value:np.asarray(value).ravel(order='F')
    shape_av=lambda value:np.asarray(value).reshape((2,5),order='F')
    # ArtVenV2p在ArtVenAdapt中重新给出波阻抗；压力和面积也按同一本构计算。
    area=np.maximum(1e-10,states[:,:10]/np.repeat(av_length,2))
    normalized=area/flat(av_a0)
    exponent=1+(flat(av_k)/3-2)/1.02
    positive_pressure=1.02*normalized**exponent
    negative_pressure=-.02/normalized
    pressure=(positive_pressure+negative_pressure)*flat(av_p0)
    speed=np.sqrt(.25+(positive_pressure*exponent-negative_pressure)*flat(av_p0)/rho)
    impedance=rho*speed/area
    av_q=np.repeat(q_av,2,axis=1)
    av_vmean=shape_av(np.maximum(np.mean(np.maximum(av_q,0)/area,axis=0),
                                    np.mean(np.maximum(-av_q,0)/area,axis=0)))
    av_pmean=shape_av(np.sqrt(np.mean(pressure**2,axis=0)))
    av_pmax=shape_av(np.max(pressure,axis=0)+np.max(impedance*area*flat(av_impact),axis=0))
    av_factor=(1+np.sqrt(.7698e-6/av_a0))*av_vmean/av_velocity_target
    av_pressure_factor=np.maximum(av_pmean,.1*av_pmax)/av_p0
    av_volume_factor=av_factor**.15
    av_new_a0=av_a0*av_volume_factor
    av_new_p0=av_p0*av_pressure_factor**(.45/av_k)
    # TubeAdapt: 使用真实两端流均值；abs(mean(pTrans))，不是RMS。
    tube_q=.5*(qp+qd)
    tube_vmean=np.maximum(np.mean(np.maximum(tube_q,0)/tube_area,axis=0),
                          np.mean(np.maximum(-tube_q,0)/tube_area,axis=0))
    tube_pmean=np.sqrt(np.mean(tube_pressure,axis=0)**2)
    tube_pmax=np.sqrt(1.7*np.mean(tube_pressure**2,axis=0))+np.max(tube_z*tube_area,axis=0)*tube_impact
    tube_factor=(1+np.sqrt(.7698e-6/tb_a0))*tube_vmean/tube_velocity_target
    tube_pressure_factor=tube_pmean/tb_p0
    tube_volume_factor=tube_factor**.20
    tube_new_a0=tb_a0*tube_volume_factor
    tube_new_p0=tb_p0*tube_pressure_factor**(.60/tb_k)
    geometry=dict(av_p0=av_new_p0,av_a0=av_new_a0,tube_p0=tube_new_p0,tube_a0=tube_new_a0)
    for name,value in geometry.items():
        if not np.isfinite(value).all() or np.any(value<=0):
            raise ValueError('原生Diameter公式产生非正或非有限'+name+'，不添加人为下限')
    factor=np.r_[flat(av_volume_factor),tube_volume_factor]
    newstate=state.copy();newstate[:18]*=factor
    if not np.isfinite(newstate).all() or np.any(newstate[:18]<=0):
        raise ValueError('原生Diameter容积缩放无效')
    jump=newstate[:18]-state[:18]
    diagnostic=dict(mode='原AdaptRest仅Diameter',samples=n,
        statistics='源算术mean/max；含调用者提供的端点；artven_flow由mL/s换为m3/s',
        reference_geometry=geometry,vessel_volume_multiplier=factor,
        vessel_volume_jump_m3=jump,blood_volume_jump_m3=float(jump.sum()),
        blood_volume_before_m3=float(state[:22].sum()),blood_volume_after_m3=float(newstate[:22].sum()),
        ArtVen=dict(vMean_m_s=av_vmean,vFlowMean_target_m_s=av_velocity_target,
                    flow_velocity_ratio=av_factor,pressure_factor=av_pressure_factor,
                    area_multiplier=av_volume_factor,p0_multiplier=av_new_p0/av_p0,
                    pMean_rms_pa=av_pmean,pMax_with_impact_pa=av_pmax),
        Tube=dict(vMean_m_s=tube_vmean,vFlowMean_target_m_s=tube_velocity_target,
                  flow_velocity_ratio=tube_factor,pressure_factor=tube_pressure_factor,
                  area_multiplier=tube_volume_factor,p0_multiplier=tube_new_p0/tb_p0,
                  pMean_abs_mean_pa=tube_pmean,pMax_with_impact_pa=tube_pmax),
        source_sha256=target['source']['sha256'])
    return newstate,_plain(diagnostic)
