"""独立心肺结果的中文科学图。"""
from pathlib import Path


def plot_results(folder):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    rows = []
    for i, path in enumerate(sorted(Path(folder).glob('beat_*.npz'))):
        with np.load(path, allow_pickle=False) as data:
            rows.append({key: data[key][int(i > 0):].copy() for key in
                         ('time', 'cardiac_pressure', 'cardiac_volume', 'flow', 'ppl', 'gas', 'vascular_pressure')})
    data = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    time = data['time']-data['time'][0]
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    for i, name in enumerate(('右房', '左房', '左室', '右室')):
        ax[0].plot(time, data['cardiac_pressure'][:, i], label=name)
        ax[1].plot(time, data['cardiac_volume'][:, i], label=name)
    for i, name in enumerate(('体静脉入右房', '肺静脉入左房', '左室射血', '右室射血')):
        ax[2].plot(time, data['flow'][:, i], label=name)
    ax[3].plot(time, data['ppl'], color='tab:purple', label='胸膜压')
    second = ax[3].twinx()
    second.plot(time, data['gas'][:, 3], color='tab:green', label='肺泡气室容积')
    second.set_ylabel('肺泡气室容积（L）')
    for axis, label in zip(ax, ('压力（mmHg）', '容积（mL）', '流量（mL/s）', '胸膜压（mmHg）')):
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
        axis.legend(loc='upper right', ncols=4, fontsize=8)
    ax[-1].set_xlabel('从本次起点起的时间（s）')
    fig.suptitle('Munneke20Patch基础心脏—血管—自主呼吸')
    fig.savefig(Path(folder)/'心肺波形.png', dpi=160)
    plt.close(fig)
    _plot_cardio_figures(folder, data)


def _plot_cardio_figures(folder, data):
    import numpy as np
    import matplotlib.pyplot as plt
    from .settings import load_parameters
    period=float(load_parameters()['period_s'])
    end=float(data['time'][-1])
    take=data['time']>=max(float(data['time'][0]),end-2*period)-1e-10
    t=data['time'][take]
    pressures=np.column_stack((data['cardiac_pressure'][take][:,[0,1,3,2]],
                              data['vascular_pressure'][take][:,[14,8]]))
    names=('右心房','左心房','右心室','左心室','主动脉／体动脉端','肺动脉')
    fig,axes=plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
    for i,ax in enumerate(axes.flat):
        ax.plot(t,pressures[:,i],color='#286a9b')
        ax.set_title(names[i]);ax.set_xlabel('运行时间（s）');ax.set_ylabel('压力（mmHg）');ax.grid(alpha=.22)
    fig.suptitle('Munneke基础心肺模型｜四腔与两动脉压力')
    fig.savefig(Path(folder)/'四腔与动脉压力.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    edges=((0,4),(1,5),(4,3),(5,2))
    for ax,name,(enter,leave) in zip(axes.flat,names,edges):
        ax.plot(t,data['flow'][take,enter],label='流入',color='#286a9b')
        ax.plot(t,data['flow'][take,leave],label='流出',color='#d16a2d')
        ax.axhline(0,color='gray',lw=.5);ax.legend();ax.set_title(name)
        ax.set_xlabel('运行时间（s）');ax.set_ylabel('流量（mL/s）');ax.grid(alpha=.22)
    fig.suptitle('四腔流量｜保留反向流动，无冠脉回流支路')
    fig.savefig(Path(folder)/'四腔流入流出.png',dpi=170);plt.close(fig)
    last=data['time']>=end-period-1e-10
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,name,i in zip(axes.flat,names,(0,1,3,2)):
        ax.plot(data['cardiac_volume'][last,i],data['cardiac_pressure'][last,i],color='#286a9b')
        ax.set_title(name);ax.set_xlabel('容积（mL）');ax.set_ylabel('压力（mmHg）');ax.grid(alpha=.22)
    fig.suptitle('四腔PV轨迹｜末一搏，不强制闭合')
    fig.savefig(Path(folder)/'四腔PV环.png',dpi=170);plt.close(fig)
