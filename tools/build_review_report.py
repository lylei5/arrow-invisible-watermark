"""Build the measured review from completed CSV files (no result constants)."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def main():
    root=Path(__file__).resolve().parents[1]
    rows=list(csv.DictReader((root/'experiments/holdout/rows.csv').open(encoding='utf-8')))
    methods=['legacy','reference','arc','arc_crc']
    names={'legacy':'现有母版＋v4','reference':'参考母版＋v4','arc':'等弧长＋交织（沿用旧校验）','arc_crc':'reference-arc-v1'}
    conditions=list(dict.fromkeys(r['condition'] for r in rows))
    totals={}
    by_payload={}
    for method in methods:
        positive=[r for r in rows if r['method']==method and r['negative']=='False']
        negative=[r for r in rows if r['method']==method and r['negative']=='True']
        accepted=sum(r['accepted']=='True' and r['correct']=='True' for r in positive)
        wrong=sum(r['accepted']=='True' and r['correct']!='True' for r in positive)
        null=sum(r['accepted']=='True' for r in negative)
        totals[method]=dict(n=len(positive),accepted_correct=accepted,accepted_wrong=wrong,
                            negative_n=len(negative),false_accept=null)
        payloads=sorted(set(r['digits'] for r in positive))
        by_payload[method]=np.array([np.mean([r['accepted']=='True' and r['correct']=='True'
                                              for r in positive if r['digits']==p]) for p in payloads])
    rng=np.random.default_rng(20261107)
    idx=rng.integers(0,len(by_payload['legacy']),(10000,len(by_payload['legacy'])))
    for method in methods:
        boot=by_payload[method][idx].mean(1)
        totals[method]['cluster_bootstrap_95_ci']=np.quantile(boot,[.025,.975]).tolist()
    deltas={}
    for base in ['legacy','reference']:
        boot=(by_payload['arc_crc']-by_payload[base])[idx].mean(1)
        deltas[base]=dict(difference=float(np.mean(by_payload['arc_crc']-by_payload[base])),cluster_bootstrap_95_ci=np.quantile(boot,[.025,.975]).tolist())
    stats=dict(totals=totals,paired_differences=deltas,bootstrap_unit='payload, with all six capture conditions retained',
               bootstrap_repetitions=10000,seed=20261107)
    (root/'experiments/holdout/statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    table=['| 模拟条件 | 现有母版＋v4 | 参考母版＋v4 | 等弧长＋交织＋标准 CRC |',
           '|---|---:|---:|---:|']
    values={m:[] for m in methods}
    labels=['440 px / σ 0.8 / 1200 dpi','330 px / σ 1.2 / 1200 dpi','220 px / σ 1.2 / 1200 dpi',
            '330 px / σ 1.8 / 1200 dpi','330 px / σ 1.2 / 600 dpi','330 px / σ 1.2 / 局部模糊']
    for condition,label in zip(conditions,labels):
        counts=[]
        for method in methods:
            sample=[r for r in rows if r['method']==method and r['condition']==condition and r['negative']=='False']
            count=sum(r['accepted']=='True' and r['correct']=='True' for r in sample)
            counts.append(f'{count}/{len(sample)}');values[method].append(count)
        table.append('| '+label+' | '+' | '.join([counts[0],counts[1],counts[3]])+' |')
    table.append('| 合计 | '+' | '.join(f"{totals[m]['accepted_correct']}/{totals[m]['n']}" for m in ['legacy','reference','arc_crc'])+' |')
    lo,hi=np.array(deltas['reference']['cluster_bootstrap_95_ci'])*100
    total_text='\n'.join(f"- {names[m]}：{totals[m]['accepted_correct']}/{totals[m]['n']} = {totals[m]['accepted_correct']/totals[m]['n']:.2%}；"
                         f"带码照片接受错号 {totals[m]['accepted_wrong']}，原图对照误接受 {totals[m]['false_accept']}/{totals[m]['negative_n']}。" for m in methods)
    report=f'''# ARROW 22 × 6 mm 方法与效果审查

## Material Passport

- 任务：设计图案重提取、几何嵌入方法审查、可复现实验与参考印版。
- 源代码基线：`1469211bc62ff3c1f9027988f61834c170e616f0`。
- 输入：用户提供的 1110 × 570 JPEG 设计截图；来源校验见 `assets/reference/provenance.json`。
- 证据状态：本地 CPU 模拟已执行；新印版真实打印及手机拍摄尚未验证。

## 结果

`reference-arc-v1` 保留参考字标的外轮廓，在四个内孔使用等弧长 NRZ 码元，
将每组 Hamming(7,4) 纠错位交织到多个内孔，并使用标准 CRC-7/MMC。
在独立随机种子的 144 次模拟拍摄中，正确接受率为
{totals['arc_crc']['accepted_correct']}/{totals['arc_crc']['n']}（{totals['arc_crc']['accepted_correct']/totals['arc_crc']['n']:.2%}）。
同一参考母版沿用原 v4 为 {totals['reference']['accepted_correct']}/{totals['reference']['n']}。
收益主要来自局部缺陷时不再让薄弱内孔集中压低同一纠错组的裕度。

## 参考图案与物理尺寸

ROI 为原图 `(223,232)` 至 `(931,374)`，采用最大颜色通道与灰度联合阈值去除网格和标线，
保留五个字母、四个内孔以及 R 腿的弯曲轮廓。保存原图、二值提取、叠加图、4 倍掩膜、SVG 与带 DPI 的 PNG。
栅格截图边缘和红线交叠处存在约 1 个源像素的不确定性，约合 0.028 mm；这些文件是截图重建结果。
原仓库母版与本次源图二值轮廓按包围框归一化后的 IoU 约 0.8615，这个数值仅用于描述轮廓差异。

22 × 6 mm 定义为画幅。字标保持设计图约 5:1 的长宽比，在其中居中，墨迹约 20 × 4.01 mm。
2200 × 600 px 在 2540 dpi 下对应 22 × 6 mm；SVG 显式指定 `width="22mm" height="6mm"`。
若将墨迹强行拉伸至 22 × 6 mm，会改变设计图的比例。

## 现有方法审查

1. 内孔几何调制与单色印刷相容，四个内孔可承载 77 个 NRZ 码元；在该字标上有可检验的价值。
   当前幅度 ±8 底版像素等于径向 ±0.08 mm，在放大查看时可见。尚未进行视觉不可察觉性的用户实验。
2. v4 码元按角度均分；同一孔内的实际弧长不均。v4 的 `pair_order` 为恒等映射，
   仅加扰符号不会把纠错组的相邻位分散到其他内孔。新方案按弧长划分，并保证每组跨至少三个内孔。
3. 原 `crc7()` 不是标准 CRC-7/MMC：测试字符串 `123456789` 的结果为 `0x25`，标准值为 `0x75`。
   相隔 9 位的部分双比特错误会逃过原校验。可复现例：`13800138000` 与 `13733160208` 的原校验都为 `0101110`。
   新实现通过标准测试向量，并检查 37 位载荷内所有单比特与双比特错误的可检测性。
   既有 v4 印版继续使用其原校验约定；新校验使用独立格式及明确的解码入口。
4. 命令行原来默认生成 v3，而 HTTP 服务固定 v4。已将 CLI 默认和 `--layout 4` 与生产版式对齐。
   README 中 `logo_scale` 的定义也已纠正：它是照片墨迹宽度除以 2200，不是墨迹占整张照片宽度的比例。
5. 内置仿真的常见字标尺度约 0.7–0.9，明显大于小标签实拍常见尺度。
   本次另建 220/330/440 相机像素宽的画幅、600/1200 dpi 的物理尺寸通道。
   默认常规仿真只有 `stress=True` 时才经过印刷网格，不能直接充当 22 × 6 mm 实物验证。
6. 仓库注释引用了若干真机实验、脚本与文档章节，当前检出版本未包含相应原始照片及完整实验记录。
   本次能够复核的是随仓库样张、代码和新生成的模拟数据。

## 验证设计

- 开发阶段：种子 41000，每条件 6 个载荷，六个条件；比较现有母版、参考母版、等弧长交织和几何干扰拟合。
- 独立验证：种子 81000，每条件 24 个载荷，六个条件；共 24 个号码 × 六种条件 = 144 张带码照片/方法。
- 每张带码照片都对应同通道随机种子的原图对照，因此每方法另有 144 张原图对照。
- 四种方法共 1152 次解码调用。照片由覆盖率印刷网格、降采样前抗混叠、透视、相机高斯 PSF、
  渐变照明、灰度噪声 σ=1.5 与 JPEG Q=85 组成。局部缺陷对 A 内孔邻域额外模糊。
- 编码幅度统一为 ±8 px；载荷、通道随机种子、门限统一。采用图像盲配准与生产接受门；
  未向解码器提供载荷、真实单应矩阵或拍摄参数。没有降低原有门限。
- 主指标是“接受且号码正确”。“有原始候选”“CRC 通过”均不能单独算成功。
- 字标画幅宽分别为 220、330、440 相机像素；表中 σ 为相机像素。实际墨迹宽约为画幅宽的 20/22。

{chr(10).join(table)}

{total_text}

相对同参考母版的 v4，新方案增加 {(totals['arc_crc']['accepted_correct']-totals['reference']['accepted_correct'])/144*100:.2f} 个百分点。
按载荷聚类、保留每个载荷全部六种条件的 10000 次配对 bootstrap，差值的描述性 95% 区间为
[{lo:.2f}, {hi:.2f}] 个百分点。固定模拟条件下的区间不能外推到其他打印机、烧制工艺或手机。
原校验与标准校验的码字会不同，因此两者之间单个成功样本的差异不能单独归因于校验强度。

每种方法的带码错号接受和原图误接受均为本轮有限样本中的观察。
0/144 不构成误接受率为零的保证；仍需要更广泛的负例、异版印版及真实生产图像。
几何干扰拟合在开发集上未取得一致收益，作为研究对照保留，默认解码采用原始半径读出。

## 复现与印版

```bash
pip install -r services/decode_api/requirements.txt
pip install -r experiments/requirements.txt
python tools/extract_reference.py assets/reference/source_design.jpg
python -m unittest discover -s tests -v
python experiments/channel_review.py --out experiments/reproduce --n 24 --seed 81000 --methods legacy,reference,arc,arc_crc
python tools/reference_watermark.py export --out print-assets
python tools/reference_watermark.py decode PHOTO --format reference-arc-v1
python tools/reference_watermark.py decode PHOTO --format reference-v4
```

逐样本 CSV、配置与脚本哈希位于 `experiments/holdout/`；汇总和聚类统计也可从 CSV 重新生成。
印版 SVG/PNG 与自检 manifest 位于 `print-assets/`，PNG 已写入 2540 dpi。
真实照片使用与印版一致的母版和格式。原 HTTP v4 接口对应现有母版；参考印版使用上述独立入口。

## 下一步实物检验

使用随附 A4 对照页，以 100% 实际尺寸打印，先测量 50 mm 校准尺。
每种印版至少跨两批打印、两部手机；同一载荷和条件成对拍摄，保留原始照片、打印 DPI 与相机像素宽度。
同时统计正确接受、错号接受、原图误接受和肉眼可见程度。
如需陶瓷烧制场景，应在实际烧制后的制品上重复同一对照。

本轮实验在本机 CPU 上执行，运行库版本记录于 `experiments/holdout/runtime.json`。

## 相关原始资料

- [FontCode, Xiao et al.](https://arxiv.org/abs/1707.09418)：字形扰动与纠错的研究路线。
- [Keskinarkaus et al., feature-point synchronization](https://doi.org/10.1016/j.jvcir.2012.01.010)：打印扫描后的特征同步。
- [CRC-7/MMC parameters and check vector](https://reveng.sourceforge.io/crc-catalogue/all.htm#crc.cat.crc-7-mmc)。

这些资料支持相关技术路线，不构成本次 ARROW 印版效果的外部验证。
'''
    (root/'docs/REVIEW_2026-11-07.md').write_text(report,encoding='utf-8')
    plt.rcParams['font.sans-serif']=['SimHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,1,figsize=(12,8),gridspec_kw={'height_ratios':[.85,1.6]})
    control=np.array(Image.open(root/'assets/reference/ARROW_reference.png'))
    axs[0].imshow(control,cmap='gray',vmin=0,vmax=255);axs[0].axis('off')
    axs[0].set_title('参考字标：22 × 6 mm 画幅，墨迹约 20 × 4.01 mm',fontsize=14)
    x=np.arange(6)
    for j,(m,color,label) in enumerate([('legacy','#a4a7b0','现有母版＋v4'),('reference','#5888b5','参考母版＋v4'),('arc_crc','#1e9b88','reference-arc-v1')]):
        bars=axs[1].bar(x+(j-1)*.25,np.array(values[m])/24*100,.23,label=label,color=color)
        axs[1].bar_label(bars,labels=[f'{k}/24' for k in values[m]],fontsize=9,padding=3)
    axs[1].set_xticks(x,['清晰 / 440 px','常规 / 330 px','小图 / 220 px','失焦 σ1.8','打印 600 dpi','A 内孔局部模糊'])
    axs[1].set_ylim(0,116);axs[1].set_ylabel('正确接受率 (%)');axs[1].legend(loc='lower left',frameon=False)
    axs[1].spines[['top','right']].set_visible(False);axs[1].grid(axis='y',alpha=.15);axs[1].set_axisbelow(True)
    fig.suptitle('ARROW 物理尺寸通道模拟 · 独立随机种子验证',fontsize=17)
    fig.text(.5,.02,'每方法：144 张带码照片 + 144 张原图对照；真实打印与手机拍摄效果待验证。',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(root/'experiments/holdout/comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
