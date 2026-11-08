"""Assemble the completed audit's tabular evidence and readable HTML report."""
from pathlib import Path
import argparse,json,csv,html,hashlib,sys,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'legacy':'仓库母版 + v4','vector_v4':'矢量母版 + v4','v5_angle':'矢量母版 + v5'}
LABELS={'defocus_286':'286 px / 失焦 σ=1.6','defocus_330':'330 px / 失焦 σ=2.0',
        'tiny_220':'220 px / 失焦 σ=1.6','local_O_severe':'O 内孔局部失焦','print300':'300 dpi 理想覆盖采样'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    load=lambda name:json.loads((out/name).read_text())
    validation=load('validation/summary.json');standard=load('benchmark/summary.json')
    study=load('stress/summary.json');depth=load('depth6-stress/summary.json')
    camera=load('camera-benchmark/rows.json');negative=load('negative-v5/summary.json')
    quality=load('quality.json')
    assert len(validation)==15 and all(v['n']==40 for v in validation.values())
    assert len(camera)==128
    totals={}
    for method in NAMES:
        values=[v for k,v in validation.items() if k.startswith(method+'/')]
        totals[method]={key:sum(v[key] for v in values) for key in ('n','accepted_correct','accepted_wrong','rejected','bit_errors','bit_total')}
    # Cluster bootstrap resamples entire payloads, retaining the five correlated
    # captures per payload. It estimates uncertainty within this simulator only.
    rows=list(csv.DictReader((out/'validation/rows.csv').open()))
    payloads=sorted({r['payload'] for r in rows})
    improvements=[]
    for payload in payloads:
        score=lambda m:sum(int(r['accepted_correct']) for r in rows if r['payload']==payload and r['method']==m)/5
        improvements.append(score('v5_angle')-score('vector_v4'))
    rng=np.random.default_rng(49325)
    boot=np.mean(rng.choice(improvements,(10000,len(improvements)),replace=True),axis=1)
    interval=np.percentile(boot,[2.5,97.5]).tolist()
    camera_totals={}
    for method in ('production','camera'):
        pos=[r for r in camera if r['method']==method and not r['negative']]
        neg=[r for r in camera if r['method']==method and r['negative']]
        camera_totals[method]=dict(n=len(pos),correct=sum(r['correct'] for r in pos),
            wrong=sum(r['wrong'] for r in pos),negative_n=len(neg),negative_accepted=sum(r['accepted'] for r in neg),
            median_seconds=float(np.median([r['seconds'] for r in pos])))
    summary=dict(validation=totals,paired_improvement=float(np.mean(improvements)),
        payload_cluster_bootstrap_95ci=interval,camera=camera_totals,
        v5_negative_n=sum(v['n'] for v in negative.values()),
        v5_negative_accepted=sum(v['accepted_wrong'] for v in negative.values()),
        checksum_information_rank={'legacy':7,'CRC7_MMC':7},
        checksum_check_123456789={'legacy':'0x25','CRC7_MMC':'0x75'},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        baseline_commit='1469211bc62ff3c1f9027988f61834c170e616f0',
        date='2026-11-08',validation_seed=590871,
        scope='Synthetic print/camera evaluation. No physical printing, firing or live deployment.')
    (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    x=np.arange(5);fig,ax=plt.subplots(figsize=(10.8,4.5))
    for j,(method,label,color) in enumerate(zip(NAMES,['Repository + v4','Vector + v4','Vector + v5'],['#8a94a5','#4e81a3','#0a8a72'])):
        ys=[validation[method+'/'+key]['accepted_correct']/40*100 for key in LABELS]
        bars=ax.bar(x+(j-1)*.25,ys,width=.24,label=label,color=color)
        ax.bar_label(bars,labels=[f'{round(v*.4)}/40' for v in ys],fontsize=9,padding=3)
    ax.set_xticks(x,['286 px\nblur 1.6','330 px\nblur 2.0','220 px\nblur 1.6','Local O blur','300 dpi\ncoverage model'])
    ax.set_ylabel('Exact payload acceptance (%)');ax.set_ylim(0,115)
    ax.set_title('Independent validation | 40 payloads per condition | amplitude 0.08 mm',pad=15)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.18),ncol=3,frameon=False)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(out/'validation.png',dpi=180,bbox_inches='tight');plt.close(fig)
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'
    table_validation=table(['条件','仓库母版 + v4','矢量母版 + v4','矢量母版 + v5'],
        [[label]+[str(validation[m+'/'+key]['accepted_correct'])+'/40' for m in NAMES] for key,label in LABELS.items()]+
        [['合计']+[str(totals[m]['accepted_correct'])+'/200' for m in NAMES]])
    table_camera=table(['入口','正确接受','错号接受','无码误接受','中位耗时（本机）'],
        [[label,f"{camera_totals[m]['correct']}/32",camera_totals[m]['wrong'],f"{camera_totals[m]['negative_accepted']}/32",f"{camera_totals[m]['median_seconds']:.3f} s"] for m,label in [('production','现有入口'),('camera','缩略图定位 + 原像素测量')]])
    table_quality=table(['样本','径向起伏','PSNR','变化像素比例'],
        [[name,f"±{v['depth_mm']:.2f} mm",f"{v['PSNR_dB']:.2f} dB",f"{v['changed_pixel_fraction']*100:.2f}%"] for name,v in quality.items()])
    table_study=table(['探索性配置','压力测试正确接受 / 120'],[[label,sum(v['accepted_correct'] for k,v in data.items() if k.startswith(method+'/'))] for label,data,method in [
        ('矢量母版 v4 / 0.08 mm',study,'vector_v4'),('v5 等角交织 / 0.08 mm',study,'v5_angle'),
        ('v5 等弧长 / 0.08 mm',study,'v5_arc'),('v5 等角交织 / 0.06 mm',depth,'v5_angle'),('v5 等弧长 / 0.06 mm',depth,'v5_arc')]])
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARROW 微型水印方法与效果审视</title><style>
body{{font:16px/1.8 "Microsoft YaHei",sans-serif;color:#22303b;background:#f4f6f8;margin:0}}
main{{max-width:1050px;margin:32px auto;background:white;padding:48px;box-shadow:0 4px 30px #1826380c}}
h1{{font-size:32px;line-height:1.4}}h2{{font-size:23px;margin-top:38px;border-bottom:1px solid #dce4e8;padding-bottom:10px}}
p{{margin:14px 0}}a{{color:#087b67}}img{{max-width:100%;height:auto}}.lead{{border-left:4px solid #0a8a72;padding:10px 20px;background:#edf8f4}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #e0e6ea}}th{{background:#f0f5f7}}.scroll{{overflow:auto}}
pre{{padding:20px;background:#17252e;color:#e7f3f2;white-space:pre-wrap;font:13px/1.7 Consolas,monospace}}
.meta{{color:#61727d;font-size:14px}}.note{{background:#fff7e7;padding:14px 20px}}@media(max-width:700px){{main{{margin:0;padding:20px}}}}
</style><main>
<p class="meta">2026-11-08 · 22 × 6 mm · 几何母版、编码布局与拍摄链路</p>
<h1>ARROW 微型水印方法与效果审视</h1>
<div class="lead"><strong>存在经过仿真验证的改进。</strong>在新号码的 200 次压力验证中，矢量母版 + v5 正确接受 {totals['v5_angle']['accepted_correct']}/200；同一矢量母版上的 v4 为 {totals['vector_v4']['accepted_correct']}/200。保留测量分辨率的解码入口也改善了整幅照片的识别。实际陶瓷印刷与肉眼不可见性仍待实物验证。</div>
<h2>1. 矢量母版与尺寸</h2>
<img src="master/ARROW_master_2540dpi.png" alt="ARROW矢量重建母版">
<p>画布 22 × 6 mm；字标遵循图示 5:1，实际 22 × 4.4 mm，上下各留 0.8 mm。四处字距为 0.04X、0.03X、0.03X、0.01X。两个 R 使用同一组贝塞尔曲线。输出 SVG 和 600、1200、2540 dpi PNG。</p>
<p>曲线控制点为依据设计图的人工几何重建，设计图没有标出精确控制点，因此无法宣称是官方矢量原稿。仓库旧母版栅格化后只有 4 个墨迹连通分量，两个 R 在基线附近相连；重建母版为 5 个分离字母、4 个封闭内孔。旧版实际字标约宽 20 mm，新版宽 22 mm；算法对比以“矢量母版 + v4”为控制组，避免把字标增大带来的收益归于编码变化。</p>
<p><a href="master/ARROW_master.svg">下载 SVG</a> · <a href="master/geometry.json">几何尺寸说明</a> · <a href="ARROW_print_test_22x6mm.pdf">实际尺寸打印页</a></p>
<h2>2. 方法审视</h2>
<p>当前仓库的主要方案是内孔几何水印：37 位数字载荷，加 7 位校验，经 Hamming(7,4) 编为 77 位，再把起伏写入 O、两个 R 和 A 的内孔。这个方向适合单色细小标志；它保留字母外轮廓，但放大后可以看到内孔边界起伏。</p>
<p>v4 的 bit→cell 顺序实际为恒等映射。固定异或加扰改变正负号，不能把一个纠错组的比特分散到不同孔。v5 使用按各孔配额分配的跨孔交织，每个 7 位组覆盖至少 3 个孔；保留等角采样、77 码元和 Hamming 编码。</p>
<p>v5 采用 CRC-7/MMC，ASCII 123456789 的标准检查值为 0x75，仓库旧算法为 0x25。完整 37 位输入空间上，两者的二元映射秩都为 7，各有 128 个校验值。此处是标准化与可验证性改进，不是校验容量增加。两种格式不兼容，且 CRC 不提供密码学防伪。整体对比同时改变了校验规则与交织，未单独测定二者各自的贡献。</p>
<p>代码还存在尺度语义与文档偏差：logo_scale 约为照片字标宽度除以 2200，而非字标占整幅照片的比例；生产入口的 900 像素上限仅在灰度数组路径生效。新的 CameraDecoder 将定位缩略图与测量原图分开，并统一灰度/彩色处理。</p>
<h2>3. 独立验证结果</h2>
<p>固定方案后，以新种子 590871 生成 40 个号码，每个号码测试 5 个条件。全部方法使用相同号码、信道随机种子和接受门限。表内统计“号码完全正确且被接受”；本表所有方法的错号接受均为 0。未通过者按拒绝计数。</p>
{table_validation}
<img src="validation.png" alt="独立压力验证比较">
<p>v5 相对同母版 v4 的正确接受率提高 {np.mean(improvements)*100:.1f} 个百分点。按号码成组重采样的 95% bootstrap 区间为 [{interval[0]*100:.1f}, {interval[1]*100:.1f}] 个百分点；此区间只描述本仿真样本，不代表真实产品人群。220 像素强失焦仍明显不稳定。</p>
<p>较温和的常规条件：32 个号码 × 7 类条件，仓库 v4 与矢量 v4 均为 223/224，v5 为 224/224。常规条件接近饱和，主要收益集中在较难条件，特别是局部内孔退化。</p>
<p>v5 的独立无码对照：{summary['v5_negative_n']} 次，误接受 {summary['v5_negative_accepted']} 次。负例为同一清洁字标经过不同扰动，尚未覆盖任意其他商标、复杂背景或伪造印版；这不能证明实际误识别率为零。</p>
<h2>4. 现有印版可用的拍摄链路改进</h2>
<p>把约 440 像素宽的字标放在 2400 × 1800 原图中。现有入口把整幅灰度图缩至长边 900，细节同步减少；CameraDecoder 用 900 像素图找位置，再将单应矩阵映射回原图，使用原始像素测量。32 个号码及 32 个无码对照，沿用旧母版与 v4 格式：</p>
{table_camera}
<p>该实验是构造的整幅照片回归案例。它证明了提前缩整图造成的可避免损失，不能概括为所有实拍都从 0% 提升到 100%。耗时是本机 CPU 的结果，不代表线上 0.5 CPU 服务的延迟。</p>
<h2>5. 可见性与幅度</h2>
{table_quality}
<p>0.08 mm 在 1200 dpi 下约为 3.78 个墨点，在 600 dpi 下约为 1.89 个墨点。它属于微小几何改动，无法保证凑近观察时不可见。PSNR 是整幅画布指标，会弱化局部轮廓变化的视觉影响；外轮廓一致性已做像素级测试。</p>
<img src="comparison.png" alt="母版及不同水印幅度的放大比较">
{table_study}
<p>以上为方案选择实验，种子 725103。默认候选保留 0.08 mm。0.06 mm 可用于外观优先的实物对照；0.04 mm 只提供外观样本，尚未完成对应的系统鲁棒性验证。等弧长方案在小尺寸强失焦下未优于等角方案。</p>
<h2>6. 证据边界与替代技术</h2>
<p>模拟顺序为物理打印网格采样、轻微透视、相机模糊、噪声和 JPEG；另包含 O 孔局部模糊及 0.025 mm 油墨扩散代理。解码器不使用真值单应矩阵。打印采样是理想灰度覆盖模型，300 dpi 的高通过率不能解释成真实 300 dpi 打印必然成功。实际网点、烧制、釉面高光、曲面、曝光失真、视频多帧尚未验证。</p>
<p>TrustMark 等神经水印可以作为后续对照，但缺少这块 22 × 6 mm 单色字标的真实打印训练与验证数据，当前没有证据说明它会优于上述几何方案。本轮未训练神经模型。Adobe 官方也说明其打印表现取决于打印/扫描流程，水印分布于整幅图像。<a href="https://opensource.contentauthenticity.org/docs/durable-cr/tm-faq/">官方 FAQ</a></p>
<p>CRC-7/MMC 采用 <a href="https://reveng.sourceforge.io/crc-catalogue/1-15.htm#crc.cat.crc-7-mmc">CRC RevEng 参数与检查值</a>；软件审视基于 <a href="https://github.com/lylei5/arrow-invisible-watermark">用户仓库</a> 及本地可复现测试。服务器显卡检查后未占用 249 的 GPU；本轮 CPU 已足够，本机 GPU 未用于训练。</p>
<h2>7. 使用与交付状态</h2>
<p>源代码包含独立 CLI、母版生成器、v5 编解码器、CameraDecoder、基准脚本及 11 项单元测试。已验证前导零号码往返、旧样张兼容、无码拒绝、CRC 标准向量、校验映射秩、孔拓扑、字母间距、同形 R、外轮廓不变及单应矩阵尺度还原。所有修改为本地实验实现，线上服务和 GitHub 远端未切换。</p>
<pre>python -m pip install -r requirements-research.txt
python tools/arrow_research.py master --out generated/master
python tools/arrow_research.py embed --method v5 --digits 13800138000 --out generated/plate.png
python tools/arrow_research.py decode --method v5 --image generated/plate.png
python tools/arrow_research.py decode --method v4-legacy --image your_existing_photo.jpg
python -m unittest discover -s tests -v</pre>
<p>新母版与 v5 印版需要匹配的解码器；现有线上服务仍使用旧母版。打印 PDF 时选 100% / 实际大小，量取页面上的 50 mm 标尺，再单独拍摄每个字标，记录 DPI、介质、烧制条件、手机、拍摄距离、对焦和反光。</p>
<p><a href="results.json">结果摘要 JSON</a> · <a href="validation/rows.csv">独立验证逐样本 CSV</a> · <a href="validation/manifest.json">独立验证配置</a> · <a href="negative-v5/rows.csv">无码逐样本 CSV</a> · <a href="camera-benchmark/rows.json">拍摄链路结果</a></p>
<p class="meta">输出包括人工重建矢量、实际尺寸测试页、仿真数据及研究代码。结果支持进入实物验证阶段。</p>
</main></html>'''
    (out/'report.html').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
