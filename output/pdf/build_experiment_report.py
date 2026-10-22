"""Build the 13-page study report from frozen local experiment evidence."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph

ROOT = Path('/Users/nandanadileep/projects/vllm/vllm/vllm-exp')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=ROOT/'output/pdf')
OUT = parser.parse_args().output_dir
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}
FONT_DIR = Path('/Users/nandanadileep/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype')
pdfmetrics.registerFont(TTFont('Helvetica', str(FONT_DIR/'LiberationSans-Regular.ttf')))
pdfmetrics.registerFont(TTFont('Helvetica-Bold', str(FONT_DIR/'LiberationSans-Bold.ttf')))
pdfmetrics.registerFontFamily('Helvetica', normal='Helvetica', bold='Helvetica-Bold', italic='Helvetica', boldItalic='Helvetica-Bold')


def read(path):
    p = ROOT / 'experiments' / path
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def series(name, values):
    return {'name': name, 'values': values}


def model(labels, baseline, changed, unit='seconds', control='Stock vLLM'):
    return dict(labels=labels, baseline=baseline, changed=changed, unit=unit, control=control)


pages = []


def page(number, title, change, protocol, finding, caution, sources, models):
    for p in sources:
        path = ROOT / 'experiments' / p
        SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    pages.append(dict(number=number, title=title, change=change, protocol=protocol,
                      finding=finding, caution=caution, sources=sources, models=models))


d = read('01-prefix-scheduling/evidence/batches-20261020-195946/summary.json')
def prefix_values(k):
    return [d['aggregate'][k]['candidate_duration_median_ms']/1000,
            d['per_request'][k]['A']['median_first_text_ms']/1000,
            d['per_request'][k]['B']['median_first_text_ms']/1000]
page(1, 'Prefix-aware request scheduling',
     'Rank waiting requests by reusable cached prefix; break ties by FCFS. A hybrid protects the first three requests in FCFS order and applies prefix ranking to the rest, with a waiting-time safeguard.',
     'Original concurrent A-Y workload | 25 candidates x 64 outputs | 3 running slots | 3 timing rounds per policy',
     'Prefix-first: batch 18.89s versus 18.77s FCFS. A/B first text rose from about 3.4s to 10.8s. Hybrid restored A/B to about 3.3s, but batch time was 19.01s.',
     'No established batch-speed improvement. Early-request protection improves fairness relative to prefix-first, not overall speed. Qwen3-1.7B was not measured for this experiment.',
     ['01-prefix-scheduling/RESULTS.txt'],
     {'0.6B':model(['A-Y batch','A first text','B first text'],prefix_values('fcfs'),
                  [series('Prefix-first',prefix_values('prefix_first')),series('Hybrid',prefix_values('fcfs_head_prefix'))])})

hot=read('02-frequency-eviction/evidence/summary.json')['aggregate']
ay=read('02-frequency-eviction/evidence-ay/summary.json')['aggregate']
page(2, 'Frequency-based cache retention',
     'Prefer evicting less frequently reused, zero-reference cached blocks. Keep actively referenced blocks ineligible for eviction and preserve FCFS request order.',
     'Two profiles: 42 sequential hot/cold requests and original concurrent A-Y | 3 timing rounds per policy/profile',
     'Hot-return cache reuse increased from 16 to 272 tokens; probe first text fell from 119.50ms to 52.54ms. On concurrent A-Y, reuse stayed at 2,432 tokens and LFU was 2.5% slower.',
     'The hot/cold batch median fell 8.6%, but a no-reuse control also varied. Retention is demonstrated; the full timing difference is not isolated to eviction. Only 0.6B was tested.',
     ['02-frequency-eviction/RESULTS.txt','02-frequency-eviction/RESULTS-AY.txt'],
     {'0.6B':model(['Hot/cold batch','A-Y batch'],
                  [hot['vllm']['median_batch_ms']/1000,ay['vllm']['candidate_duration_median_ms']/1000],
                  [series('LFU',[hot['lfu']['median_batch_ms']/1000,ay['lfu']['candidate_duration_median_ms']/1000])])})

d=read('03-nonprefix-repair/evidence-vllm-ay/summary.json')['aggregate']
def serving_values(row):
    return [row['candidate_duration_median_ms']/1000,row['median_first_text_ms']/1000,row['median_request_latency_ms']/1000]
page(3,'Middle-block reuse with partial repair',
     'Match token pages even when earlier context differs, reuse their KV approximately, and recompute a boundary subset. The Metal worker skips selected rows using an alternate attention path; this is not exact causal-prefix caching.',
     'Original A-Y | 25 candidates + 3 blockers | 96 KV pages | 3 slots | 3 matched timing pairs',
     'Reused 352 matching non-prefix tokens and skipped 308 rows/layer/run. Group-median batch time was 1.85% lower, but repair was slower in two of three paired rounds.',
     'No reliable speedup or quality preservation established. Only 10/25 candidate answers matched stock exactly; semantic quality was not scored. No 1.7B serving comparison exists.',
     ['03-nonprefix-repair/RESULTS-VLLM-AY.txt'],
     {'0.6B':model(['A-Y batch','Median first text','Median request'],serving_values(d['baseline']),[series('Approx. repair',serving_values(d['repair']))])})

d=read('04-partial-prefix/evidence/summary.json')['aggregate']
page(4,'Exact partial-prefix caching',
     'Index matching prefixes inside a 16-token physical page. Reuse the exact causal prefix and copy the partial page before continuing, preserving page ownership and invalidation. Physical pages remain 16 tokens.',
     'Original A-Y | same native Metal worker | baseline, fine-hash control and partial reuse | 3 rotated rounds',
     'Candidate cached tokens rose from 2,432 to 2,563; the 32-token probe reused 31 instead of 16. Batch median fell 0.64%, with small improvements in all three baseline pairs.',
     'The timing gain is small and exploratory. Partial reuse matched 16/25 candidate answers; fine hashes alone matched 25/25. Exact cache identity does not guarantee bitwise generation equality.',
     ['04-partial-prefix/RESULTS.txt'],
     {'0.6B':model(['A-Y batch','Median first text','Median request'],serving_values(d['baseline']),[series('Partial reuse',serving_values(d['partial'])),series('Fine hashes only',serving_values(d['fine_hash']))])})

d=read('05-skip-continue/evidence/summary.json')
def skip_values(k):
    return [d['aggregate'][k]['duration_median_ms']/1000,
            d['group_latency'][k]['long']['first_text_ms']['median_of_run_medians']/1000,
            d['group_latency'][k]['short']['first_text_ms']['median_of_run_medians']/1000]
page(5,'Bounded skip-and-continue scheduling',
     'With chunked prefill disabled, bypass a waiting head that cannot fit the remaining token budget and admit a later fitting request. Restore skipped order; bound the scan to 16 requests, four bypasses/request and 30-second age protection.',
     'Targeted A-Y: 13 long then 12 short prompts | 8 blockers/slots | 512-token batch | 3 matched pairs',
     'Batch median fell 3.04% and throughput rose 3.14%. Short-prompt first text fell 43.9%, while long-prompt first text rose 161.5%. All three targeted batch pairs were faster.',
     'Original A-Y did not activate skipping. The targeted packing workload demonstrates a fairness tradeoff; the bypass bounds do not guarantee an admission deadline. Only 0.6B was tested.',
     ['05-skip-continue/RESULTS.txt'],
     {'0.6B':model(['Targeted batch','Long first text','Short first text'],skip_values('baseline'),[series('Skip/continue',skip_values('skip'))],control='Unchunked FCFS')})

d=read('06-combined/evidence/summary.json')['aggregate']
profiles=['ay','targeted','temporal']
page(6,'Combined cache and scheduling policies',
     'Enable experiment 02 frequency retention, experiment 04 exact partial-prefix reuse, and experiment 05 bounded skip/continue together. Keep approximate middle-block repair disabled; both controls use unchunked prefill.',
     'A-Y, targeted packing and sequential temporal profiles | 24 fresh-server runs including diagnostics | 3 pairs/profile',
     'Cache gains activated consistently: A-Y +131 cached tokens, targeted +149, and hot-return reuse 16 to 277. Group-median batch times improved, but paired completion changes swung widely.',
     'No stable overall speedup established. Targeted long-prompt first text rose 2.580s to 6.194s; short prompts improved. The later two-model scaling study is experiment 07, not a replication of these exact profiles.',
     ['06-combined/RESULTS.txt'],
     {'0.6B':model(['A-Y batch','Targeted batch','Temporal batch'],[d[p]['baseline']['batch_median_ms']/1000 for p in profiles],[series('Combined',[d[p]['combined']['batch_median_ms']/1000 for p in profiles])])})

models={}
for size in ['0.6B','1.7B']:
    rows=[read(f'07-scale-and-model/evidence/{size}-{n}-summary.json')['aggregate'] for n in [25,50,100]]
    models[size]=model(['25 candidates','50 candidates','100 candidates'],[r['baseline']['duration_median_ms']/1000 for r in rows],[series('Combined',[r['combined']['duration_median_ms']/1000 for r in rows])])
page(7,'Request-count and model-size scaling',
     'Scale the combined policy from experiment 06 to 25, 50 and 100 candidates on Qwen3-0.6B and Qwen3-1.7B. This changes workload scale and model size; it introduces no new policy.',
     'Both pinned models | 320/64-token long/short prompts | 8 slots | 256 pages | 48 runs | 3 pairs/scale point',
     'Combined was faster in only 8/18 matched pairs. Cached-token gains repeated on both models: +212, +441 and +885 at 25/50/100 candidates. The 1.7B/50 group median improved 12.43%, but 1.7B/100 regressed 3.37%.',
     'Run variability is large: 0.6B/50 was faster in absolute time than 0.6B/25. These bars are observed medians, not a clean scaling law or proof of a generally faster scheduler.',
     ['07-scale-and-model/RESULTS.txt'],models)

models={}
for size,sub in [('0.6B',''),('1.7B','qwen3-1.7B/')]:
    rows=[read(f'08-queue-retention/{sub}evidence/{p}-summary.json')['aggregate'] for p in ['targeted','ay']]
    vals=lambda k:[r[k]['median_duration_ms']/1000 for r in rows]
    models[size]=model(['Targeted batch','A-Y batch'],vals('vllm'),[series('Queue-aware',vals('queue')),series('Cumulative LFU',vals('lfu'))])
page(8,'Retain cache that waiting requests need',
     'Use prefixes of already waiting requests to prefer retaining relevant zero-reference cached blocks. Keep FCFS request order and allow normal reclamation rather than permanently reserving pages.',
     'Targeted return-prefix and original A-Y profiles | 3 policies | 24 runs/model | 3 rounds/policy/profile',
     'First-return reuse increased from 80 stock tokens (16 under LFU) to 176 queue-aware tokens on both models. Targeted batch median fell 9.82% on 0.6B and 1.42% on 1.7B.',
     'On A-Y, every policy reused the same 2,432 tokens. Queue-aware 1.7B A-Y was 10.30% slower and lost all three stock pairs. Retention generalizes; a consistent speed advantage does not.',
     ['08-queue-retention/RESULTS.txt','08-queue-retention/qwen3-1.7B/RESULTS.txt'],models)

models={}
for size,sub in [('0.6B',''),('1.7B','qwen3-1.7B/')]:
    rows=[read(f'09-aging-lfu/{sub}evidence/{p}-summary.json')['aggregate'] for p in ['shift','stable']]
    vals=lambda k:[r[k]['median_duration_ms']/1000 for r in rows]
    models[size]=model(['A-to-B shift','Stable hotset'],vals('vllm'),[series('Aging LFU',vals('aged_lfu')),series('Cumulative LFU',vals('lfu'))])
page(9,'Let old cache popularity fade',
     'Halve all reuse scores every 64 physical-block allocations, including scores of active blocks. Prefer low-score zero-reference victims and preserve upstream free-queue tie order. Compare against stock and cumulative LFU.',
     '84-request A-to-B shift and 42-request stable control | 32 pages | 1 slot | 24 runs/model | 3 rounds',
     'After cold scans, aging retained all 240 B-prefix tokens from probe cycle 2 on both models; stock retained 16 and cumulative LFU 32. Shift-probe first text improved 38.7% on 0.6B and 61.6% on 1.7B.',
     'Shift batch group medians were slower than stock by 2.86%/7.10%, despite faster return probes. Stable retention stayed at 272 tokens for both LFU policies. All timed texts matched stock.',
     ['09-aging-lfu/RESULTS.txt','09-aging-lfu/qwen3-1.7B/RESULTS.txt'],models)

d=read('10-cache-overhead/evidence/summary.json')['aggregate']
page(10,'Reduce cache implementation overhead',
     'Publish only newly computed or invalidated prompt ranges instead of rescanning prompt blocks after every step. Replace repeated free-list sorting with a lazy-invalidated LFU heap, preserving selection ties, copies and lifecycle updates.',
     'Matched reference vs incremental LFU + partial reuse | 0.6B only | shift/stable profiles | 16 GPU runs | 3 pairs/profile',
     'Publisher visits fell about 87%; selector host time fell 29-37%. Batch medians improved 0.56% on shift and 1.46% on stable, but stable won only one of three matched pairs. Decisions, hits and answers matched.',
     'Control is the earlier study implementation, not stock vLLM. The 1.7B follow-up reran stock/LFU/aging, not a matched reference/incremental A/B; historical timing changes do not isolate overhead. Its shift/stable aging batches were 41.194s/22.843s vs current stock 43.729s/24.753s.',
     ['10-cache-overhead/RESULTS.txt','10-cache-overhead/aging-replication/RESULTS.txt'],
     {'0.6B':model(['Shift batch','Stable batch'],[d[p]['reference']['batch_median_ms']/1000 for p in ['shift','stable']],[series('Incremental',[d[p]['incremental']['batch_median_ms']/1000 for p in ['shift','stable']])],control='Study reference')})

models={}
for size,sub in [('0.6B',''),('1.7B','qwen3-1.7B/')]:
    p=ROOT/f'experiments/11-partial-threshold/{sub}evidence/per-request.csv'
    SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(newline='') as f:rows=list(csv.DictReader(f))
    def vals(config):
        subset=[r for r in rows if r['config']==config]
        assert len(subset)==135
        return [statistics.median(float(r[k]) for r in subset) for k in ['first_text_ms','pair_ms']]
    models[size]=model(['Target first text','Source + target'],vals('g16-m16'),[series(label,vals(k)) for label,k in [('g1 / min0','g1-m0'),('g1 / min8','g1-m8'),('g4 / min0','g4-m0'),('g8 / min0','g8-m0')]],unit='milliseconds')
page(11,'Partial-reuse thresholds and granularity',
     'Accept partial hits only when they save a configured minimum number of extra token computations. Sweep 1/4/8-token matching and minimum 0/8/16; physical pages remain 16 tokens. min16 is a same-granularity recomputation control.',
     '15 divergence offsets | 128-token prompts | 1 output/source and target | 135 timed targets/config | 40 runs/model',
     'Unrestricted g1 matching reduced target first text by 15.5% on 0.6B and 22.7% on 1.7B versus stock. Source+target time fell 7.3%/10.8%. The eight-token minimum discarded useful small hits.',
     'Charts show stock and four of ten measured configurations; same-granularity min16 and other threshold controls remain in the source data. These are one-output-token probes, not concurrent throughput results. All 135 answers/config matched the recomputation control.',
     ['11-partial-threshold/RESULTS.txt','11-partial-threshold/qwen3-1.7B/RESULTS.txt'],models)

d=read('12-concurrent-partial/evidence/aggregate.json')
vals=lambda k:[d[c][k]['median_duration_ms']/1000 for c in ['4','12']]
page(12,'Partial reuse under concurrent mixed traffic',
     'Carry exact one-token partial reuse into a mixed workload: unrestricted hits versus minimum eight saved tokens. Vary client concurrency while keeping four server slots and the same finite KV capacity. No new source policy is added.',
     '1.7B only | 48 requests | 64-512 prompt / 16-64 output tokens | 16 seeded sources | 4/12 clients | 24 runs',
     'Unrestricted batch medians improved 3.46% at four clients and 1.90% at twelve; minimum8 was nearly tied. At twelve clients, unrestricted p95 request latency worsened 4.81%. All paired texts matched stock.',
     'Closed-loop arrivals depend on completions; the cache is seeded but churns. Seed time is excluded. Gains were not consistent in every round, and concurrency alone is not isolated from mixed lengths, longer decoding and finite-cache effects. No 0.6B replication was run.',
     ['12-concurrent-partial/RESULTS.txt'],
     {'1.7B':model(['4 clients batch','12 clients batch'],vals('stock'),[series('Unrestricted',vals('partial')),series('Minimum8',vals('minimum8'))])})

d=read('13-adaptive-prefill/evidence/aggregate.json')
vals=lambda k:[d[k]['batch_s'],d[k]['burst_ttft_ms']/1000,d[k]['p95_latency_ms']/1000]
page(13,'Adapt prefill chunks to current traffic',
     'Compare stock chunked prefill with built-in fair-share adaptation and a new decoder-load cap: max(16,128/(1+2D)), clamped to the budget. An unfinished prompt older than 750ms raises the global per-request cap to at least 64.',
     '1.7B only | 4 ongoing generators + 2 bursts of 10 long prompts | 8 slots | 128-token budget | caching/skip off | 12 runs',
     'The new cap reduced ongoing p95 text-chunk gaps from 201ms to 169ms (16%), but batch time rose 6.3% and burst first text 14.2%. Built-in adaptation reduced gaps to 145ms while batch time rose 42.8%.',
     'This is a streaming smoothness tradeoff, not a speed win. All timed texts matched stock. Chunk gaps are not exact token ITL; the age boost is not a strict waiting bound. Three fresh-server rounds with one short warmup are exploratory.',
     ['13-adaptive-prefill/RESULTS.txt'],
     {'1.7B':model(['Full batch','Burst first text','p95 request'],vals('stock'),[series('Decoder + age',vals('decode_adaptive')),series('Built-in adaptive',vals('existing_adaptive'))])})

# All numeric plot values originate in the frozen summaries or per-request CSVs.
for p in pages:
    for m in p['models'].values():
        assert len(m['baseline'])==len(m['labels'])
        assert all(len(s['values'])==len(m['labels']) for s in m['changed'])
        assert all(math.isfinite(x) and x>0 for x in m['baseline'])
report_data=dict(pages=pages, source_sha256=SOURCES,
                 notes='No new model runs. Missing model/control comparisons remain explicitly unmeasured.')
(OUT/'comparison-data.json').write_text(json.dumps(report_data,indent=2)+'\n')

W,H=842,595
INK=colors.HexColor('#172A3A')
MUTED=colors.HexColor('#506579')
BASE=colors.HexColor('#54738E')
TEAL=colors.HexColor('#137C78')
AMBER=colors.HexColor('#B87525')
PURPLE=colors.HexColor('#7955A1')
RED=colors.HexColor('#B35E68')
PALETTE=[TEAL,AMBER,PURPLE,RED]
BG=colors.HexColor('#F4F7F9')


def paragraph(c,text,x,y,width,size=9.2,leading=12,color=INK,bold=False):
    style=ParagraphStyle('body',fontName='Helvetica-Bold' if bold else 'Helvetica',
                         fontSize=size,leading=leading,textColor=color,alignment=TA_LEFT)
    item=Paragraph(text,style)
    _,height=item.wrap(width,1000)
    item.drawOn(c,x,y-height)
    return height


def chart(c,x,y,w,h,size,which,data,maximum):
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor('#D8E1E8'))
    c.roundRect(x,y,w,h,7,fill=1,stroke=1)
    c.setFillColor(INK)
    c.setFont('Helvetica-Bold',10)
    control=data['control'] if data else 'Original / control'
    title=f'Qwen3-{size} | '+(control if which=='baseline' else 'Modified variants')
    c.drawString(x+12,y+h-18,title)
    if data is None:
        message='No saved matched measurements for this model.'
        if size=='1.7B' and current_page==10:
            message='No matched reference/incremental overhead A/B.'
        c.setFillColor(BG)
        c.roundRect(x+12,y+14,w-24,h-45,4,stroke=0,fill=1)
        paragraph(c,'NOT MEASURED',x+25,y+80,w-50,12,15,MUTED,True)
        paragraph(c,message+' Values are not estimated or borrowed from another experiment.',x+25,y+59,w-50,9,12,MUTED)
        return
    values=[series(data['control'],data['baseline'])] if which=='baseline' else data['changed']
    palette=[BASE] if which=='baseline' else PALETTE
    # Legend is separate from the chart to preserve fixed panel geometry.
    lx=x+13
    for i,s in enumerate(values):
        c.setFillColor(palette[i]);c.rect(lx,y+h-34,6,6,fill=1,stroke=0)
        c.setFillColor(MUTED);c.setFont('Helvetica',7.2)
        c.drawString(lx+9,y+h-33,s['name'])
        lx+=c.stringWidth(s['name'],'Helvetica',7.2)+22
    drawing=Drawing(w-20,h-43)
    bars=VerticalBarChart()
    bars.x=34;bars.y=24;bars.width=w-70;bars.height=h-94
    bars.data=[s['values'] for s in values]
    bars.categoryAxis.categoryNames=data['labels']
    bars.categoryAxis.labels.fontName='Helvetica'
    bars.categoryAxis.labels.fontSize=7.2
    bars.categoryAxis.labels.dy=-2
    bars.categoryAxis.strokeColor=colors.HexColor('#B9C9D5')
    bars.valueAxis.valueMin=0
    bars.valueAxis.valueMax=maximum
    bars.valueAxis.valueStep=maximum/4
    bars.valueAxis.labels.fontSize=7
    bars.valueAxis.labels.fillColor=MUTED
    bars.valueAxis.labelTextFormat='%.0f' if maximum>=10 else '%.1f'
    bars.valueAxis.visibleGrid=True
    bars.valueAxis.gridStrokeColor=colors.HexColor('#E4EAF0')
    bars.valueAxis.gridStrokeWidth=0.4
    bars.valueAxis.strokeColor=colors.HexColor('#B9C9D5')
    bars.barLabelFormat='%.2f' if data['unit']=='seconds' else '%.1f'
    bars.barLabels.fontName='Helvetica'
    bars.barLabels.fontSize=6.9
    bars.barLabels.nudge=5
    bars.barLabels.fillColor=INK
    bars.groupSpacing=10
    bars.barSpacing=2
    for i in range(len(values)):
        bars.bars[i].fillColor=palette[i]
        bars.bars[i].strokeColor=None
    drawing.add(bars)
    renderPDF.draw(drawing,c,x+8,y+2)
    c.setFont('Helvetica',6.8);c.setFillColor(MUTED)
    c.drawRightString(x+w-12,y+h-18,data['unit']+' | lower is better')


pdf=OUT/'vllm-experiment-comparison.pdf'
c=canvas.Canvas(str(pdf),pagesize=(W,H),pageCompression=1)
c.setTitle('vLLM Scheduler Study - 13 Experiment Comparisons')
c.setAuthor('Nandana | local experiment evidence')
c.setSubject('Changes, matched controls, two-model coverage, results and limitations')
for p in pages:
    current_page=p['number']
    c.bookmarkPage(f'exp{current_page}')
    c.addOutlineEntry(f'{current_page:02}. {p["title"]}',f'exp{current_page}',level=0)
    c.setFillColor(BG);c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor(TEAL);c.rect(0,H-7,W,7,fill=1,stroke=0)
    c.setFillColor(MUTED);c.setFont('Helvetica-Bold',8)
    c.drawString(28,H-28,'PERFORMANCE ENGINEERING / vLLM SCHEDULER STUDY')
    c.drawRightString(W-28,H-28,f'EXPERIMENT {current_page:02} / 13')
    c.setFillColor(INK);c.setFont('Helvetica-Bold',21)
    c.drawString(28,H-55,p['title'])
    paragraph(c,'<b>What changed.</b> '+p['change'],28,H-69,W-56,9.1,11.8)
    paragraph(c,p['protocol'],28,H-112,W-56,8,10,MUTED)
    # Four fixed graph panels: model rows, control/modified columns.
    for size,y in [('0.6B',306),('1.7B',139)]:
        data=p['models'].get(size)
        maximum=None
        if data:
            maximum=max(data['baseline']+[v for s in data['changed'] for v in s['values']])*1.23
            maximum=math.ceil(maximum/4)*4
        chart(c,28,y,385,153,size,'baseline',data,maximum)
        chart(c,429,y,385,153,size,'changed',data,maximum)
    paragraph(c,'<b>Observed result.</b> '+p['finding'],28,126,W-56,9,11.3)
    paragraph(c,'<b>Interpretation.</b> '+p['caution'],28,92,W-56,8.3,10.4,MUTED)
    c.setStrokeColor(colors.HexColor('#D7E1E8'));c.line(28,43,W-28,43)
    source='Sources: '+' | '.join('experiments/'+s for s in p['sources'])
    paragraph(c,source,28,36,W-56,6.8,8,MUTED)
    c.setFont('Helvetica',6.6);c.setFillColor(MUTED)
    c.drawString(28,10,'MacBook Air M5 / Metal | three timed rounds unless stated | graph scales match within each model row; pages use different workloads')
    c.drawRightString(W-28,10,f'{current_page:02}')
    c.showPage()
c.save()
print(pdf)
print('PAGES',len(pages),'SOURCE FILES',len(SOURCES))
