"""Organize existing figures; no analysis or experiment execution."""
from collections import defaultdict
from pathlib import Path
import hashlib
import html
import json
import os
import posixpath
import re
import shutil
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BASE=ROOT/'results/bo_validation'
SOURCE=BASE/'M2_blending12_and_previous24_results_light_20261119.zip'
TARGET=BASE/'M2_blending12_and_previous24_results_grouped_20261119.zip'
PREFIX='M2_blending12_and_previous24'
COHORTS=[
    dict(folder='M2_temporal_blending_12_20261119',name='01_temporal_blending_12',title='Temporal blending — latest 12 episodes',
         summary='7 successes / 12 episodes (58.3%). Blending α=0.6; action-rate weight 15.',
         groups={
             '01_success_and_outcomes':[1,2,3,4],
             '02_pose_errors':[21,22],
             '03_blending_and_smoothness':[5,6,7,8,10,11,15,16],
             '04_temporal_alignment':[13,14],
             '05_timing_and_overhead':[17,19,20],
             '06_hardware_clipping':[9],
             '07_robot_motion_and_tracking':[12,18]}),
    dict(folder='M2_profile_latest_h4_rate15_24_20261119',name='02_previous_unblended_24',title='Previous unblended 24 episodes',
         summary='4 successes / 24 episodes (16.7%). Blending disabled; action-rate weight 15.',
         groups={
             '01_success_and_outcomes':list(range(1,6)),
             '02_pose_errors':list(range(6,12))+list(range(33,37)),
             '03_action_rate_and_smoothness':list(range(21,30)),
             '05_timing_and_feedback':list(range(12,18)),
             '06_hardware_clipping':[18,19,20],
             '07_robot_motion_and_tracking':[30,31,32]})]
STYLE='''body{font:16px/1.5 system-ui;max-width:1240px;margin:35px auto;padding:0 24px;background:#f7f9fc;color:#223249}a{color:#08778b}nav{background:white;padding:18px;border:1px solid #dce3eb;border-radius:8px}nav a{display:inline-block;margin:5px 18px 5px 0}h1{line-height:1.2}h2{margin-top:44px;scroll-margin-top:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}.card{background:white;border:1px solid #dce3eb;border-radius:8px;padding:14px;overflow:hidden}.card img{width:100%;height:280px;object-fit:contain}figcaption{font-size:14px;color:#53627a}code{background:#e9edf4;padding:2px 5px}small{color:#53627a}.top{float:right;font-size:14px}'''

def document(title,body):return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style>'+body+'</html>'
def title(group):return group[3:].replace('_',' ').title()
def sha(payload):return hashlib.sha256(payload).hexdigest()

def main():
    mapping={};catalog=[]
    for cohort in COHORTS:
        report=BASE/cohort['folder'];dest=HERE/'PLOTS'/cohort['name'];dest.mkdir(parents=True,exist_ok=True)
        old_catalog=json.loads((report/'graph_catalog.json').read_text());assigned={i:g for g,indices in cohort['groups'].items() for i in indices}
        assert len(assigned)==len(old_catalog)
        groups=defaultdict(list)
        for c in old_catalog:
            name=c['name'];number=int(name.split('_',1)[0]);group=assigned[number]
            for suffix in ['png','pdf']:
                rel=Path('PLOTS')/cohort['name']/group/('pdf' if suffix=='pdf' else '')/(name+'.'+suffix)
                target=HERE/rel;target.parent.mkdir(parents=True,exist_ok=True)
                source=report/'individual_graphs'/(name+'.'+suffix);shutil.copy2(source,target)
                mapping[str(source.relative_to(ROOT))]=rel.as_posix()
            groups[group].append(c);catalog.append(dict(cohort=cohort['name'],topic=group,name=name,caption=c['caption']))
        body=f'<h1 id="top">{html.escape(cohort["title"])}</h1><p>{html.escape(cohort["summary"])}</p><p><a href="../../OPEN_PLOTS.html">← All experiments</a> · <a href="{os.path.relpath(report/"index.html",dest)}">Full report and metrics</a></p>'
        body+='<nav>'+''.join(f'<a href="#{g}">{html.escape(title(g))} ({len(groups[g])})</a>' for g in sorted(groups))+'</nav>'
        for group in sorted(groups):
            body+=f'<h2 id="{group}">{html.escape(title(group))}<a class="top" href="#top">Back to topics ↑</a></h2><p><code>{group}/</code> — PNG images here; matching PDFs in <code>pdf/</code>.</p><div class="grid">'
            for c in groups[group]:
                n=c['name'];body+=f'<figure class="card"><a href="{group}/{n}.png"><img loading="lazy" src="{group}/{n}.png" alt="{html.escape(n)}"></a><figcaption><strong>{html.escape(n.replace("_"," "))}</strong><p>{html.escape(c["caption"])}</p><a href="{group}/{n}.png">Open PNG</a> · <a href="{group}/pdf/{n}.pdf">Open PDF</a></figcaption></figure>'
            body+='</div>'
        (dest/'index.html').write_text(document(cohort['title'],body))
    body='<h1>Grouped M2 experiment plots</h1><p>Choose an experiment, then a topic. Each gallery has previews, descriptions and links to separate PNG and PDF figures.</p><div class="grid">'
    for c in COHORTS:
        body+=f'<section class="card"><h2>{html.escape(c["title"])}</h2><p>{html.escape(c["summary"])}</p><p><a href="PLOTS/{c["name"]}/index.html"><strong>Browse grouped plots →</strong></a></p><ul>'+''.join(f'<li><a href="PLOTS/{c["name"]}/index.html#{g}">{title(g)}</a></li>' for g in c['groups'])+'</ul></section>'
    body+='</div><p>The cohorts differ in task weights. See the full reports for comparison limits. Raw bridge telemetry, large controller JSON dumps and videos are excluded from the compact archive.</p>'
    (HERE/'OPEN_PLOTS.html').write_text(document('Grouped M2 plots',body));(HERE/'grouped_plot_catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    # Existing on-disk reports retain all original flat-image links, while gaining
    # an entry point to the new galleries. No existing result files are moved.
    for c in COHORTS:
        index=BASE/c['folder']/'index.html';text=index.read_text()
        gallery=HERE/'PLOTS'/c['name']/'index.html';href=os.path.relpath(gallery,index.parent)
        banner=f'<nav id="grouped-plots-navigation"><strong><a href="{href}">Browse plots grouped by topic →</a></strong></nav>'
        if 'id="grouped-plots-navigation"' in text:text=re.sub(r'<nav id="grouped-plots-navigation">.*?</nav>',lambda _:banner,text,count=1,flags=re.S)
        else:text=re.sub(r'(<h1\b[^>]*>)',lambda m:banner+m.group(1),text,count=1)
        index.write_text(text)
    package(mapping,catalog)

def package(mapping,catalog):
    partial=TARGET.with_suffix('.zip.partial');manifest=[]
    def add(z,name,payload):
        z.writestr(PREFIX+'/'+name,payload);manifest.append(dict(path=name,bytes=len(payload),sha256=sha(payload)))
    # Relative URLs from unchanged archive report locations into root PLOTS.
    def report_html(payload,old_name):
        parent=posixpath.dirname(old_name);text=payload.decode()
        for source,target in mapping.items():
            if posixpath.dirname(source)==parent+'/individual_graphs':
                text=text.replace('individual_graphs/'+posixpath.basename(source),posixpath.relpath(target,parent))
        cohort=next(c for c in COHORTS if c['folder']==Path(parent).name)
        href=posixpath.relpath('PLOTS/'+cohort['name']+'/index.html',parent)
        text=re.sub(r'<nav id="grouped-plots-navigation">.*?</nav>','',text,flags=re.S)
        text=re.sub(r'(<h1\b[^>]*>)',lambda m:f'<nav><a href="{href}"><strong>Browse plots grouped by topic →</strong></a></nav>'+m.group(1),text,count=1)
        return text.encode()
    with zipfile.ZipFile(SOURCE) as old,zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as new:
        metadata=json.loads(old.read(PREFIX+'/MANIFEST.json'))
        for entry in old.infolist():
            rel=entry.filename.removeprefix(PREFIX+'/')
            if rel in ['README.txt','MANIFEST.json'] or rel in mapping:continue
            if any(rel==f'results/bo_validation/{c["folder"]}/index.html' for c in COHORTS):payload=report_html((ROOT/rel).read_bytes(),rel)
            else:payload=old.read(entry.filename)
            add(new,rel,payload)
        for path in sorted(HERE.rglob('*')):
            if not path.is_file() or path.name=='organize.py':continue
            rel=path.relative_to(HERE).as_posix();payload=path.read_bytes()
            if rel.startswith('PLOTS/') and rel.endswith('/index.html'):
                cohort=next(c for c in COHORTS if c['name']==Path(rel).parent.name)
                original=BASE/cohort['folder']/'index.html'
                local_href=os.path.relpath(original,path.parent)
                archive_href=posixpath.relpath(str(original.relative_to(ROOT)),posixpath.dirname(rel))
                payload=payload.decode().replace(local_href,archive_href).encode()
            add(new,rel,payload)
        add(new,'organize.py',(HERE/'organize.py').read_bytes())
        readme='''OPEN OPEN_PLOTS.html FIRST

Plots are grouped under PLOTS/:
  01_temporal_blending_12/   latest 12 blended episodes (7 successes)
  02_previous_unblended_24/ previous 24 episodes (4 successes)

Each experiment has a clickable index.html and numbered topic folders:
success/outcomes, pose errors, smoothness/action-rate/blending, alignment,
timing/feedback, hardware clipping, robot motion/tracking.

PNG images are directly in each topic folder; matching PDFs are in pdf/.
Original report pages link to these same grouped images. Flat image copies
were removed from the archive to avoid duplication. Exported graph ZIPs
inside the original reports are retained for compatibility.

Reports, numerical metrics, CSVs, plot data, scripts and smaller source logs
remain under their original results/... paths. Raw bridge_telemetry.jsonl,
large controller.json dumps and videos remain excluded. The previous
24-episode results exclude the aborted episode from completed-episode metrics.
Nothing was removed from original result folders in the workspace.

Raw source references in provenance manifests can point to files omitted from
this compact package. Plot-report scripts retain their original repository
paths; use the original workspace for rebuilding, or adjust output image paths.
MANIFEST.json describes the archive and records included file hashes.
'''
        add(new,'README.txt',readme.encode())
        new.writestr(PREFIX+'/MANIFEST.json',json.dumps(dict(completed_episodes=36,blended_episodes=12,previous_episodes=24,
            grouped_by_experiment_and_topic=True,plots=len(catalog),videos_included=False,large_raw_files_excluded=True,
            excluded_files=metadata.get('excluded_files',[]),files=manifest),indent=2)+'\n')
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None
        names=set(z.namelist());bad=[]
        for name in names:
            if not name.endswith('.html'):continue
            content=z.read(name).decode()
            for href in re.findall(r'(?:href|src)="([^"]+)"',content):
                if href.startswith(('#','http:','https:')):continue
                target=href.split('#',1)[0]
                if target and posixpath.normpath(posixpath.join(posixpath.dirname(name),target)) not in names:bad.append((name,href))
        assert not bad,bad
        assert sum('/PLOTS/' in n and n.endswith('.png') for n in names)==58
        assert sum('/PLOTS/' in n and n.endswith('.pdf') for n in names)==58
        assert not any('/individual_graphs/' in n for n in names)
    os.replace(partial,TARGET)
    TARGET.with_suffix('.zip.sha256').write_text(sha(TARGET.read_bytes())+'  '+TARGET.name+'\n')
    print('ARCHIVE',TARGET);print('SIZE_MB',round(TARGET.stat().st_size/1e6,2));print('PLOTS',len(catalog));print('ALL_HTML_LINKS_VERIFIED')

if __name__=='__main__':main()
