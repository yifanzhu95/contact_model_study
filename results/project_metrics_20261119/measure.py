"""Read-only project inventory, with explicit source exclusions and size accounting."""
from pathlib import Path
from collections import Counter,defaultdict
import ast,csv,io,json,os,stat,subprocess,tokenize
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
IGNORE={'.aws','.codex','.agents'}
CACHE={'__pycache__','.pytest_cache','.ruff_cache','.mypy_cache','.venv','venv','node_modules'}
CONFIG={'.json','.yaml','.yml','.toml','.ini','.cfg'}
CPP={'.c','.cc','.cpp','.cxx','.h','.hh','.hpp','.cu','.cuh'}
SCRIPT={'.sh','.bash','.slurm'}
MEDIA={'.mp4','.mkv','.avi','.mov','.webm','.m4v'}
MODEL={'.xml','.obj','.stl','.ply','.mjb','.urdf','.mjcf','.dae','.glb','.gltf'}
def add(counts,key,size,allocated):
    x=counts[key];x['files']+=1;x['logical_bytes']+=size;x['allocated_bytes']+=allocated
def empty():return dict(files=0,logical_bytes=0,allocated_bytes=0)
def kind(suffix):
    if suffix=='.py':return 'Python'
    if suffix in CPP:return 'C/C++/CUDA'
    if suffix in SCRIPT:return 'Shell/HPC scripts'
    if suffix in CONFIG or suffix=='.xml':return 'Configuration/XML'
    return None
def core_source(p):
    a=p.parts
    if any(x in CACHE for x in a):return False
    if len(a)==1:return kind(p.suffix) is not None
    if a[0] in {'ContactModelStudy','experiments','tests','test_scripts','analysis','configs'}:return kind(p.suffix) is not None
    if a[0]=='camera_hand_calibration':
        return not any(x in {'results','logs','cache','runtime','model','data','captures'} for x in a[1:]) and kind(p.suffix) is not None
    if a[0]=='rl':
        # Project hardware evaluation integration; omit supplied policy package/runtime.
        return ('hardware_eval' in a or 'tests' in a or p.name in {'run_rl_hardware.py','launch_rl_hardware.py'}) and kind(p.suffix) is not None
    return False
def py_code_lines(path):
    source=path.read_text(errors='replace');lines=source.splitlines();docs=set()
    try:
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and node.body:
                e=node.body[0]
                if isinstance(e,ast.Expr) and isinstance(e.value,ast.Constant) and isinstance(e.value.value,str):docs.update(range(e.lineno,e.end_lineno+1))
    except SyntaxError:pass
    occupied=set()
    try:
        for t in tokenize.generate_tokens(io.StringIO(source).readline):
            if t.type in {tokenize.COMMENT,tokenize.NL,tokenize.NEWLINE,tokenize.INDENT,tokenize.DEDENT,tokenize.ENDMARKER,tokenize.ENCODING}:continue
            occupied.update(range(t.start[0],t.end[0]+1))
    except (tokenize.TokenError,IndentationError):
        occupied={i for i,x in enumerate(lines,1) if x.strip() and not x.lstrip().startswith('#')}
    return dict(physical_lines=len(lines),nonblank_lines=sum(bool(x.strip()) for x in lines),
        code_lines_excluding_comments_docstrings=sum(i not in docs and 0<i<=len(lines) and bool(lines[i-1].strip()) for i in occupied))
def distribution(a):
    x=np.array(a,dtype=float)
    return {k:float(v) for k,v in dict(n=len(x),min=x.min(),median=np.median(x),mean=x.mean(),p25=np.percentile(x,25),p75=np.percentile(x,75),p95=np.percentile(x,95),max=x.max()).items()} if len(x) else None
def main():
    top=defaultdict(empty);types=defaultdict(empty);roles=defaultdict(empty);results_groups=defaultdict(empty);all_files=[];source_rows=[];analysis_rows=[];errors=[];symlinks=0;run_sizes=defaultdict(empty);run_roles=defaultdict(lambda:defaultdict(int))
    seen_inodes={};unique=0
    for base,dirs,files in os.walk(ROOT,followlinks=False,onerror=lambda e:errors.append(str(e))):
        parent=Path(base)
        dirs[:]=[d for d in dirs if d not in IGNORE and parent/d!=OUT and not (parent/d).is_symlink()]
        for name in files:
            path=parent/name;rel=path.relative_to(ROOT)
            try:s=path.lstat()
            except OSError as e:errors.append(str(e));continue
            if stat.S_ISLNK(s.st_mode):symlinks+=1;continue
            if not stat.S_ISREG(s.st_mode):continue
            allocated=s.st_blocks*512;size=s.st_size;key=(s.st_dev,s.st_ino)
            if key not in seen_inodes:unique+=allocated;seen_inodes[key]=str(rel)
            add(top,rel.parts[0] if len(rel.parts)>1 else '[root files]',size,allocated);add(types,path.suffix.lower() or '[no extension]',size,allocated)
            suffix=path.suffix.lower()
            role=('video' if suffix in MEDIA else 'bridge_telemetry' if name=='bridge_telemetry.jsonl' else 'controller_episode_json' if name=='controller.json' else
                  'archive' if suffix in {'.zip','.tar','.gz','.bz2','.xz','.zst'} else 'numerical_data' if suffix in {'.npz','.npy','.csv','.h5','.hdf5','.parquet','.pkl','.pickle','.pt','.pth','.onnx','.safetensors'} else
                  'text_logs' if suffix in {'.log','.jsonl'} else 'model_geometry' if suffix in MODEL else 'other')
            add(roles,role,size,allocated)
            all_files.append(dict(path=str(rel),bytes=size,allocated_bytes=allocated,role=role))
            if rel.parts[0]=='results':
                add(results_groups,rel.parts[1] if len(rel.parts)>2 else '[root result files]',size,allocated)
                for j,part in enumerate(rel.parts):
                    if part.startswith('leap_experiment_'):
                        run=str(Path(*rel.parts[:j+1]));add(run_sizes,run,size,allocated);run_roles[run][role]+=size;break
            if core_source(rel):
                line=py_code_lines(path) if suffix=='.py' else dict(physical_lines=sum(1 for _ in path.open(errors='replace')))
                source_rows.append(dict(path=str(rel),kind=kind(suffix),bytes=size,**line))
            elif suffix=='.py' and (rel.parts[0]=='results' or (rel.parts[0]=='camera_hand_calibration' and 'results' in rel.parts)) and not any(x in CACHE or x in {'code','site-packages'} or 'snapshot' in x for x in rel.parts) and '.before' not in rel.name:
                analysis_rows.append(dict(path=str(rel),**py_code_lines(path)))
    def source_summary(rows):
        return dict(files=len(rows),by_kind=dict(Counter(r.get('kind','Python') for r in rows)),
                    physical_lines=sum(r['physical_lines'] for r in rows),
                    python_files=sum(r.get('kind','Python')=='Python' for r in rows),
                    python_physical_lines=sum(r['physical_lines'] for r in rows if r.get('kind','Python')=='Python'),
                    python_code_lines_excluding_comments_docstrings=sum(r.get('code_lines_excluding_comments_docstrings',0) for r in rows))
    runs=[]
    for run,sizes in run_sizes.items():
        row=dict(path=run,**sizes,video_bytes=run_roles[run].get('video',0),bridge_bytes=run_roles[run].get('bridge_telemetry',0))
        summary=ROOT/run/'summary.json'
        if summary.exists():
            try:
                d=json.loads(summary.read_text());row.update(completed=d.get('completed_episodes'),started=d.get('started_episodes'),aborted=d.get('aborted_episodes'),wall_time_s=d.get('wall_time_s'),controller_actions=d.get('mppi_time',{}).get('count'),status=d.get('status'),
                    n_samples=d.get('args',{}).get('n_samples'),horizon=d.get('args',{}).get('time_horizon'),model=d.get('args',{}).get('rollout_model'))
            except (ValueError,OSError):pass
        row['without_video_bytes']=sizes['logical_bytes']-row['video_bytes'];runs.append(row)
    eligible=[r for r in runs if (r.get('completed') or 0)>=6]
    git_count=len(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).split(b'\0'))-1
    result=dict(as_of_date_phoenix='2026-11-19',workspace_root=str(ROOT),
        workspace=dict(files=len(all_files),logical_bytes=sum(f['bytes'] for f in all_files),allocated_bytes=sum(f['allocated_bytes'] for f in all_files),unique_inode_allocated_bytes=unique,symlink_files_not_followed=symlinks),
        git_tracked_paths=git_count,top_level=dict(top),results_groups=dict(results_groups),file_types=dict(types),artifact_roles=dict(roles),
        primary_project_source=source_summary(source_rows),additional_project_analysis_scripts=source_summary(analysis_rows),
        hardware_experiment_directories=len(runs),hardware_experiments_with_summary=sum('completed' in r for r in runs),
        completed_episode_sum_non_deduplicated=sum(r.get('completed') or 0 for r in runs),
        completed6plus_run_size_bytes=distribution([r['logical_bytes'] for r in eligible]),
        completed6plus_without_video_bytes=distribution([r['without_video_bytes'] for r in eligible]),
        recent_sample=[r for r in sorted(runs,key=lambda r:Path(r['path']).name,reverse=True) if (r.get('completed') or 0)>=6][:12],
        errors=errors,methodology=dict(workspace='All regular files currently under the checkout, including .git, external vision checkout, logs, caches and archives. Symlinks not followed; excludes client metadata directories .aws/.codex/.agents and this measurement output folder.',
            primary_source='Project source/config files at root, ContactModelStudy, experiments, tests, test_scripts, analysis and configs; calibration code/config excluding runtime/data/model/results/cache; project RL hardware integration. Excludes external vision checkout, scenes/assets, generated results, caches and supplied policy runtime.',
            loc='Physical line count includes comments and blank lines. Python code-line count excludes blank/comment-only lines and AST-recognized docstrings; tokenize-based physical source lines, not statements.',
            analysis_scripts='Additional authored Python under results/calibration results excluding frozen code snapshots and caches; counted separately.',
            run_counts='leap_experiment_* folders in results; not assumed unique independent configurations. Episode sum not de-duplicated and not used as a claim of unique episodes.'))
    OUT.mkdir(exist_ok=True);(OUT/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('source_files.csv',source_rows),('hardware_run_sizes.csv',runs)]:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (OUT/name).open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
    (OUT/'additional_analysis_scripts.json').write_text(json.dumps(analysis_rows,indent=2)+'\n')
    print('WORKSPACE',result['workspace']);print('TOPLEVEL',json.dumps(dict(top)));print('SOURCE',json.dumps(result['primary_project_source']));print('ANALYSIS',result['additional_project_analysis_scripts']);print('ARTIFACT_ROLES',json.dumps(dict(roles)));print('RUNS',len(runs),'eligible6plus',len(eligible));print('SCAN_ERRORS',errors)

if __name__=='__main__':main()
