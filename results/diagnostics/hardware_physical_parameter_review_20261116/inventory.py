"""Read-only inventory of completed hardware episodes and saved physical models."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,hashlib,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
paths=sorted((ROOT/'results').rglob('controller.json'))
known={name:json.loads((ROOT/path).read_text()) for name,path in {
 'original':'configs/m2_original_P011_clip020/weights.json',
 'trial314':'results/isolated_comparisons/cube_joint_reference_weights_20261116/trial314/config.json'}.items() if (ROOT/path).is_file()}
# Obtain full precision 314/622 weights without relying on a file label.
comparison=json.loads((ROOT/'results/bo_validation/analysis_trial622_vs314_20261115/report.json').read_text())
for name,key in [('trial314','314_paced'),('trial622','622_paced')]:known[name]=json.loads((ROOT/comparison['sources'][key][0]).read_text())['weights']

def vector(text,default):
 if text is None:return list(default)
 v=list(map(float,text.split()));return v+list(default[len(v):])

def physical(xml):
 root=ET.fromstring(xml);classes={'main':dict(friction=[1.,.005,.0001],contype=1,conaffinity=1)}
 def defaults(el,parent):
  value=dict(parent);g=el.find('geom')
  if g is not None:
   value['friction']=vector(g.get('friction'),value['friction'])
   for k in ['contype','conaffinity']:
    if g.get(k) is not None:value[k]=int(g.get(k))
  classes[el.get('class','main')]=value
  for child in el.findall('default'):defaults(child,value)
 for el in root.findall('default'):defaults(el,classes['main'])
 groups=defaultdict(list);object_mass=0.;explicit_mass=None
 def body_walk(body,childclass='main'):
  nonlocal object_mass,explicit_mass
  childclass=body.get('childclass',childclass);name=body.get('name','')
  if name=='obj' and body.find('inertial') is not None:explicit_mass=float(body.find('inertial').get('mass'))
  for g in body.findall('geom'):
   props=dict(classes.get(g.get('class',childclass),classes['main']))
   for k in ['contype','conaffinity']:
    if g.get(k) is not None:props[k]=int(g.get(k))
   if not (props['contype'] or props['conaffinity']):continue
   friction=vector(g.get('friction'),props['friction']);gn=g.get('name','')
   group='cube' if name=='obj' else 'tip' if gn in ['if_tip','mf_tip','rf_tip','th_tip'] else 'palm' if name=='palm' else 'finger' if name.startswith(('if_','mf_','rf_','th_')) else None
   if group:groups[group].append(friction)
   if name=='obj' and g.get('mass'):object_mass+=float(g.get('mass'))
  for b in body.findall('body'):body_walk(b,childclass)
 world=root.find('worldbody')
 if world is not None:body_walk(world)
 return dict(mass_kg=explicit_mass if explicit_mass is not None else object_mass,friction={k:sorted({tuple(v) for v in values}) for k,values in groups.items()})

rows=[];seen=set();skipped=Counter();modes=Counter();errors=[]
for i,path in enumerate(paths):
 try:d=json.loads(path.read_text())
 except (ValueError,OSError) as e:errors.append([str(path),str(e)]);continue
 if not isinstance(d,dict):continue
 modes[str(d.get('mode'))]+=1
 args=d.get('args',{})
 if not d.get('hardware_config') or args.get('observe_only') or not d.get('episodes'):skipped['no_hardware_config_or_observation_or_no_episodes']+=1;continue
 if not any(e.get('steps') and e.get('completed') for e in d['episodes']):skipped['no_completed_episodes']+=1;continue
 scene=d.get('scenes',{}).get('rollout',{}).get('scene_xml')
 if not scene:skipped['no_scene_snapshot']+=1;continue
 try:phys=physical(scene)
 except (ValueError,ET.ParseError,TypeError) as e:errors.append([str(path),str(e)]);continue
 weights=d.get('weights',{});family=next((name for name,w in known.items() if isinstance(w,dict) and set(weights)==set(w) and all(abs(weights[k]-w[k])<=1e-6*max(1,abs(w[k])) for k in w)),None)
 if family is None:family='other_'+hashlib.sha256(json.dumps(weights,sort_keys=True).encode()).hexdigest()[:8]
 for e in d['episodes']:
  if not e.get('completed') or e.get('finish_reason') not in ['success','failure','timeout']:continue
  steps=[s for s in e.get('steps',[]) if s.get('action_published')]
  if not steps:skipped['no_published_actions']+=1;continue
  identity=(e.get('started_at_utc'),tuple(e.get('goal',[])),tuple(e.get('target_pos',[])),tuple(steps[0].get('q',[])),e.get('n_steps'))
  if identity in seen:skipped['duplicate_episode']+=1;continue
  seen.add(identity)
  fric=phys['friction']
  row=dict(source=str(path.relative_to(ROOT)),source_episode=e.get('episode',0)+1,started=e.get('started_at_utc'),outcome=e['finish_reason'],model=args.get('rollout_model',d.get('rollout_model')),task=args.get('task'),hand_acc=args.get('hand_acc'),obj_acc=args.get('obj_acc'),weight_family=family,
   mass_kg=phys['mass_kg'],friction=json.dumps(fric,sort_keys=True),cube_mu=fric.get('cube',[(None,)])[0][0],tip_mu=fric.get('tip',[(None,)])[0][0],palm_mu=fric.get('palm',[(None,)])[0][0],finger_mu=fric.get('finger',[(None,)])[0][0],
   clip=json.dumps(d['hardware_config'].get('ctrl_range')),delta_reference=d['hardware_config'].get('delta_reference','measured_position'),n_samples=args.get('n_samples'),temperature=args.get('temperature'),noise=args.get('noise_sigma'),warm_start=args.get('warm_start'),horizon=args.get('time_horizon'),paced=bool(args.get('pace_control',False)),steps=len(steps),wall_s=e.get('manipulation_wall_time_s'),
   reference_hash=hashlib.sha256(json.dumps(e.get('home',d.get('joint_reference')),sort_keys=True).encode()).hexdigest()[:10],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  rows.append(row)
 if (i+1)%100==0:print('scanned',i+1,'/',len(paths),flush=True)
with (OUT/'episodes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
keys=['model','task','hand_acc','obj_acc','weight_family','mass_kg','cube_mu','tip_mu','palm_mu','finger_mu','clip','delta_reference','n_samples','temperature','noise','warm_start','horizon','paced']
grouped=defaultdict(list)
for r in rows:grouped[tuple(r[k] for k in keys)].append(r)
for key,rs in grouped.items():
 count=Counter(r['outcome'] for r in rs);summary.append(dict(zip(keys,key),episodes=len(rs),successes=count['success'],drops=count['failure'],timeouts=count['timeout'],success_rate=count['success']/len(rs),sources=list(dict.fromkeys(r['source'] for r in rs))))
(OUT/'inventory.json').write_text(json.dumps(dict(files=len(paths),modes=modes,skipped=skipped,errors=errors,unique_completed_episodes=len(rows),groups=summary),indent=2)+'\n')
print('FILES',len(paths),'EPISODES',len(rows),'MODES',modes,'SKIPPED',skipped,'ERRORS',errors,flush=True)
for g in sorted(summary,key=lambda g:(g['model'] or '',g['weight_family'],g['mass_kg'],g['cube_mu'] or -1)):
 if g['model']=='M2':print({k:g[k] for k in keys+['episodes','successes','drops','timeouts']},flush=True)
