#!/usr/bin/env python3
"""Inventory named/static LEAP hand poses without importing hardware code."""
from __future__ import annotations
import argparse
import ast
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parent
JOINTS=[f'{finger}_{joint}' for finger in ['if','mf','rf'] for joint in ['mcp','rot','pip','dip']]+['th_cmc','th_axl','th_mcp','th_ipl']
OBJECTS=['cube','ball','duck','spam','tomato']
KEYS={'robot_qref','init_ctrl','init_qpos','joint_reference','hold_targets','initial_hand',
      'home','home_state','qref','q_ref','q0','qpos0','reference_pose','neutral_pose','open_hand',
      'closed_hand','grasp_pose','best_pose','init_pos','default_dof_pos','default_joint_positions',
      'rest_pose','ref_qpos','default_pose','initial_pose','initial_qpos','best_init_ctrl','home_finger_qpos'}
SKIP_KEYS={'steps','failed_step','step_records','trials','trace','ranked','x_iters','func_vals',
           'source_snapshot_sha256','source_sha256','inputs_sha256','raw_controller_hashes',
           'q','q_dot','u','planned_u','action','action_delta','goal','goal_quat','target_quat'}


def literal(node):
    """Numeric literals/array constructors only: never execute discovered code."""
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):return node.value
    if isinstance(node,(ast.List,ast.Tuple)):return [literal(x) for x in node.elts]
    if isinstance(node,ast.UnaryOp):
        value=np.asarray(literal(node.operand))
        if isinstance(node.op,ast.USub):return -value
        if isinstance(node.op,ast.UAdd):return value
    if isinstance(node,ast.Attribute) and node.attr=='pi':return np.pi
    if isinstance(node,ast.Name) and node.id in ['pi','PI']:return np.pi
    if isinstance(node,ast.BinOp):
        left,right=literal(node.left),literal(node.right)
        if isinstance(node.op,ast.Mult) and isinstance(left,list) and isinstance(right,int):return left*right
        a,b=np.asarray(left),np.asarray(right)
        if isinstance(node.op,ast.Add):return a+b
        if isinstance(node.op,ast.Sub):return a-b
        if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node.op,ast.Div):return a/b
    if isinstance(node,ast.Call):
        name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
        if name in ['array','asarray','list','tuple']:return literal(node.args[0])
        if name in ['zeros','ones'] and int(literal(node.args[0])) in [16,23]:
            return np.full(int(literal(node.args[0])),0. if name=='zeros' else 1.)
        if name=='full' and int(literal(node.args[0])) in [16,23]:return np.full(int(literal(node.args[0])),literal(node.args[1]))
    raise ValueError('Not a static numeric pose')


class Catalog:
    def __init__(self):self.poses=[];self.scanned=[];self.errors=[];self.other_robot_poses=[]

    def add(self,values,name,path,role,*,obj=None,line=None,status='defined',detail=None):
        try:q=np.asarray(values,dtype=float)
        except (TypeError,ValueError):return
        if q.ndim!=1 or q.size not in [16,23] or not np.isfinite(q).all():return
        object_state=q[16:].tolist() if q.size==23 else None
        q=q[:16]
        if np.max(np.abs(q))>10:return
        source=dict(name=name,path=str(Path(path).resolve()),line=line,role=role,object=obj,status=status)
        if detail:source['detail']=detail
        if object_state is not None:source['object_state_xyz_wxyz']=object_state
        for pose in self.poses:
            deviation=float(np.max(np.abs(np.asarray(pose['q'])-q)))
            if deviation<=1e-6:
                source['max_difference_rad']=deviation
                if source not in pose['sources']:pose['sources'].append(source)
                if name not in pose['aliases']:pose['aliases'].append(name)
                return
        self.poses.append(dict(id=f'P{len(self.poses)+1:03d}',name=name,q=q.tolist(),aliases=[name],sources=[source]))

    def python(self,path,status='defined'):
        try:text=path.read_text();tree=ast.parse(text)
        except (OSError,UnicodeError,SyntaxError) as exc:
            self.errors.append(dict(path=str(path),error=str(exc)[:200]));return
        def walk(node,obj=None):
            if isinstance(node,ast.ClassDef):
                obj=next((o for o in OBJECTS if o in node.name.lower()),obj)
            if isinstance(node,ast.Dict):
                for key,value in zip(node.keys,node.values):
                    if isinstance(key,ast.Constant) and isinstance(key.value,str):
                        name=key.value.lower()
                        if name in KEYS:
                            try:self.add(literal(value),f'{obj or path.stem}.{key.value}',path,key.value,obj=obj,line=key.lineno,status=status)
                            except (ValueError,TypeError,IndexError,OverflowError):pass
                        walk(value,name if name in OBJECTS else obj)
                return
            if isinstance(node,(ast.Assign,ast.AnnAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                for target in targets:
                    name=target.id if isinstance(target,ast.Name) else target.attr if isinstance(target,ast.Attribute) else ''
                    if name.lower().strip('_') in KEYS:
                        try:self.add(literal(node.value),f'{obj or path.stem}.{name}',path,name,obj=obj,line=node.lineno,status=status)
                        except (ValueError,TypeError,IndexError,OverflowError):pass
            for child in ast.iter_child_nodes(node):walk(child,obj)
        walk(tree)
        # Recover inactive commented array definitions, preserving their status.
        lines=text.splitlines()
        for index,line in enumerate(lines):
            match=re.match(r'^\s*#\s*[\"\']?(init_qpos|init_ctrl|robot_qref|joint_reference|qref|q_ref)[\"\']?\s*[:=]\s*(.*)',line)
            if not match:continue
            role,expression=match.groups()
            for end in range(index,min(index+35,len(lines))):
                if end>index:
                    extra=re.match(r'^\s*#\s?(.*)',lines[end])
                    if not extra:break
                    expression+='\n'+extra.group(1)
                try:
                    value=literal(ast.parse(expression.strip().rstrip(','),mode='eval').body)
                    obj=next((o for o in OBJECTS if o in path.stem.lower()),None)
                    self.add(value,f'{obj or path.stem}.{role} (commented L{index+1})',path,role,
                             obj=obj,line=index+1,status='commented / inactive')
                    break
                except (SyntaxError,ValueError,TypeError,IndexError):pass
        self.scanned.append(str(path))

    def json(self,path,status='saved'):
        try:data=json.loads(path.read_text())
        except (OSError,ValueError,UnicodeError):return
        def walk(value,trail='',obj=None):
            if isinstance(value,dict):
                obj=value.get('object',obj)
                for key,child in value.items():
                    if key in SKIP_KEYS:continue
                    location=(trail+'.' if trail else '')+key
                    if key.lower() in KEYS or trail.endswith('references'):
                        self.add(child,f'{obj or path.parent.name}.{key}',path,location,obj=obj,status=status)
                    if isinstance(child,(dict,list)):walk(child,location,obj)
            elif isinstance(value,list):
                for i,child in enumerate(value):
                    if isinstance(child,(dict,list)):walk(child,f'{trail}[{i}]',obj)
        walk(data);self.scanned.append(str(path))

    def xml(self,path):
        try:tree=ET.parse(path)
        except (ET.ParseError,OSError):return
        for key in tree.findall('.//keyframe/key'):
            for role in ['qpos','ctrl']:
                if key.get(role):
                    if 'allegro' in str(path).lower():
                        self.other_robot_poses.append(dict(name=f'{path.stem}.{key.get("name")}.{role}',robot='Allegro',path=str(path),role=role,values=np.fromstring(key.get(role),sep=' ').tolist()))
                        continue
                    self.add(np.fromstring(key.get(role),sep=' '),f'{path.stem}.keyframe.{key.get("name", "unnamed")}.{role}',path,role,status='XML keyframe')
        self.scanned.append(str(path))


def cube_poses(poses):
    """Use object metadata, not experiment-directory names containing 'cube'."""
    return [p for p in poses if any(s.get('object')=='cube' or
        (s.get('object') is None and ('cube' in Path(s['path']).name.lower() or
         s['role'] in ['robot_qref','move_to_zero','home_finger_qpos'])) for s in p['sources'])]


def build():
    cat=Catalog()
    hardware=ROOT/'configs/leap_hardware.json'
    cat.json(hardware,status='current hardware configuration')
    for obj in OBJECTS[:3]:
        cat.python(ROOT/f'ContactModelStudy/Tasks/{obj.title()}Reorient.py',status='current task definition')
    cat.json(ROOT/'configs/leap_joint_references/cube.json',status='current captured hardware reference')
    cat.add(np.zeros(16),'Zero / reset pose',ROOT/'ContactModelStudy/Hardware/Episodes.py','move_to_zero',status='current reset command')
    # Source searches include frozen experiment code, not third-party vision code.
    paths=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','*.py','-g','*.json','-g','*.xml',
        '-g','!.git/**','-g','!iris-foundationpose-plusplus/**','-g','!.runtime/**',str(ROOT)],
        capture_output=True,text=True,check=True).stdout.splitlines()
    seen=set(cat.scanned)
    for raw in sorted(paths):
        path=Path(raw)
        rel=path.relative_to(ROOT)
        if str(path) in seen or any(x in rel.parts for x in ['tests','__pycache__','.venv','site-packages']):continue
        if path.name in ['build_leap_pose_catalog.py','view_leap_poses.py']:continue
        if path.suffix=='.py':cat.python(path,status='frozen experiment source' if 'results' in rel.parts else 'source definition')
        elif path.suffix=='.xml':
            if 'scenes' in rel.parts:cat.xml(path)
        elif path.suffix=='.json':
            # Controller files contain the effective cost references. Skip all
            # per-step samples in json(), retaining named configuration poses.
            if 'results' not in rel.parts or path.name=='controller.json' or any(s in path.name for s in ['reference','manifest','config','inputs']):
                cat.json(path,status='saved experiment reference' if 'results' in rel.parts else 'saved configuration')
    external=[Path('/home/chetan/comfree_dex/test_leap_hardware.py'),
              Path('/home/abhijit/contact_model_study/contact_study/tasks/grasp_reorient.py')]
    for path in external:
        if path.exists():cat.python(path,status='external / legacy definition')
    sdk=Path('/home/chetan/leap_agentic_sysid/third_party/LEAP_Hand_Hardware/python/iris_leap/iris_leap.py')
    if sdk.exists():
        native=np.full(16,3.14159);native[[0,4,8]]=3.14
        order=[1,0,2,3,5,4,6,7,9,8,10,11,12,13,14,15]
        cat.add((native-np.pi)[order],'LEAP SDK startup home (converted)',sdk,'allegro_to_LEAPhand(zeros(16))',
                status='SDK default; converted from native motor coordinates',
                detail='Subtract pi and swap MCP/rotation indices per non-thumb finger, as Hardware/Hand.py does. SDK uses rounded 3.14159/3.14.')
    # Useful short labels while every original alias remains in sources.
    names={'robot_qref':'Hardware preparation: user robot_qref'}
    for pose in cat.poses:
        first=pose['sources'][0]
        if first['role'] in names and first['path']==str(hardware):pose['name']=names[first['role']]
        elif '/history/' in first['path']:pose['name']=Path(first['path']).stem+' / '+first['role']
        elif first['path'].endswith('configs/leap_joint_references/cube.json'):pose['name']='Cube captured hardware / '+first['role']
    return cat


def write(cat,destination):
    destination.mkdir(parents=True,exist_ok=True)
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/leap/hardware/calibrated_hand.xml'))
    lo=np.array([model.joint(n).range[0] for n in JOINTS])
    hi=np.array([model.joint(n).range[1] for n in JOINTS])
    for pose in cat.poses:
        q=np.asarray(pose['q'])
        pose['outside_calibrated_joint_limits']=[JOINTS[i] for i in np.flatnonzero((q<lo)|(q>hi))]
    payload=dict(created_at=datetime.now(timezone.utc).isoformat(),joint_order=JOINTS,units='radians',
        scope='Named/static LEAP joint poses, commented alternatives, saved cost references and captured hold targets. Dynamic measured states/planned trajectories and synthetic test data excluded.',
        deduplication_tolerance_rad=1e-6,poses=cat.poses,other_robot_poses=cat.other_robot_poses,files_scanned=len(set(cat.scanned)),parse_errors=cat.errors)
    (destination/'catalog.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (destination/'poses.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['id','name',*JOINTS,'source_count'])
        for pose in cat.poses:writer.writerow([pose['id'],pose['name'],*pose['q'],len(pose['sources'])])
    lines=['LEAP HAND POSE CATALOGUE',f'{len(cat.poses)} distinct poses; {len(set(cat.scanned))} files scanned.',
           payload['scope'],'All joint values are radians in MuJoCo joint order.','Finger rows: index, middle, ring, thumb; four joints each.',
           'Entries within 1e-6 rad are grouped; catalog.json preserves every source/alias and any object state.','']
    for pose in cat.poses:
        lines.append(f"{pose['id']}  {pose['name']}  ({len(pose['sources'])} sources)")
        for i,finger in enumerate(['Index ','Middle','Ring  ','Thumb ']):lines.append(f"  {finger}: ["+', '.join(f'{v:.9g}' for v in pose['q'][4*i:4*i+4])+']')
        source=pose['sources'][0]
        lines.append(f"  {source['status']} | {source['path']}"+(f":{source['line']}" if source['line'] else '')+f" | {source['role']}")
        if pose['outside_calibrated_joint_limits']:lines.append('  Outside calibrated limits: '+', '.join(pose['outside_calibrated_joint_limits']))
        lines.append('')
    (destination/'poses.txt').write_text('\n'.join(lines))
    subset={**payload,'poses':cube_poses(cat.poses),'other_robot_poses':[]}
    (destination/'cube_poses.json').write_text(json.dumps(subset,indent=2)+'\n')
    cube_lines=['LEAP CUBE POSES (radians; Index / Middle / Ring / Thumb, four joints each)','']
    for pose in subset['poses']:
        cube_lines.extend([pose['id']+' '+pose['name'],*['  '+finger+': '+str([round(v,9) for v in pose['q'][i*4:i*4+4]]) for i,finger in enumerate(['Index','Middle','Ring','Thumb'])],''])
    (destination/'cube_poses.txt').write_text('\n'.join(cube_lines))
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/pose_catalog_20261116')
    args=parser.parse_args()
    payload=write(build(),args.output)
    print(f"{len(payload['poses'])} distinct poses in {args.output}; scanned {payload['files_scanned']} files; {len(payload['parse_errors'])} parse errors")
    for p in payload['poses']:print(p['id'],p['name'],f"({len(p['sources'])} sources)")
