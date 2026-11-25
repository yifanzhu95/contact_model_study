"""Inspect the actual tracked mesh's face labels, without rendering hardware."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/leap_face_matplotlib')
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[3]
mesh=trimesh.load(ROOT/'iris-foundationpose-plusplus/meshes/cube_70_mm_2_adjusted/cube_70_mm_2.obj',process=False)
texture=np.asarray(mesh.visual.material.image.convert('RGB'))
uv=mesh.visual.uv[mesh.faces].mean(axis=1)
h,w=texture.shape[:2]
pix=np.c_[np.clip(uv[:,0]*(w-1),0,w-1),np.clip((1-uv[:,1])*(h-1),0,h-1)].astype(int)
colors=texture[pix[:,1],pix[:,0]]/255.
fig,axs=plt.subplots(2,3,figsize=(10,7),layout='constrained')
for ax,(axis,sign,right,up) in zip(axs.flat,[(0,1,[0,1,0],[0,0,1]),(1,1,[-1,0,0],[0,0,1]),(2,1,[1,0,0],[0,1,0]),(0,-1,[0,-1,0],[0,0,1]),(1,-1,[1,0,0],[0,0,1]),(2,-1,[-1,0,0],[0,1,0])]):
 selected=(mesh.face_normals[:,axis]*sign>.75)
 triangles=mesh.triangles[selected]
 polygon=np.stack([triangles@np.array(right),triangles@np.array(up)],axis=-1)
 ax.add_collection(PolyCollection(polygon,facecolors=colors[selected],edgecolors='face',linewidths=.15))
 ax.set(xlim=(-.039,.039),ylim=(-.039,.039),aspect='equal',title=('+' if sign>0 else '-')+'XYZ'[axis])
 ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Tracked cube mesh: approximate texture colours sampled per triangle')
fig.savefig(Path(__file__).with_name('tracked_mesh_faces.png'),dpi=160)
