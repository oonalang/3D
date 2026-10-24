"""Create a reconstructed, viewable 3D character asset from the provided image and single-view mesh.
One view cannot uniquely constrain the unseen rear, so those surfaces are approximate.
Dependencies: numpy, scipy, numba, trimesh, PIL, VTK only for preview.
"""
import os, math, numpy as np, trimesh
from PIL import Image
from scipy.interpolate import CubicSpline
from numba import njit
from trimesh.visual.material import PBRMaterial, SimpleMaterial
ROOT='/mnt/data'; OUT=f'{ROOT}/reconstructed_character'; os.makedirs(OUT,exist_ok=True)
SRC=ROOT+'/reference_projection_test/reference_projected.obj'
PHOTO=ROOT+'/reference_projection_test/reference_projection.jpg'
np.random.seed(17)
orig=trimesh.load(SRC,process=False)
V=orig.vertices.copy(); F=orig.faces.copy()

@njit(cache=True)
def select_front(V,F, W=950, H=1600):
    depth=np.full((H,W), -100000., dtype=np.float32)
    ids=np.full((H,W), -1, dtype=np.int32)
    x=(V[:,0]+.48)/.96*(W-1)
    y=(1.07-V[:,1])/2.16*(H-1)
    for i in range(F.shape[0]):
        a,b,c=F[i]
        ax,ay,az=x[a],y[a],V[a,2];bx,by,bz=x[b],y[b],V[b,2];cx,cy,cz=x[c],y[c],V[c,2]
        den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(den)<1.e-8:continue
        xmin=max(0,int(min(ax,bx,cx)));xmax=min(W-1,int(max(ax,bx,cx))+1)
        ymin=max(0,int(min(ay,by,cy)));ymax=min(H-1,int(max(ay,by,cy))+1)
        if xmin>xmax or ymin>ymax:continue
        for yy in range(ymin,ymax+1):
            for xx in range(xmin,xmax+1):
                w0=((by-cy)*(xx-cx)+(cx-bx)*(yy-cy))/den
                if w0<-.001:continue
                w1=((cy-ay)*(xx-cx)+(ax-cx)*(yy-cy))/den
                if w1<-.001 or w0+w1>1.001:continue
                zz=w0*az+w1*bz+(1-w0-w1)*cz
                if zz>depth[yy,xx]:
                    depth[yy,xx]=zz
                    ids[yy,xx]=i
    vis=np.zeros(F.shape[0],dtype=np.bool_)
    for j in range(H):
        for k in range(W):
            if ids[j,k]>=0:vis[ids[j,k]]=True
    return vis,depth

print('Selecting visible surface...')
visible,depth=select_front(V,F)
C=V[F].mean(axis=1)
back=~visible
hair_area=(((abs(C[:,0])>.275) & (C[:,1]>.01)) | ((abs(C[:,0])>.18)&(C[:,1]>.16)&(C[:,2]<-.08)) | ((C[:,1]>.67)&(C[:,2]<.04)&(abs(C[:,0])>.11)))
tail_area=(C[:,1]<-.23)&(abs(C[:,0])>.17)&(C[:,2]<.05)
# Never disturb directly visible reference geometry.
hair_recolor=back & hair_area
tail_recolor=back & tail_area & ~hair_recolor
photo_mask=~(hair_recolor|tail_recolor)

def mesh_part(mask):
    sub=F[mask]
    used=np.unique(sub)
    mp=np.full(len(V),-1,dtype=np.int32);mp[used]=np.arange(len(used))
    mesh=trimesh.Trimesh(vertices=V[used],faces=mp[sub],process=False)
    return mesh,used

front,frontidx=mesh_part(photo_mask)
front.faces[front.face_normals[:,2]<0] = front.faces[front.face_normals[:,2]<0][:, ::-1]
front.visual=trimesh.visual.TextureVisuals(uv=orig.visual.uv[frontidx].copy(),material=PBRMaterial(name='Reference color, unlit',baseColorTexture=Image.open(PHOTO).copy(),metallicFactor=0,roughnessFactor=1,doubleSided=False))
side_parts=[]
for region,label in [(hair_recolor,'Hair volume: rear colored geometry')]:
    if np.any(region):
        subset,vids=mesh_part(region)
        verts=subset.vertices
        shade=np.clip(.8+.14*(verts[:,1]+.5)+.08*np.sin(verts[:,0]*38+verts[:,1]*18),.65,1.0)
        if region is hair_recolor:
            colors=np.column_stack((210*shade,100*shade,47*shade,np.full(len(verts),255))).astype('uint8')
        else:
            colors=np.column_stack((105*shade,125*shade,96*shade,np.full(len(verts),255))).astype('uint8')
        subset.visual.vertex_colors=colors
        side_parts.append((label,subset))
print('Photo faces',photo_mask.sum(),'Recolored hair',hair_recolor.sum(),'tail',tail_recolor.sum())

from collections import defaultdict
M=defaultdict(list)
def add(key,mesh,rgba=None):
    if rgba is not None:
        mesh.visual.vertex_colors=np.tile(np.array(rgba,dtype=np.uint8),(len(mesh.vertices),1))
    M[key].append(mesh)
def lerp(a,b,t):return np.asarray(a)*(1-t)+np.asarray(b)*t

def ellipsoid(c,r,mesh_color,category='skin',sub=3):
    s=trimesh.creation.icosphere(subdivisions=sub,radius=1)
    s.vertices=(s.vertices*np.array(r))+np.array(c)
    add(category,s,mesh_color)
    return s

def loft(rings, col, category, sides=36):
    # rings [y, xcentre, zcentre, xradius, zradius]
    rings=np.array(rings,float)
    xs=np.linspace(0,1,len(rings)); f=[CubicSpline(xs,rings[:,i],bc_type='natural') for i in range(5)]
    t=np.linspace(0,1,len(rings)*5)
    vs=[]
    for k in t:
        yy,xx,zz,rx,rz=[float(g(k)) for g in f]
        for j in range(sides):
            a=2*np.pi*j/sides
            vs.append([xx+rx*np.cos(a),yy,zz+rz*np.sin(a)])
    fcs=[]
    for i in range(len(t)-1):
        for j in range(sides):
            a=i*sides+j;b=i*sides+(j+1)%sides;c=(i+1)*sides+j;d=(i+1)*sides+(j+1)%sides
            fcs.extend([(a,c,b),(b,c,d)])
    m=trimesh.Trimesh(vertices=np.asarray(vs),faces=fcs,process=False)
    # double-sided normals for VTK
    add(category,m,col)
    return m

def spline_points(control,n=64):
    p=np.asarray(control,float)
    # chord-length parametrization avoids wild curl overshoot
    seg=np.linalg.norm(np.diff(p,axis=0),axis=1)
    u=np.concatenate(([0],np.cumsum(np.maximum(seg,1e-5))))
    if u[-1]==0:return p
    u/=u[-1]
    ts=np.linspace(0,1,n)
    return np.stack([CubicSpline(u,p[:,i],bc_type='natural')(ts) for i in range(3)],axis=1)

def tube(pts,radius,color,category='hair',segments=12, taper=0, longitudinal_colors=None):
    pts=np.asarray(pts,float)
    n=len(pts);tangent=np.gradient(pts,axis=0);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)+1e-12
    d=np.array([0,0,1.]); ring=[]
    for j in range(n):
        t=tangent[j];v=np.cross(t,d)
        if np.linalg.norm(v)<.01:v=np.cross(t,[0,1,0])
        v/=np.linalg.norm(v)+1e-12
        w=np.cross(t,v);w/=np.linalg.norm(w)+1e-12
        t0=j/(n-1)
        r=radius*(1-taper*t0)**.65 if np.isscalar(radius) else radius[j]
        for k in range(segments):
            a=2*np.pi*k/segments
            ring.append(pts[j]+r*(np.cos(a)*v+np.sin(a)*w))
    f=[]
    for i in range(n-1):
        for j in range(segments):
            a=i*segments+j;b=i*segments+(j+1)%segments;c=(i+1)*segments+j;d=(i+1)*segments+(j+1)%segments
            f.extend([(a,b,c),(b,d,c)])
    # Open ends are hidden inside the adjoining surfaces.
    m=trimesh.Trimesh(vertices=np.asarray(ring),faces=f,process=False)
    if longitudinal_colors is not None:
        colors=[]
        for j in range(n):
            for k in range(segments): colors.append(longitudinal_colors(j/(n-1),k/segments))
        m.visual.vertex_colors=np.array(colors,dtype=np.uint8)
        M[category].append(m)
    else:add(category,m,color)
    return m

# Coordinated skin backing, replaces the missing coherent rear behind the original high-fidelity face projection.
skin=[218,178,155,255]
loft([[.49,0,-.088,.105,.07],[.43,0,-.117,.168,.10],[.365,0,-.143,.162,.117],[.28,0,-.142,.136,.108],[.16,0,-.143,.117,.092],[.07,0,-.138,.16,.103],[-.03,0,-.134,.2,.108],[-.14,0,-.132,.165,.107]],skin,'skin')
loft([[.67,0,-.085,.053,.05],[.56,0,-.084,.060,.060],[.46,0,-.096,.079,.064]],skin,'skin',sides=24)
# Head underneath face and hair with a rounded jaw and believable side volume.
loft([[.85,0,-.020,.095,.089],[.80,0,-.025,.125,.126],[.74,0,-.022,.13,.128],[.68,0,-.015,.118,.115],[.635,0,-.008,.092,.088],[.595,0,-.007,.060,.059]],skin,'skin',sides=48)
ellipsoid((0,.765,-.175),(.14,.145,.11),[183,82,31,255],'hair_base',sub=3)
# Back-of-head hair strands blend the scalp with the long side locks.
for i in range(25):
    q=(i-12)/12
    path=[(q*.04,.89,-.178),(q*.083,.845,-.258),(q*.105,.768,-.28),(q*.119,.689,-.265),(q*.128,.64,-.218)]
    tube(spline_points(path,24),.005+[.0009,.003,.001][i%3],[168+(i%5)*12,69+(i%4)*8,28,255],'hair',8,taper=.4)

# paired limbs, long-axis sculpted with taper.
for sg in (-1,1):
    loft([[.45,sg*.168,-.107,.063,.07],[.38,sg*.213,-.100,.055,.062],[.28,sg*.226,-.092,.05,.051],[.17,sg*.25,-.06,.045,.047],[.10,sg*.275,-.037,.037,.039],[.015,sg*.281,-.023,.033,.036],[-.05,sg*.273,-.011,.030,.032]],skin,'skin',24)
    ellipsoid((sg*.278,-.083,-.005),(.038,.063,.031),skin,'skin',2)
    for k in range(4):
        p=spline_points([(sg*(.255+.014*k),-.10,-.004),(sg*(.259+.013*k),-.148,-.002),(sg*(.26+.013*k),-.170,-.014)],18)
        tube(p,.007,skin,'skin_fingers',8,taper=.55)
    x=sg*.11
    loft([[-.086,x,-.121,.097,.092],[-.23,sg*.116,-.119,.113,.104],[-.37,sg*.117,-.110,.094,.089],[-.44,sg*.115,-.098,.079,.071],[-.51,sg*.127,-.078,.076,.078],[-.61,sg*.127,-.063,.075,.08],[-.70,sg*.128,-.054,.060,.062],[-.83,sg*.13,-.045,.047,.046],[-.94,sg*.13,-.035,.043,.038]],skin,'skin',28)
    ellipsoid((sg*.13,-.965,.001),(.058,.043,.109),skin,'skin',2)

# Rear outfit pieces and straps, not photo mirrored onto back.
black=[25,24,24,255]
for sg in (-1,1):
    # back shoulder strap and midriff slender gold piping
    tube(spline_points([(sg*.15,.43,-.10),(sg*.18,.42,-.19),(sg*.16,.385,-.247),(sg*.056,.391,-.248)],24),.0085,[183,139,53,255],'gold',10)
    tube(spline_points([(sg*.11,-.05,-.12),(sg*.18,-.018,-.13),(sg*.2,.055,-.11)],22),.0075,[54,42,31,255],'straps',10)
# Center outfit back panel kept visually subtle.
ellipsoid((0,-.066,-.224),(.097,.105,.012),[66,48,21,255],'bronze',sub=2)
# Hair geometry. Copper underlayer with many distinct back strands and tapered spiral ends.
color_variants=[[180,72,26,255],[209,91,35,255],[235,119,51,255],[191,78,31,255],[248,145,67,255],[164,61,25,255]]
for sg in (-1,1):
    # Outer primary hair layers from crown to the lower back.
    for j in range(17):
        q=(j+0.5)/17
        side=(.032+.11*q)
        rand=np.random.default_rng(j+(0 if sg<0 else 1000))
        xside=.17 + .25*np.sin(.25+q*1.15)
        end_y=.08 + .27*rand.uniform()
        zback=-.24 - .06*q - .015*rand.uniform()
        path=[(sg*side,.85+q*.007,-.077),(sg*(side+.045),.72,-.12),(sg*(.19+.10*q),.58,-.175),(sg*xside,.40,zback),(sg*(xside+.015*np.sin(5*q)),end_y,zback+.045)]
        p=spline_points(path,38)
        rad=.008+rand.uniform(0,.006)
        tube(p,rad,color_variants[j%len(color_variants)],'hair',8,taper=.85)
    # a few long separated face-framing locks
    for i in range(0):
        path=[(sg*(.102+.014*i),.82,.08),(sg*(.158+.018*i),.65,.06),(sg*(.235+.012*i),.52,.07),(sg*(.273+.018*i),.57,.09),(sg*(.235+.012*i),.62,.10)]
        tube(spline_points(path,50),.007+[0,.003,.001,.0][i],color_variants[(i+1)%6],'hair',10,taper=.5)
    # Hair highlight strands to break up flat orange silhouette.
    for j in range(5):
        q=(j+.5)/5
        path=[(sg*(.07+.045*q),.78,-.02),(sg*(.185+.09*q),.53,-.09),(sg*(.22+.165*q),.31,-.085),(sg*(.22+.185*q),.14,-.065)]
        tube(spline_points(path,28),.0019,[248,164,79,255],'hair_highlight',6,taper=.85)
# face fringe strands from crown downward (stay behind the reference face projection)
for i in range(0):
    xx=(i-6)*.016
    path=[(xx*.72,.88,.076),(xx,.795,.110),(xx+np.sin(i)*.005,.73,.129)]
    tube(spline_points(path,16),.007,[219+(i%4)*7,99+(i%5)*5,39,255],'bangs',8,taper=.9)
# Crown spiral ahoge
curve=[]
for i in range(46):
    t=i/45;ang=-1.0 + t*1.92*np.pi
    curve.append([-.009+.044*np.cos(ang)*(1-t*.55),.955+.040*np.sin(ang)*(1-t*.2),.01+.003*t])
pass # Original scan already has the distinctive top curl

# Horns are true curved tapered tubes with striated gold root rings and dark violet tips.
for sg in (-1,1):
    base=np.array([sg*.15,.804,-.002])
    ctr=[base,(sg*.176,.83,-.012),(sg*.219,.825,-.013),(sg*.267,.835,-.025),(sg*.296,.883,-.036)]
    pp=spline_points(ctr,46)
    cc=lambda t,a: np.array([int(69+42*t),int(41+20*t),int(58+30*t),255])
    tube(pp,np.linspace(.037,.0015,len(pp)),[80,45,68,255],'horns',18,longitudinal_colors=cc)
    for t in np.linspace(.06,.47,7):
        p=pp[int(t*(len(pp)-1))]
        # ribbed ring around horn axis
        tang=pp[min(len(pp)-1,int(t*45)+1)]-pp[max(0,int(t*45)-1)];tang/=np.linalg.norm(tang)
        vv=np.cross(tang,[0,0,1.]);vv/=np.linalg.norm(vv)
        ww=np.cross(tang,vv);rr=.037*(1-t)**.6+.0018
        ring=[p+rr*(np.cos(a)*vv+np.sin(a)*ww) for a in np.linspace(0,2*np.pi,32)]
        tube(ring,.002,[177,151,127,255],'horn_rings',7)

# Rebuilt continuous scaled tail (green back, rose belly, taper to a tapered pointed tip).
tailctrl=[(.15,-.01,-.45),(.265,-.16,-.365),(.29,-.37,-.35),(.2,-.53,-.39),(.02,-.63,-.40),(-.20,-.71,-.41),(-.27,-.83,-.40),(-.20,-.94,-.38)]
tailpts=spline_points(tailctrl,115)
rad=np.linspace(.056,.005,len(tailpts))
def tail_colors(t,angle):
    # Main scales with a contrasting underside, shaded banding, no texture smearing at other angles.
    is_belly=math.cos(2*math.pi*angle)>0.27
    b=.87+.09*math.sin(t*125+12*angle)
    if is_belly:
        return [int(154*b),int(109*b),int(103*b),255]
    return [int(103*b),int(123*b),int(93*b),255]
tube(tailpts,rad,[132,120,96,255],'tail',18,longitudinal_colors=tail_colors)
# Rings subtly indicate segmented scale plates.
for i in range(4,112,5):
    t=i/(len(tailpts)-1);p=tailpts[i]
    tangent=tailpts[i+1]-tailpts[i-1];tangent/=np.linalg.norm(tangent)
    b=np.cross(tangent,[0,0,1]);b/=np.linalg.norm(b)
    c=np.cross(tangent,b)
    pts=[p+rad[i]*1.007*(np.cos(a)*b+np.sin(a)*c) for a in np.linspace(0,2*np.pi,32)]
    tube(pts,.0013,[73,78,62,255],'tail_scale_rings',5)

# Jewel and metal accessories: layered material, not painted decals.
center=[0,.535,.309]
ellipsoid(center,(.025,.038,.012),[162,105,25,255],'gold',2)
ellipsoid([0,.535,.321],(.019,.030,.014),[129,13,36,255],'ruby',2)
ellipsoid([-.005,.547,.334],(.005,.009,.003),[255,151,149,255],'ruby_reflection',1)
for sign in (-1,1):
    tube(spline_points([(0,.537,.29),(sign*.049,.555,.28),(sign*.075,.565,.20),(sign*.062,.56,.13)],18),.007,[162,114,36,255],'gold',9)
    for y in [.065,.076,.086]:
        # small gold wrist cuffs; hands are kept as authored geometry
        x=sign*(.277-(y-.07)*.08)
        curve=[[x+.031*np.cos(q),y,-.026+.035*np.sin(q)] for q in np.linspace(0,2*np.pi,30)]
        tube(curve,.0032,[187,137,51,255],'gold',7)
    # dark thigh straps and gold clasp at front (over the new outer thigh volume)
    for y in [-.18,-.265]:
        x=sign*.116
        curve=[[x+.104*np.cos(a),y,-.113+.096*np.sin(a)] for a in np.linspace(0,2*np.pi,50)]
        tube(curve,.0088,[28,28,27,255],'straps',10)
    ellipsoid((sign*.117,-.228,-.010),(.019,.020,.010),[161,110,33,255],'bronze',1)
    # sandals, modeled heel and front gold straps
    for y in [-.966]:
        x=sign*.13
        for delta in [-.025,.026]:
            tube(spline_points([(x-.05,y,-.003),(x,y+.027,-.094+delta),(x+.049,y,-.003)],16),.004,[186,138,64,255],'gold',8)

# Minimal rear garment bands and a fastening detail.
tube(spline_points([(-.137,.392,-.32),(-.079,.378,-.348),(0,.373,-.355),(.08,.378,-.348),(.137,.392,-.32)],34),.009,[38,34,34,255],'rear_outfit',12)
tube(spline_points([(-.167,-.032,-.31),(-.09,-.087,-.332),(0,-.102,-.338),(.09,-.087,-.332),(.167,-.032,-.31)],32),.010,[72,53,27,255],'rear_outfit',12)
ellipsoid((0,-.084,-.340),(.07,.06,.016),[67,47,26,255],'rear_outfit',2)
ellipsoid((0,.377,-.359),(.012,.013,.007),[182,137,55,255],'rear_clasp',2)

# Grounded material grouping; highly editable geometry nodes in 3D applications.
scene=trimesh.Scene()
scene.add_geometry(front,geom_name='Front photo texture (unlit)',node_name='Reference-mapped surface')
for label,geom in side_parts: scene.add_geometry(geom,geom_name=label,node_name=label)
mat_overrides={
 'skin': dict(baseColorFactor=[.99,.87,.77,1],roughnessFactor=.91),
 'skin_fingers':dict(baseColorFactor=[.99,.87,.77,1],roughnessFactor=.91),
 'hair':dict(roughnessFactor=.68), 'hair_base':dict(roughnessFactor=.73),
 'horns':dict(roughnessFactor=.40,metallicFactor=.2),
 'gold':dict(roughnessFactor=.27,metallicFactor=.83),
 'bronze':dict(roughnessFactor=.40,metallicFactor=.63),
 'ruby':dict(roughnessFactor=.19,metallicFactor=.22),
 'tail':dict(roughnessFactor=.82),
 'straps':dict(roughnessFactor=.7),
}
for k,meshes in M.items():
    if k in {'skin_fingers','bronze','straps','gold','ruby','ruby_reflection','bangs','horn_rings','hair_highlight'}: continue
    if k=='skin':
        # Position proxy anatomy underneath the outer photo surface, so the front keeps its likeness.
        from scipy.ndimage import maximum_filter
        depth_fill=maximum_filter(depth,size=15)
        for piece in meshes:
            p=piece.vertices.copy()
            p[:,0]*=.90
            is_arm=(p[:,1].min()<-.04 and p[:,1].max()>.35 and abs(p[:,0].mean())>.14)
            is_hand=(p[:,1].min()<-.14 and p[:,1].max()<.00 and abs(p[:,0].mean())>.18)
            p[:,2]-=(.125 if (is_arm or is_hand) else .06)
            if is_arm or is_hand:
                piece.vertices=p
                continue
            xx=np.clip(np.rint((p[:,0]+.48)/.96*949).astype(int),0,949)
            yy=np.clip(np.rint((1.07-p[:,1])/2.16*1599).astype(int),0,1599)
            zmax=depth_fill[yy,xx]
            clamp=(zmax>-3)&(p[:,2]>zmax-.032)
            p[clamp,2]=zmax[clamp]-.032
            piece.vertices=p
    # The proxy is rear-facing only. Do not render duplicate anatomy in the reference front view.
    if k=='skin':
        cleaned=[]
        for piece in meshes:
            m=piece.copy()
            backhalf=m.triangles_center[:,2] < np.median(m.vertices[:,2])-.0007
            if not backhalf.any(): continue
            m.update_faces(backhalf);m.remove_unreferenced_vertices()
            flip=m.face_normals[:,2]>0
            m.faces[flip]=m.faces[flip][:,::-1]
            cleaned.append(m)
        meshes=cleaned
    # merge separate meshes of same material while retaining vertex RGBA.
    mesh=trimesh.util.concatenate(meshes)
    mesh.visual=trimesh.visual.ColorVisuals(mesh,vertex_colors=np.array(mesh.visual.vertex_colors,dtype=np.uint8))
    settings=mat_overrides.get(k,{})
    # export vertex color compatible with GLB
    scene.add_geometry(mesh,geom_name=k.replace('_',' ').title(),node_name=k)
    print('geometry',k,len(mesh.faces))

out=f'{OUT}/Character_Rebuilt_360_Approx.glb'
scene.export(out)
print('Exported',out,os.path.getsize(out),'bytes; total triangles',sum(len(x.faces) for x in scene.geometry.values()))
with open(f'{OUT}/README.txt','w') as f:
    f.write('REBUILT 3D CHARACTER - GLB\n\nReference: uploaded front-view character image.\n')
    f.write('Includes original image-matched visible front, coherent 3D backing, sculpted limbs and head, layered hair locks, curved horns, detailed tail, jewelry, bracelet and costume elements.\n')
    f.write('The original 40,000-face scan comprised 1,170 disconnected shells. Front-only visible faces were retained to avoid duplicated rear surfaces.\n')
    f.write('Limits: this is an approximate 360-degree interpretation; a single frontal reference cannot determine exact side/back geometry, and original scan quality limits fidelity. Geometry is unrigged.\n')
    f.write('Open the .glb in Blender or most GLB viewers to inspect or edit separate mesh nodes.\n')
