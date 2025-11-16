#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador sintético de escenarios DANA (España) con salidas raster COG + visualización + recorte.
- Genera dominio GRANDE (p.ej., 25x25 km) y opcionalmente recorta a un subdominio (p.ej., 3x2 km).
- Variables: lluvia (R), rachas (G), rayos (L) y acumulados (P1H, P6H, P24H, P_event).
"""
import json, os, math, argparse
from dataclasses import dataclass, asdict
import numpy as np
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

# ----------------------------- Utilidades -----------------------------------

def set_seed(seed: int): np.random.seed(seed)

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def moving_average2d(arr, k=3):
    if k <= 1: return arr
    pad = k // 2
    arr_p = np.pad(arr, ((pad,pad),(pad,pad)), mode="edge")
    kernel = np.ones((k,k))/ (k*k)
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            patch = arr_p[i:i+k,j:j+k]
            out[i,j] = np.sum(patch*kernel)
    return out

def roll_advection(field, sx, sy):
    fx, fy = int(round(sx)), int(round(sy))
    adv = np.roll(field, shift=(fy,fx), axis=(0,1))
    return moving_average2d(adv, k=3)

def gaussian_random_field(shape, lc_px, alpha=3.0):
    ny,nx = shape
    # Cap de lc relativo a la malla
    lc_px_eff = max(1.0, min(lc_px, 0.3 * min(ny, nx)))
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    kx, ky = np.meshgrid(kx,ky)
    k = np.sqrt(kx**2+ky**2)
    k[0,0]=1e-6
    spec = 1.0/(k**alpha)
    sigma = 1.0/(2*math.pi*lc_px_eff)
    gauss = np.exp(-(k**2)/(2*sigma**2))
    spec *= gauss
    noise = (np.random.normal(size=(ny,nx))+1j*np.random.normal(size=(ny,nx)))
    f_field = noise*np.sqrt(spec)
    field = np.fft.ifft2(f_field).real
    std = np.std(field)
    if std < 1e-6:
        field = np.random.normal(size=(ny,nx))
        std = np.std(field)
    field = (field-np.mean(field))/(std+1e-6)
    # Mezcla leve con ruido blanco para robustez
    eps = 0.05
    white = np.random.normal(size=(ny,nx))
    white = (white - white.mean())/(white.std()+1e-6)
    field = (1-eps)*field + eps*white
    return field.astype(np.float32)

# ----------------------- Configuración de escenario -------------------------
@dataclass
class ScenarioConfig:
    bbox: tuple
    crs: str = 'EPSG:25830'
    dx: float = 250.0
    dt_minutes: int = 10
    hours: int = 48
    severity: str = 'p90'
    seed: int = 12345
    def __post_init__(self):
        self.target_p_event_mm = {"p50":150.0,"p90":300.0,"p99":450.0}
        self.target_r_peak_mm_h = {"p50":30.0,"p90":60.0,"p99":110.0}
        self.target_gust_ms     = {"p50":15.0,"p90":25.0,"p99":35.0}
        self.target_lambda_fl_km2_h = {"p50":0.2,"p90":1.0,"p99":3.0}

def generate_time_index(cfg):
    return np.arange(int((cfg.hours*60)/cfg.dt_minutes))

# ---------------------------- Campos sintéticos -----------------------------

def rainfall_stack(cfg,ny,nx):
    T=len(generate_time_index(cfg))
    R=np.zeros((T,ny,nx),dtype=np.float32)
    # Longitud de correlación (en px); para tiles grandes puede ser 8–18 km
    lc_km_map = {"p50": 8.0, "p90": 12.0, "p99": 18.0}
    lc_px=(lc_km_map[cfg.severity]*1000.0)/cfg.dx
    # Probabilidad de activación convectiva por paso
    base_rate={"p50":1.0,"p90":0.9,"p99":0.8}[cfg.severity]
    p_on=min(1.0, base_rate*(cfg.dt_minutes/60.0))
    # Advección SW->NE típica
    adv_pix_per_step={"p50":0.3,"p90":0.5,"p99":0.7}[cfg.severity]
    sx=adv_pix_per_step*math.cos(math.radians(225))
    sy=adv_pix_per_step*math.sin(math.radians(225))
    # Patrón base
    base=gaussian_random_field((ny,nx),lc_px=lc_px)
    base=(base-base.min())/(base.max()-base.min()+1e-6)
    r_peak=cfg.target_r_peak_mm_h[cfg.severity]
    for t in range(T):
        if np.random.rand()<p_on:
            intensity=np.random.lognormal(mean=np.log(max(r_peak/2.0,1e-3)),sigma=0.5)
            field=roll_advection(base,sx*t,sy*t)
            field=np.power(field,0.6)
            R[t]=(field*intensity).astype(np.float32)
        else:
            R[t]=(0.05*base).astype(np.float32)
    # Normalización del acumulado
    step_h=cfg.dt_minutes/60.0
    accum=R.sum(axis=0)*step_h
    mean_accum=float(np.mean(accum))
    p_target=float(cfg.target_p_event_mm[cfg.severity])
    if mean_accum>1e-6:
        scale=p_target/mean_accum
        R*=scale
    else:
        R[int(T/2)] += (p_target/step_h)/(ny*nx)
    return R

def gust_stack(cfg,R):
    T,ny,nx=R.shape
    G=np.zeros_like(R)
    base_gust=cfg.target_gust_ms[cfg.severity]
    for t in range(T):
        noise=gaussian_random_field((ny,nx),lc_px=10.0,alpha=2.5)
        noise=(noise-noise.min())/(noise.max()-noise.min()+1e-6)
        conv=R[t]/(np.percentile(R[t],95)+1e-6)
        conv=np.clip(conv,0,2.0)
        G[t]=base_gust*(0.6+0.4*noise)*(1.0+0.8*conv)
    return G.astype(np.float32)

def lightning_stack(cfg,R):
    T,ny,nx=R.shape
    L=np.zeros_like(R)
    lam_target=cfg.target_lambda_fl_km2_h[cfg.severity]
    R95=max(np.percentile(R,95),1e-3)
    k=lam_target/R95
    for t in range(T):
        lam=k*np.maximum(R[t]-5.0,0.0)
        L[t]=np.random.poisson(lam).astype(np.float32)
    return L

def accum_products(cfg,R):
    T=R.shape[0]; step_h=cfg.dt_minutes/60.0
    def rolling_sum(arr_t,wh):
        w=max(1,int(round(wh/step_h)))
        out=np.zeros_like(arr_t)
        csum=np.cumsum(arr_t,axis=0)
        for i in range(T):
            j0=max(0,i-w+1)
            out[i]=csum[i]-(csum[j0-1] if j0>0 else 0)
        return out
    P1H = rolling_sum(R*step_h,1.0).astype(np.float32)
    P6H = rolling_sum(R*step_h,6.0).astype(np.float32)
    P24H= rolling_sum(R*step_h,24.0).astype(np.float32)
    P_event = np.sum(R*step_h,axis=0,keepdims=True).astype(np.float32)
    return {"P1H":P1H,"P6H":P6H,"P24H":P24H,"P_event":P_event}

# ---------------------------- Escritura & BBox -------------------------------

def make_transform_and_shape(cfg):
    xmin,ymin,xmax,ymax=cfg.bbox
    nx=int(math.ceil((xmax-xmin)/cfg.dx))
    ny=int(math.ceil((ymax-ymin)/cfg.dx))
    return (ny,nx), from_origin(xmin,ymax,cfg.dx,cfg.dx)


def write_cog(path,stack,transform,crs,units,meta):
    ensure_dir(os.path.dirname(path))
    T,ny,nx=stack.shape
    prof={'driver':'GTiff','height':ny,'width':nx,'count':T,'dtype':'float32','crs':crs,'transform':transform,'tiled':True,'compress':'DEFLATE','interleave':'band','blockxsize':512,'blockysize':512}
    with rasterio.open(path,'w',**prof) as dst:
        for i in range(T): dst.write(stack[i],i+1)
        tags={'UNITS':units}
        tags.update(meta)
        dst.update_tags(**tags)


def crop_stack(stack, base_bbox, dx, crop_bbox):
    """Recorta un stack (T,ny,nx) a un bbox. Alinea al grid más cercano."""
    xmin, ymin, xmax, ymax = base_bbox
    cxmin, cymin, cxmax, cymax = crop_bbox
    T, NY, NX = stack.shape
    col0 = int(np.floor((cxmin - xmin) / dx))
    col1 = int(np.ceil((cxmax - xmin) / dx))
    row0 = int(np.floor((ymax - cymax) / dx))
    row1 = int(np.ceil((ymax - cymin) / dx))
    col0 = max(0, min(NX, col0)); col1 = max(0, min(NX, col1))
    row0 = max(0, min(NY, row0)); row1 = max(0, min(NY, row1))
    if col1 <= col0 or row1 <= row0:
        raise ValueError("crop_bbox fuera del dominio o vacío")
    sub = stack[:, row0:row1, col0:col1]
    rxmin = xmin + col0 * dx
    rymax = ymax - row0 * dx
    transform_crop = from_origin(rxmin, rymax, dx, dx)
    return sub, transform_crop

# ------------------------------- Visualización ------------------------------

def quick_viz(outdir,R=None,G=None,prefix=''):
    ensure_dir(outdir)
    if R is not None:
        accum=R.sum(axis=0)*(10/60)
        rmax=np.max(R,axis=0)
        fig,ax=plt.subplots(1,2,figsize=(9,4))
        im0=ax[0].imshow(accum,cmap='Blues'); ax[0].set_title('Lluvia total (mm)'); plt.colorbar(im0,ax=ax[0])
        im1=ax[1].imshow(rmax,cmap='inferno'); ax[1].set_title('Máx. instantáneo (mm/h)'); plt.colorbar(im1,ax=ax[1])
        plt.tight_layout(); plt.savefig(os.path.join(outdir,f'quickview_rain{prefix}.png'),dpi=150); plt.close(fig)
    if G is not None:
        gmax=np.max(G,axis=0)
        fig,ax=plt.subplots(1,1,figsize=(5,4))
        im=ax.imshow(gmax); ax.set_title('Racha máxima (m/s)'); plt.colorbar(im,ax=ax)
        plt.tight_layout(); plt.savefig(os.path.join(outdir,f'quickview_gust{prefix}.png'),dpi=150); plt.close(fig)

# ------------------------------- Orquestador --------------------------------

def build_and_write(cfg,outdir, write_r=True, write_g=True, write_l=True, write_accum=True, crop_bbox=None):
    set_seed(cfg.seed)
    (ny,nx),transform=make_transform_and_shape(cfg)
    base_bbox = cfg.bbox
    meta={'SEVERITY':cfg.severity,'HOURS':cfg.hours,'DT_MIN':cfg.dt_minutes,'CRS':cfg.crs}

    # 1) Lluvia
    R=rainfall_stack(cfg,ny,nx)
    # 2) Rachas y Rayos
    G=gust_stack(cfg,R) if write_g else None
    L=lightning_stack(cfg,R) if write_l else None
    # 3) Acumulados
    ACC=accum_products(cfg,R) if write_accum else {}

    ensure_dir(outdir)

    # --- Escritura dominio completo ---
    if write_r:
        write_cog(os.path.join(outdir,'R_hourly_COG.tif'),R,transform,cfg.crs,'mm/h',meta)
    if write_g and G is not None:
        write_cog(os.path.join(outdir,'G_gust10_COG.tif'),G,transform,cfg.crs,'m/s',meta)
    if write_l and L is not None:
        write_cog(os.path.join(outdir,'L_flashdens_COG.tif'),L,transform,cfg.crs,'fl/km2/h',meta)
    if write_accum and ACC:
        write_cog(os.path.join(outdir,'P1H_accum_COG.tif'),ACC['P1H'],transform,cfg.crs,'mm',meta)
        write_cog(os.path.join(outdir,'P6H_accum_COG.tif'),ACC['P6H'],transform,cfg.crs,'mm',meta)
        write_cog(os.path.join(outdir,'P24H_accum_COG.tif'),ACC['P24H'],transform,cfg.crs,'mm',meta)
        write_cog(os.path.join(outdir,'P_event_COG.tif'),ACC['P_event'],transform,cfg.crs,'mm',meta)

    quick_viz(outdir,R=R,G=G,prefix='')

    # --- Recorte opcional ---
    if crop_bbox is not None:
        if write_r:
            R_c, tr_c = crop_stack(R, base_bbox, cfg.dx, crop_bbox)
            write_cog(os.path.join(outdir,'R_hourly_COG_CROP.tif'), R_c, tr_c, cfg.crs, 'mm/h', meta)
        if write_g and G is not None:
            G_c, tr_g = crop_stack(G, base_bbox, cfg.dx, crop_bbox)
            write_cog(os.path.join(outdir,'G_gust10_COG_CROP.tif'), G_c, tr_g, cfg.crs, 'm/s', meta)
        if write_l and L is not None:
            L_c, tr_l = crop_stack(L, base_bbox, cfg.dx, crop_bbox)
            write_cog(os.path.join(outdir,'L_flashdens_COG_CROP.tif'), L_c, tr_l, cfg.crs, 'fl/km2/h', meta)
        if write_accum and ACC:
            for k,v in ACC.items():
                V_c, tr_a = crop_stack(v, base_bbox, cfg.dx, crop_bbox)
                write_cog(os.path.join(outdir,f'{k}_accum_COG_CROP.tif'), V_c, tr_a, cfg.crs, 'mm', meta)
        # Quickviews del recorte
        quick_viz(outdir, R=(R_c if write_r else None), G=(G_c if (write_g and G is not None) else None), prefix='_CROP')

# ------------------------------- CLI ----------------------------------------

def parse_args():
    p=argparse.ArgumentParser(description='Generador sintético de escenarios DANA (raster COG) con recorte opcional.')
    p.add_argument('--bbox',nargs=4,type=float,required=True, help='Dominio grande: xmin ymin xmax ymax')
    p.add_argument('--crop_bbox',nargs=4,type=float, help='Subdominio a recortar: xmin ymin xmax ymax (opcional)')
    p.add_argument('--outdir',required=True)
    p.add_argument('--crs',default='EPSG:25830')
    p.add_argument('--dx',type=float,default=250.0)
    p.add_argument('--hours',type=int,default=48)
    p.add_argument('--dt',type=int,default=10)
    p.add_argument('--severity',choices=['p50','p90','p99'],default='p90')
    p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--no_wind',action='store_true')
    p.add_argument('--no_lightning',action='store_true')
    p.add_argument('--no_accum',action='store_true')
    return p.parse_args()

def main_cli():
    a=parse_args()
    cfg=ScenarioConfig(bbox=tuple(a.bbox),crs=a.crs,dx=a.dx,hours=a.hours,dt_minutes=a.dt,severity=a.severity,seed=a.seed)
    crop = tuple(a.crop_bbox) if a.crop_bbox else None
    build_and_write(cfg,a.outdir, write_r=True, write_g=(not a.no_wind), write_l=(not a.no_lightning), write_accum=(not a.no_accum), crop_bbox=crop)

if __name__=='__main__':
    main_cli()
