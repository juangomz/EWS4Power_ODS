#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador sintético de escenarios DANA (España) con salidas raster COG + visualización rápida.
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
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    kx, ky = np.meshgrid(kx,ky)
    k = np.sqrt(kx**2+ky**2)
    k[0,0]=1e-6
    spec = 1.0/(k**alpha)
    sigma = 1.0/(2*math.pi*max(lc_px,1.0))
    gauss = np.exp(-(k**2)/(2*sigma**2))
    spec *= gauss
    noise = (np.random.normal(size=(ny,nx))+1j*np.random.normal(size=(ny,nx)))
    f_field = noise*np.sqrt(spec)
    field = np.fft.ifft2(f_field).real
    field = (field-np.mean(field))/(np.std(field)+1e-6)
    return field.astype(np.float32)

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
        self.target_p_event_mm = {"p50":150,"p90":300,"p99":450}
        self.target_r_peak_mm_h = {"p50":30,"p90":60,"p99":110}
        self.target_gust_ms = {"p50":15,"p90":25,"p99":35}
        self.target_lambda_fl_km2_h = {"p50":0.2,"p90":1.0,"p99":3.0}

def generate_time_index(cfg):
    return np.arange(int((cfg.hours*60)/cfg.dt_minutes))

def rainfall_stack(cfg,ny,nx):
    T=len(generate_time_index(cfg))
    R=np.zeros((T,ny,nx),dtype=np.float32)
    lc_px=( {"p50":8,"p90":12,"p99":18}[cfg.severity]*1000)/cfg.dx
    p_on=min(1.0,{"p50":1.0,"p90":0.7,"p99":0.5}[cfg.severity]*(cfg.dt_minutes/60))
    adv_pix_per_step={"p50":0.3,"p90":0.5,"p99":0.7}[cfg.severity]
    sx=adv_pix_per_step*math.cos(math.radians(225))
    sy=adv_pix_per_step*math.sin(math.radians(225))
    base=gaussian_random_field((ny,nx),lc_px=lc_px)
    base=(base-base.min())/(base.max()-base.min()+1e-6)
    r_peak=cfg.target_r_peak_mm_h[cfg.severity]
    for t in range(T):
        if np.random.rand()<p_on:
            intensity=np.random.lognormal(mean=np.log(r_peak/2+1e-6),sigma=0.5)
            field=roll_advection(base,sx*t,sy*t)
            field=np.power(field,0.6)
            R[t]=(field*intensity).astype(np.float32)
        else:
            R[t]=(0.1*base).astype(np.float32)
    p_target=cfg.target_p_event_mm[cfg.severity]
    accum=R.sum(axis=0)*(cfg.dt_minutes/60)
    scale=p_target/np.mean(accum)
    R*=scale
    return R

def make_transform_and_shape(cfg):
    xmin,ymin,xmax,ymax=cfg.bbox
    nx=int(math.ceil((xmax-xmin)/cfg.dx))
    ny=int(math.ceil((ymax-ymin)/cfg.dx))
    return (ny,nx), from_origin(xmin,ymax,cfg.dx,cfg.dx)

def write_cog(path,stack,transform,crs,units,meta):
    ensure_dir(os.path.dirname(path))
    T,ny,nx=stack.shape
    prof={'driver':'GTiff','height':ny,'width':nx,'count':T,'dtype':'float32','crs':crs,'transform':transform,'tiled':True,'compress':'DEFLATE'}
    with rasterio.open(path,'w',**prof) as dst:
        for i in range(T): dst.write(stack[i],i+1)
        dst.update_tags(UNITS=units,**meta)

def quick_viz(R,outdir):
    accum=R.sum(axis=0)*(10/60)
    rmax=np.max(R,axis=0)
    fig,ax=plt.subplots(1,2,figsize=(8,4))
    im0=ax[0].imshow(accum,cmap='Blues');ax[0].set_title('Lluvia total (mm)');plt.colorbar(im0,ax=ax[0])
    im1=ax[1].imshow(rmax,cmap='inferno');ax[1].set_title('Máx. instantáneo (mm/h)');plt.colorbar(im1,ax=ax[1])
    plt.tight_layout();plt.savefig(os.path.join(outdir,'quickview.png'),dpi=150)
    plt.close(fig)

def build_and_write(cfg,outdir):
    set_seed(cfg.seed)
    (ny,nx),transform=make_transform_and_shape(cfg)
    R=rainfall_stack(cfg,ny,nx)
    meta={'SEVERITY':cfg.severity,'HOURS':cfg.hours}
    write_cog(os.path.join(outdir,'R_hourly_COG.tif'),R,transform,cfg.crs,'mm/h',meta)
    quick_viz(R,outdir)

def main_cli():
    p=argparse.ArgumentParser()
    p.add_argument('--bbox',nargs=4,type=float,required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--hours',type=int,default=48)
    p.add_argument('--dt',type=int,default=10)
    p.add_argument('--severity',default='p90')
    a=p.parse_args()
    cfg=ScenarioConfig(bbox=tuple(a.bbox),hours=a.hours,dt_minutes=a.dt,severity=a.severity)
    build_and_write(cfg,a.outdir)
if __name__=='__main__': main_cli()
