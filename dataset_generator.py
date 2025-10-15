from pathlib import Path
from hypso import Hypso1, Hypso2
import numpy as np
import sys

dir_base = Path('data') / 'PhilipBerg' / 'SeaLandCloudH2'

dir_target = Path('dataset') / 'h2_normal'

if dir_target.is_dir():
    sys.exit("the target folder for this dataset is allredy created, write a new")
else:
    dir_target.mkdir()  


def load_hyperspectral_data(Load_NC_Path: Path):
    satobj_h2 = Hypso2(path=Load_NC_Path, verbose=True)
    satobj_h2.generate_l1b_cube()
    data_cube = satobj_h2.l1b_cube.to_numpy()
    data_cube = processing_drop_bands(data_cube)
    return data_cube.astype(np.float32, copy=False)

def processing_drop_bands(data_cube):
    drop = [0,1,2,3,4,5, 119, 118, 117]
    data_cube = np.delete(data_cube, drop, axis=-1)

    return data_cube

def processing_keep_every_n(data_cube, n):
    drop = [0,1,2,3,4,5, 119, 118, 117]
    data_cube = np.delete(data_cube, drop, axis=-1)

    return data_cube[..., ::n]

def processing_bin_nonoverlap_mean(data_cube, n):
    drop = [0,1,2,3,4,5, 119, 118, 117]
    data_cube = np.delete(data_cube, drop, axis=-1)

    H, W, B = data_cube.shape
    B2 = B - (B % n)
    c = data_cube[..., :B2].reshape(H, W, -1, n)
    return c.mean(-1)

def processing_bin_overlap_mean(data_cube, n, stride):
    drop = [0,1,2,3,4,5, 119, 118, 117]
    data_cube = np.delete(data_cube, drop, axis=-1)

    B = data_cube.shape[-1]
    starts = range(0, B - n + 1, stride)
    out = [data_cube[..., s:s+n].mean(-1) for s in starts]
    return np.stack(out, axis=-1)

def processing_bin_overlap_triangular(data_cube, n, stride):
    drop = [0,1,2,3,4,5, 119, 118, 117]
    data_cube = np.delete(data_cube, drop, axis=-1)

    i = np.arange(n); w = 1 - np.abs(i - (n-1)/2)/max((n-1)/2, 1e-9)
    w = (w / w.sum()).astype(data_cube.dtype)
    B = data_cube.shape[-1]
    starts = range(0, B - n + 1, stride)
    out = [(data_cube[..., s:s+n] * w).sum(-1) for s in starts]
    return np.stack(out, axis=-1)



def load_labels(dat_path: Path, H: int, W: int) -> np.ndarray:
    return np.fromfile(dat_path, dtype=np.uint8).reshape(H, W)

i = 0
for p in dir_base.iterdir():
    H, W, B = 0, 0, 0
    if p.is_dir():
        for f in p.glob('*.nc'):
            #print("NC:", f)
            path_nc = dir_target / f"data_{i}.npy"
            data_cube = load_hyperspectral_data(f)
            H, W, B = data_cube.shape
            np.save(path_nc, data_cube)  
            break
        for f in p.glob('*.dat'):
            #print("DAT:", f)
            path_dat = dir_target / f"label_{i}.npy"
            labels = load_labels(f, H, W)
            np.save(path_dat, labels)
            break

    i += 1



