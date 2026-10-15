import numpy as np
from pystackreg import StackReg
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import LogNorm
import matplotlib.animation as manimation
from matplotlib.widgets import Slider
import tifffile
import h5py
from scipy.optimize import curve_fit
from scipy.special import wofz  # For Voigt
from scipy import ndimage
#import csv
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import os
import sys
#sys.path.insert(0, '/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
sys.path.insert(0,'/nsls2/data/hxn/legacy/home/xf03id/src/hxntools')
try:
    from hxntools.CompositeBroker import db, get_path as get_path_new
except ImportError:
    print("dataBroker not installed")
try:
    from hxntools.scan_info import get_scan_positions
except ImportError:
    print("hxntools not found")

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import concurrent.futures as concurrent
import gc
import json
import tkinter as tk
from tkinter import filedialog
import warnings
import pandas as pd
import json
import multiprocessing as mp
import psutil

global db_key
db_key = 'new'

def set_db_key(value):
    global db_key
    db_key = value

def get_sid_list(str_list, interval):
    num_elem = np.size(str_list)
    for i in range(num_elem):
        str_elem = str_list[i].split('-')
        if i == 0:
            if np.size(str_elem) == 1:
                tmp = int(str_elem[0])
            else:
                tmp = np.arange(int(str_elem[0]),int(str_elem[1])+1,interval)
            sid_list = np.reshape(tmp,(-1,))
        else:
            if np.size(str_elem) == 1:
                tmp = int(str_elem[0])
            else:
                tmp = np.arange(int(str_elem[0]),int(str_elem[1])+1,interval)
            tmp = np.reshape(tmp,(-1,))
            sid_list = np.concatenate((sid_list,tmp))
    return sid_list


# def load_ims(file_list):
#     # stacking is along the first axis
#     num_ims = np.size(file_list)
#     for i in tqdm(range(num_ims),desc="Progress"):
#         file_name = file_list[i]
#         im = tifffile.imread(file_name)
#         im_row, im_col = np.shape(im)
#         if i == 0:
#             im_stack = np.reshape(im,(1,im_row,im_col))
#         else:
#             #im_stack_num = i 
#             im_stack_num, im_stack_row,im_stack_col = np.shape(im_stack)
#             row = np.maximum(im_row,im_stack_row)
#             col = np.maximum(im_col,im_stack_col)
#             if im_row < im_stack_row:
#                 r_s = np.round((im_stack_row-im_row)/2)
#             else:
#                 r_s = 0
#             if im_col < im_stack_col:
#                 c_s = np.round((im_stack_col-im_col)/2)
#             else:
#                 c_s = 0
#             im_stack_tmp = np.zeros((im_stack_num+1,row,col))
#             im_stack_tmp[0:im_stack_num,0:im_stack_row,0:im_stack_col] = im_stack
            
#             im_stack_tmp[im_stack_num,r_s:im_row+r_s,c_s:im_col+c_s] = im
#             im_stack = im_stack_tmp
#     return im_stack

def load_ims(file_list):
    """
    Load a stack of TIFF images and center-pad them to the same dimensions.

    Parameters
    ----------
    file_list : list of str
        List of file paths to the TIFF images.

    Returns
    -------
    im_stack : np.ndarray
        3D array containing the stacked images with shape (num_ims, max_row, max_col).
    """
    # First pass: determine max image size
    max_row, max_col = 0, 0
    for fname in file_list:
        im = tifffile.imread(fname)
        r, c = im.shape
        max_row = max(max_row, r)
        max_col = max(max_col, c)

    num_ims = len(file_list)
    im_stack = np.zeros((num_ims, max_row, max_col), dtype=np.float32)

    # Second pass: load and center-pad each image
    for i, fname in enumerate(tqdm(file_list, desc="Loading images")):
        im = tifffile.imread(fname)
        r, c = im.shape

        r_start = (max_row - r) // 2
        c_start = (max_col - c) // 2

        im_stack[i, r_start:r_start+r, c_start:c_start+c] = im

    return im_stack


def load_txts(file_list):
    """
    Load a stack of text-based images (e.g., .txt files) and center-pad them to the same dimensions.

    Parameters
    ----------
    file_list : list of str
        List of file paths to the text-based images.

    Returns
    -------
    im_stack : np.ndarray
        3D array containing the stacked images with shape (num_ims, max_row, max_col).
    """
    # stacking is along the first axis
    num_ims = np.size(file_list)
    for i in tqdm(range(num_ims),desc="Progress"):
        file_name = file_list[i]
        im = np.loadtxt(file_name)
        im_row, im_col = np.shape(im)
        if i == 0:
            im_stack = np.reshape(im,(1,im_row,im_col))
        else:
            #im_stack_num = i 
            im_stack_num, im_stack_row,im_stack_col = np.shape(im_stack)
            row = np.maximum(im_row,im_stack_row)
            col = np.maximum(im_col,im_stack_col)
            if im_row < im_stack_row:
                r_s = np.round((im_stack_row-im_row)/2)
            else:
                r_s = 0
            if im_col < im_stack_col:
                c_s = np.round((im_stack_col-im_col)/2)
            else:
                c_s = 0
            im_stack_tmp = np.zeros((im_stack_num+1,row,col))
            im_stack_tmp[0:im_stack_num,0:im_stack_row,0:im_stack_col] = im_stack
            
            im_stack_tmp[im_stack_num,r_s:im_row+r_s,c_s:im_col+c_s] = im
            im_stack = im_stack_tmp
    return im_stack


def create_file_list(data_path, prefix, postfix, sid_list):
    num = np.size(sid_list)
    file_list = [f"{data_path}{prefix}{sid}{postfix}" for sid in sid_list]
    return file_list

def get_fluorescence_image_dimensions(data_path, sid_list, elem_list):
    """
    Read the first available fluorescence image to determine target dimensions.
    
    Parameters
    ----------
    data_path : str
        Path to the XRF data directory (e.g., '//data//users//2026Q2//Sun_2026Q2//xrf//')
    sid_list : array-like
        List of scan IDs
    elem_list : list
        List of element names (e.g., ['Fe_K', 'Cr_K'])
    
    Returns
    -------
    tuple
        (target_height, target_width) dimensions of the fluorescence images
    """
    # Try each element and sid combination to find first available file
    for elem in elem_list:
        for sid in sid_list:
            test_file = f"{data_path}output_tiff_scan2D_{sid}//detsum_{elem}_norm.tiff"
            if os.path.exists(test_file):
                try:
                    im = tifffile.imread(test_file)
                    height, width = im.shape
                    print(f"Detected fluorescence image dimensions from {test_file}: {height}x{width}")
                    return (height, width)
                except Exception as e:
                    print(f"Error reading {test_file}: {e}")
                    continue
    raise FileNotFoundError(f"Could not find any valid fluorescence image in {data_path}")


def downsample_image_stack(im_stack, target_shape):
    """
    Downsample (or upsample) image stack to target dimensions using cubic interpolation.
    
    Parameters
    ----------
    im_stack : ndarray
        Image stack with shape (num_frames, height, width)
    target_shape : tuple
        Target dimensions (target_height, target_width)
    
    Returns
    -------
    ndarray
        Downsampled image stack with shape (num_frames, target_height, target_width)
    """
    num_frames, curr_height, curr_width = im_stack.shape
    target_height, target_width = target_shape
    
    # Calculate zoom factors for each dimension
    zoom_factors = (
        1.0,  # no scaling along frame axis
        target_height / curr_height,
        target_width / curr_width
    )
    
    # Apply zoom using cubic interpolation (order=3)
    im_stack_downsampled = ndimage.zoom(im_stack, zoom_factors, order=3)
    
    print(f"Downsampled from {curr_height}x{curr_width} to {target_height}x{target_width}")
    print(f"Image stack shape: {im_stack.shape} → {im_stack_downsampled.shape}")
    
    return im_stack_downsampled.astype(im_stack.dtype)

def align_im_stack(im_stack):
    # default stacking axis is zero
    #im_stack = np.moveaxis(im_stack,2,0)
    sr = StackReg(StackReg.TRANSLATION)
    #sr = StackReg(StackReg.RIGID_BODY)
    tmats = sr.register_stack(im_stack, reference='previous')
    out = sr.transform_stack(im_stack)
    a = tmats[:,0,2]
    b = tmats[:,1,2]
    trans_matrix = np.column_stack([-b,-a])
    return out, trans_matrix

def align_im_stack_v1(im_stack, method='TRANSLATION', ref='previous'):
    # default stacking axis is zero
    if method == 'TRANSLATION':
        sr = StackReg(StackReg.TRANSLATION)
    elif method == 'RIGID_BODY':
        sr = StackReg(StackReg.RIGID_BODY)
    elif method == 'AFFINE':
        sr = StackReg(StackReg.AFFINE)

    #sr = StackReg(StackReg.RIGID_BODY)
    tmats = sr.register_stack(im_stack, reference=ref)
    out = sr.transform_stack(im_stack)
    return out, tmats



def load_h5_data(file_list, roi, mask):
    # load a list of scans, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    data_type = 'float32'
    
    num_scans = np.size(file_list)
    #det = '/entry/instrument/detector/data'
    det = '/diff_data/merlin1/det_images'
    for i in tqdm(range(num_scans),desc="Progress"):
        file_name = file_list[i]
        f = h5py.File(file_name,'r')       
        if mask is None:
            data = f[det]
        else:
            data = f[det]*mask
        if roi is None:
            data = np.flip(data[:,:,:],axis = 1)
        else:
            data = np.flip(data[:,roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]],axis = 1)
        if i == 0:
            raw_size = np.shape(f[det])
            print("Total scan points: {}; raw image row: {}; raw image col: {}".format(raw_size[0],raw_size[1],raw_size[2]))
            data_size = np.shape(data)
            print("Total scan points: {}; data image row: {}; data image col: {}".format(data_size[0],data_size[1],data_size[2]))
            diff_data = np.zeros(np.append(num_scans,np.shape(data)),dtype=data_type)
        sz = diff_data.shape    
        diff_data[i] = np.resize(data,(sz[1],sz[2],sz[3])) # in case there are lost frames
    if  num_scans == 1: # assume it is a rocking curve scan
        diff_data = np.swapaxes(diff_data,0,1) # move angle to the first axis
        print("Assume it is a rocking curve scan; number of angles = {}".format(diff_data.shape[0]))
    return diff_data 

def load_h5_data_single(file_name,data_name,roi,mask,threshold,bins=1,flip=True):
    # Each thread will open a file, process it and return the index, data, and raw-size.
    # Using a context manager ensures the file is closed.
    with h5py.File(file_name, 'r', locking=False) as f:
        dset = f[data_name]
        if roi is None:
            rs, cs = slice(None), slice(None)
        else:
            rs, cs = slice(roi[0], roi[0] + roi[2]), slice(roi[1], roi[1] + roi[3])
        n, nr, nc = dset.shape
        shape = (n, len(range(*rs.indices(nr))), len(range(*cs.indices(nc))))
        # float32 avoids integer overflow when summing pixels of raw integer frames
        data = np.empty(shape, dtype=np.float32)
        dset.read_direct(data, source_sel=(slice(None), rs, cs))
    if mask is not None:
        data *= mask[rs, cs]
    if bins > 1:
        data = np.array([bin_image(frame, bins) for frame in data])
    if flip:
        data = np.flip(data, axis=1)
    if threshold is not None:
        data[data < threshold[0]] = 0
        data[data > threshold[1]] = 0
    return data

def _estimate_h5_task_bytes(file_name, data_name, roi=None):
    with h5py.File(file_name, 'r', locking=False) as f:
        n, rows, cols = f[data_name].shape
    if roi is not None:
        rows, cols = roi[2], roi[3]
    return n * rows * cols * 4  # float32 ROI buffer read by one worker

def load_h5_data_parallel(file_list, roi=None, mask=None, threshold=None, h5_type='diff_view', bins=1, max_workers=10, memory_safety=0.7):
    """
    Load a list of HDF5 (h5) files, apply optional ROI and mask,
    flip the data along axis=1, and stack the results along the first axis.
    
    file_list: list of filenames (paths) to load
    roi: [row_start, col_start, row_size, col_size] or None
    mask: an array of the same shape as each image data, or None.
    threshold: set pixel outside the bound [low, high] to zero, or None 
    h5_type: 'diff_view' for files exported from diff_view, otherwise raw detector files
    bins: binning factor applied to each frame after the ROI crop
    max_workers: upper limit on worker processes; further reduced to fit available memory
    memory_safety: fraction of available memory the workers may use; outside [0, 1] uses max_workers as is
    """
    data_type = 'float32'
    num_scans = len(file_list)
    if h5_type == 'diff_view':
        data_name = '/diff_data/merlin1/det_images' # for h5 files exported from diff_view
    else:
        data_name = '/entry/instrument/detector/data' # for h5 raw data


    if file_list is None or len(file_list) == 0:
        raise ValueError("file_list is empty or None.")

    max_workers = min(max_workers, os.cpu_count() or 1)
    max_workers = _cap_workers_by_memory(max_workers, _estimate_h5_task_bytes(file_list[0], data_name, roi), safety=memory_safety)
    ctx = mp.get_context("spawn")

    results = None  # allocated on the first result to avoid a second full-size copy
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(load_h5_data_single, file_list[i],data_name,roi,mask,threshold,bins): i for i in range(num_scans)}
        for future in tqdm(as_completed(futures), total=num_scans, desc="Progress"):
            idx = futures[future]
            try:
                result = future.result()
                if results is None:
                    results = np.zeros((num_scans,) + result.shape, dtype=result.dtype)
                results[idx] = result
                del result
            except Exception as e:
                print(f"Error processing file index {idx}: {e}")

    # If there is only one scan, assume a rocking curve scan and swap axes.
    if num_scans == 1:
        results = np.swapaxes(results, 0, 1)
        print("Assume it is a rocking curve scan; number of angles = {}".format(results.shape[0]))

    return results

def load_h5_data_db(sid,det,mon=None,roi=None,mask=None,threshold=None):
        sid = int(sid)
        file_names = get_path(sid, det)
        num_subscan = len(file_names)
        data_name = '/entry/instrument/detector/data'
        data_type = 'float32'
        # Load and optionally concatenate data from multiple files
        if num_subscan == 1:
            with h5py.File(file_names[0], 'r') as f:
                data = np.asarray(f[data_name], dtype=data_type)
        else:
            sorted_files = sort_files_by_creation_time(file_names)
            data = None
            for idx, fname in enumerate(sorted_files):
                with h5py.File(fname, 'r') as f:
                    d_temp = np.asarray(f[data_name], dtype=data_type)
                data = d_temp if idx == 0 else np.concatenate([data, d_temp], axis=0)
                
        # Apply threshold normalization if defined.
        if threshold is not None:
            data[data < threshold[0]] = 0
            data[data > threshold[1]] = 0
        
        # Apply monitor normalization if provided.
        if mon is not None:
            mon_array = np.asarray(list(db[sid].data(mon))).squeeze()
            avg = np.mean(mon_array[mon_array != 0])
            mon_array[mon_array == 0] = avg
            data = data / mon_array[:, np.newaxis, np.newaxis]
        
        # Apply mask if defined.
        if mask is not None:
            data = data * mask
        
        # Extract ROI (or flip entire image).
        if roi is None:
            proc_data = np.flip(data[:, :, :], axis=1)
        else:
            proc_data = np.flip(data[:, roi[0]:roi[0]+roi[2],
                                       roi[1]:roi[1]+roi[3]], axis=1)
        
        # Resize (if necessary) and return the processed scan.
        return proc_data

def load_h5_data_db_v1(sid, det, mon=None, roi=None, mask=None, threshold=None, bins=1, flip=True):
    """
    Load HDF5 data from the database with optional ROI, mask, thresholding, and binning.

    Parameters:
        sid (int): Scan ID.
        det (str): Detector name.
        mon (str, optional): Monitor channel for normalization.
        roi (tuple, optional): Region of interest (r0, c0, rsz, csz) for each frame.
        mask (ndarray, optional): Mask to apply to the data for the scan region.
        threshold (tuple, optional): Threshold values (min, max).
        bins (int, optional): Binning factor (default is 1).
        flip (bool, optional): Flip vertically (axis=1) before returning (default is True).

    Returns:
        ndarray: Processed data array.
    """


    sid = int(sid)
    file_names = get_path(sid, det)
    if not isinstance(file_names,list):
        file_names = [file_names]
    file_names = sort_files_by_creation_time(file_names)
    data_name = '/entry/instrument/detector/data'
   
    data_blocks = []
    
    for fname in file_names:
        with h5py.File(fname, 'r',locking=False) as f:
            dset = f[data_name]
            shape = dset.shape
            dtype = np.float32  # target dtype

            # Determine ROI slice
            if roi is None:
                roi_slices = (slice(None), slice(None), slice(None))
            else:
                r0, c0, rsz, csz = roi
                roi_slices = (slice(None), slice(r0, r0 + rsz), slice(c0, c0 + csz))

            # Allocate array for reading (smaller than full dataset)
            temp = np.empty(dset[roi_slices].shape, dtype=dtype)

            dset.read_direct(temp, source_sel=roi_slices)
            if bins > 1:
                temp = np.array([bin_image(frame, bins) for frame in temp])

            data_blocks.append(temp)

    data = np.concatenate(data_blocks, axis=0) if len(data_blocks) > 1 else data_blocks[0]

    # Threshold
    if threshold is not None:
        np.clip(data, threshold[0], threshold[1], out=data)
        data[(data == threshold[0]) | (data == threshold[1])] = 0

    # Monitor normalization
    if mon is not None:
        mon_array = np.asarray(list(db[sid].data(mon))).squeeze()
        avg = np.mean(mon_array[mon_array != 0])
        mon_array[mon_array == 0] = avg
        if mon_array.shape[0] > data.shape[0]:
            data = np.pad(data, ((0,mon_array.shape[0]-data.shape[0]), (0,0), (0,0)),mode='constant') 
        data /= mon_array[:, np.newaxis, np.newaxis]*mon_array[0]

    # Mask
    if mask is not None:
        data *= mask

    # Flip vertically (along axis=1)
    proc_data = np.flip(data, axis=1) if flip else data

    return proc_data

def _cap_workers_by_memory(max_workers, per_task_bytes, safety=0.7, overhead=2.0, reserve_bytes=0):
    # overhead covers temporaries and the pickled result copy held while a task finishes
    # reserve_bytes is memory already promised elsewhere, e.g. arrays allocated but not yet touched
    if safety > 1 or safety < 0:  # out-of-range disables the guard
        return max_workers
    avail = psutil.virtual_memory().available
    n_fit = int((avail * safety - reserve_bytes) // (per_task_bytes * overhead))
    capped = max(1, min(max_workers, n_fit))
    if capped < max_workers:
        print(f"Memory guard: max_workers reduced from {max_workers} to {capped} "
              f"({avail/1e9:.1f} GB available, ~{per_task_bytes/1e9:.2f} GB per task).")
    return capped

def _estimate_scan_bytes(sid, det, roi=None):
    file_names = get_path(int(sid), det)
    if not isinstance(file_names, list):
        file_names = [file_names]
    frames = 0
    for fname in file_names:
        with h5py.File(fname, 'r', locking=False) as f:
            shape = f['/entry/instrument/detector/data'].shape
        frames += shape[0]
    rows, cols = (shape[1], shape[2]) if roi is None else (roi[2], roi[3])
    return frames * rows * cols * 4

def load_h5_data_db_parallel(sid_list, det, mon=None, roi=None, mask=None, threshold=None, bins=1, max_workers = 10, memory_safety=0.7):
    """
    Load diffraction data from a list of scans (given by sid_list) through databroker
    with data being stacked along the first axis.
    
    Parameters:
      sid_list: list or array of scan IDs.
      det: key to locate the detector information (passed to get_path).
      mon: if provided, use monitor data to normalize each scan.
      roi: a list of [row_start, col_start, row_size, col_size]. If None, the full image is used.
      mask: an array to be multiplied with the detector data. Must be of appropriate shape.
      threshold: two-element list [lower, upper] to clip pixel values.
      bins (int): Binning factor for each frame.
      max_workers: upper limit on worker processes; further reduced to fit available memory.
      memory_safety: fraction of available memory the workers may use; outside [0, 1] uses max_workers as is.
      
    Returns:
      diff_data: stacked diffraction data with shape (num_scans, frames, row, col)
                 (or, if only one scan exists, a rocking curve with angle moved to the first axis).
    """
    num_scans = np.size(sid_list)
    
    diff_data = None  # allocated on the first result to avoid a second full-size copy
    # === Process scans 1...num_scans-1 concurrently ===
    num_cores = os.cpu_count() or 1
    max_workers = min(max_workers, num_cores)
    max_workers = _cap_workers_by_memory(max_workers, _estimate_scan_bytes(sid_list[0], det, roi), safety=memory_safety)
    ctx = mp.get_context("spawn") #ensure to create a new db session for each subprocess to be fork-safe
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(load_h5_data_db_v1, sid_list[i],det, mon,roi,mask,threshold, bins): i for i in range(num_scans)}
        for future in tqdm(as_completed(futures), total=num_scans, desc="Progress"):
            idx = futures[future]
            try:
                result = future.result()
                if diff_data is None:
                    diff_data = np.zeros((num_scans,) + result.shape, dtype=result.dtype)
                diff_data[idx] = result
                del result
            except Exception as e:
                print(f"Error processing scan index {idx}: {e}")
    
    # If only one scan was loaded, assume a rocking curve scan: swap axes.
    if num_scans == 1:
        diff_data = np.swapaxes(diff_data, 0, 1)
        print("Assume it is a rocking curve scan; number of angles = {}".format(diff_data.shape[0]))
    
    return diff_data

def sum_h5_data_db_single(sid, det, sum_over, mon=None, roi=None, mask=None, threshold=None, bins=1):
    if sum_over == 'frames':
        # unflipped so the summed image stays in the raw detector orientation used by roi
        data = load_h5_data_db_v1(sid, det, mon, roi, mask, threshold, bins, flip=False)
        return np.sum(data, axis=0)
    data = load_h5_data_db_v1(sid, det, mon, roi, mask, threshold, bins)
    return np.sum(data, axis=(1, 2))

def sum_h5_data_db_parallel(sid_list, det, sum_over='frames', mon=None, roi=None, mask=None, threshold=None, bins=1, max_workers=10, memory_safety=0.7):
    """
    Sum the detector data of each scan (located through databroker) without keeping the frames.
    Multiple subscans of one scan ID are concatenated before summing.

    sum_over:
      'frames'     : sum over the scan points and over all scans; returns one detector image (row, col).
      'det_pixels' : sum over the detector plane for each frame; returns (num_scans, num_positions).
    Other parameters are the same as in load_h5_data_db_parallel.
    """
    if sum_over not in ('frames', 'det_pixels'):
        raise ValueError("sum_over must be 'frames' or 'det_pixels'")
    num_scans = np.size(sid_list)
    max_workers = min(max_workers, os.cpu_count() or 1)
    max_workers = _cap_workers_by_memory(max_workers, _estimate_scan_bytes(sid_list[0], det, roi), safety=memory_safety)
    ctx = mp.get_context("spawn") # fresh db session per subprocess
    results = [None] * num_scans
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(sum_h5_data_db_single, sid_list[i], det, sum_over, mon, roi, mask, threshold, bins): i for i in range(num_scans)}
        for future in tqdm(as_completed(futures), total=num_scans, desc="Processing Scans"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error processing scan {sid_list[idx]}: {e}")
    results = np.stack(results)
    return np.sum(results, axis=0) if sum_over == 'frames' else results

def _resolve_source(file_list, source):
    if source not in ('auto', 'h5', 'db'):
        raise ValueError("source must be 'auto', 'h5', or 'db'")
    if source == 'db':
        return 'db'
    missing = [f for f in file_list if not os.path.isfile(f)]
    if source == 'h5':
        if missing:
            raise FileNotFoundError(f"{len(missing)} of {len(file_list)} diffraction files are missing: {missing}")
        return 'h5'
    if not missing:
        return 'h5'
    if len(missing) == len(file_list):
        return 'db'
    raise ValueError(f"{len(missing)} of {len(file_list)} diffraction files are missing: {missing}")

def load_diff_data(file_list, sid_list, det, mon=None, roi=None, mask=None, threshold=None, bins=1, max_workers=10, source='auto', memory_safety=0.7):
    """
    Load diffraction data from the h5 files in file_list or through databroker.

    Parameters:
      file_list: list of h5 file paths corresponding to the scans.
      sid_list: list or array of scan IDs.
      det: key to locate the detector information (passed to get_path).
      mon: if provided, use monitor data to normalize each scan.
      roi: a list of [row_start, col_start, row_size, col_size]. If None, the full image is used.
      mask: an array to be multiplied with the detector data, applied to the sample image. Must be of appropriate shape.
      threshold: two-element list [lower, upper] to clip pixel values.
      bins (int): Binning factor for each frame.
      max_workers: upper limit on worker processes; further reduced to fit available memory.
      source: 'auto' (h5 files if all exist, databroker if none do, ValueError if only some do),
              'h5' (require the files), or 'db' (always databroker).
      memory_safety: fraction of available memory the workers may use; outside [0, 1] uses max_workers as is.
      
    Returns:
      diff_data: stacked diffraction data with shape (num_scans, frames, row, col)
                 (or, if only one scan exists, a rocking curve with angle moved to the first axis).
    """
    if _resolve_source(file_list, source) == 'h5':
        if mon is not None:
            warnings.warn("mon is ignored when loading from h5 files.")
        print("Loading diffraction data from h5 files.")
        return load_h5_data_parallel(file_list, roi=roi, mask=mask, threshold=threshold, bins=bins,
                                    max_workers=max_workers, memory_safety=memory_safety)
    print("Loading diffraction data from databroker.")
    return load_h5_data_db_parallel(sid_list, det, mon=mon, roi=roi, mask=mask, threshold=threshold, bins=bins,
                                    max_workers=max_workers, memory_safety=memory_safety)

def load_and_sum_db(sid, det):
        sid = int(sid)
        data_type  = 'float32'
        data_name  = '/entry/instrument/detector/data'
        file_names = get_path(sid, det)  # get_path() must be defined in your environment
        if not isinstance(file_names, list):
            file_names = [file_names]
        num_subscan = len(file_names)
        # If there is only one subscan, load it directly.
        if num_subscan == 1:
            with h5py.File(file_names[0], 'r',locking=False) as f:
                # Load the dataset (here reading the whole array) 
                data = np.asarray(f[data_name])
        else:
            # For multiple subscans, sort them (using your own sort_files_by_creation_time)
            sorted_files = sort_files_by_creation_time(file_names)
            # Read and concatenate the subscans along the first axis.
            for ind, name in enumerate(sorted_files):
                with h5py.File(name, 'r') as f:
                    d_temp = np.asarray(f[data_name], dtype=data_type)
                if ind == 0:
                    data = d_temp
                else:
                    data = np.concatenate([data, d_temp], axis=0)
        
        # Sum over the first axis of the read data.
        sum_data = np.sum(data, axis=0)
        return sum_data

def sum_all_h5_data_db_parallel(sid_list, det, max_workers=10):
    """
    Load a list of scans using databroker information (from sid_list) and sum
    the detector data (located at '/entry/instrument/detector/data') from each scan.
    Multiple subscans for a single SID are concatenated before summing.
    
    Parameters:
      sid_list        : list or array of scan IDs.
      det             : key or parameter passed to get_path() to locate data.
      desired_workers : the maximum number of threads to use (capped by the system cores).
      
    Returns:
      sum_all_data    : The accumulated sum of the data (summed along the detector scan axis).
    """
    
    num_scans  = np.size(sid_list)

    # Get the number of available CPU cores and cap the workers appropriately.
    num_cores = os.cpu_count() or 1
    max_workers = min(max_workers, num_cores)
    ctx = mp.get_context("spawn") #ensure to create a new db session for each subprocess to be fork-safe
    
    print(f"Using {max_workers} workers (desired: {max_workers}, available cores: {num_cores}).")
    
    # List to hold per-scan summed data (preserving original order).
    scan_sum_list = [None] * num_scans

    # Process scans concurrently.
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        # Map each scan ID to its future.
        futures = {executor.submit(load_and_sum_db, sid_list[i], det): i for i in range(num_scans)}
        # Use as_completed (with tqdm for progress) to collect the results.
        for future in tqdm(as_completed(futures), total=num_scans, desc="Processing Scans"):
            idx = futures[future]
            try:
                scan_sum_list[idx] = future.result()
            except Exception as e:
                print(f"Error processing scan {sid_list[idx]}: {e}")
    scan_sum_list = np.asarray(scan_sum_list)
    # Now accumulate all per-scan summed data.
    if num_scans > 1:
        scan_sum_list = np.sum(scan_sum_list,0)

    return scan_sum_list


def sum_h5_data_single(file_name, data_name, sum_over, roi=None, mask=None, threshold=None, bins=1):
    if sum_over == 'frames':
        # unflipped so the summed image stays in the raw detector orientation used by roi
        data = load_h5_data_single(file_name, data_name, roi, mask, threshold, bins, flip=False)
        return np.sum(data, axis=0)
    data = load_h5_data_single(file_name, data_name, roi, mask, threshold, bins)
    return np.sum(data, axis=(1, 2))

def sum_h5_data_parallel(file_list, sum_over='frames', h5_type='diff_view', roi=None, mask=None, threshold=None, bins=1, max_workers=10, memory_safety=0.7):
    """
    Sum the detector data of each h5 file in file_list without keeping the frames.
    File-based counterpart of sum_h5_data_db_parallel.

    sum_over:
      'frames'     : sum over the scan points and over all files; returns one detector image (row, col).
      'det_pixels' : sum over the detector plane for each frame; returns (num_files, num_positions).
    h5_type: 'diff_view' for files exported from diff_view, otherwise raw detector files.
    Other parameters are the same as in load_h5_data_parallel.
    """
    if sum_over not in ('frames', 'det_pixels'):
        raise ValueError("sum_over must be 'frames' or 'det_pixels'")
    if file_list is None or len(file_list) == 0:
        raise ValueError("file_list is empty or None.")
    num_scans = len(file_list)
    if h5_type == 'diff_view':
        data_name = '/diff_data/merlin1/det_images'
    else:
        data_name = '/entry/instrument/detector/data'

    with h5py.File(file_list[0], 'r') as f:
        per_task_bytes = f[data_name].size * 4  # float32 working copy
    max_workers = min(max_workers, os.cpu_count() or 1)
    max_workers = _cap_workers_by_memory(max_workers, per_task_bytes, safety=memory_safety)

    results = [None] * num_scans
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(sum_h5_data_single, file_list[i], data_name, sum_over,
                                   roi, mask, threshold, bins): i for i in range(num_scans)}
        for future in tqdm(as_completed(futures), total=num_scans, desc="Processing Scans"):
            idx = futures[future]
            results[idx] = future.result()

    results = np.stack(results)
    return np.sum(results, axis=0) if sum_over == 'frames' else results

def sum_diff_data(file_list, sid_list, det, max_workers=10, sum_over='frames', mon=None, roi=None, mask=None, threshold=None, bins=1, source='auto', memory_safety=0.7):
    """
    Sum diffraction data from the h5 files in file_list or through databroker.
    source: 'auto' (h5 files if all exist, databroker if none do, ValueError if only some do),
            'h5' (require the files), or 'db' (always databroker).
    memory_safety: fraction of available memory the workers may use; outside [0, 1] uses max_workers as is.
    See sum_h5_data_db_parallel for sum_over.
    """
    if _resolve_source(file_list, source) == 'h5':
        if mon is not None:
            warnings.warn("mon is ignored when summing from h5 files.")
        print("Summing diffraction data from h5 files.")
        return sum_h5_data_parallel(file_list, sum_over=sum_over, roi=roi, mask=mask,
                                    threshold=threshold, bins=bins, max_workers=max_workers,
                                    memory_safety=memory_safety)
    print("Summing diffraction data from databroker.")
    return sum_h5_data_db_parallel(sid_list, det, sum_over=sum_over, mon=mon, roi=roi, mask=mask,
                                   threshold=threshold, bins=bins, max_workers=max_workers,
                                   memory_safety=memory_safety)


def process_3d(subset, shift_matrix, index):
    i = index
    ly = int(np.floor(shift_matrix[(i, 0)]))
    hy = int(np.ceil(shift_matrix[(i, 0)]))
    lx = int(np.floor(shift_matrix[(i, 1)]))
    hx = int(np.ceil(shift_matrix[(i, 1)]))
    lxly_subset = np.roll(subset, (ly, lx), axis = (0, 1))
    lxhy_subset = np.roll(subset, (hy, lx), axis = (0, 1))
    hxly_subset = np.roll(subset, (ly, hx), axis = (0, 1))
    hxhy_subset = np.roll(subset, (hy, hx), axis = (0, 1))
    ry = shift_matrix[(i, 0)] - ly
    rx = shift_matrix[(i, 1)] - lx
    return (1 - rx) * (1 - ry) * lxly_subset + rx * (1 - ry) * hxly_subset + (1 - rx) * ry * lxhy_subset + rx * ry * hxhy_subset


def process_4d(subset, shift_matrix, index):
    i = index
    l_bound = int(np.floor(shift_matrix[i]))
    h_bound = int(np.ceil(shift_matrix[i]))
    l_subset = np.roll(subset, l_bound, axis = 0)
    h_subset = np.roll(subset, h_bound, axis = 0)
    r = shift_matrix[i] - l_bound
    return (1 - r) * l_subset + r * h_subset


def process_5d(subset, shift_matrix, index):
    i = index
    ly = int(np.floor(shift_matrix[(i, 0)]))
    hy = int(np.ceil(shift_matrix[(i, 0)]))
    lx = int(np.floor(shift_matrix[(i, 1)]))
    hx = int(np.ceil(shift_matrix[(i, 1)]))
    lxly_subset = np.roll(subset, (ly, lx), axis = (0, 1))
    lxhy_subset = np.roll(subset, (hy, lx), axis = (0, 1))
    hxly_subset = np.roll(subset, (ly, hx), axis = (0, 1))
    hxhy_subset = np.roll(subset, (hy, hx), axis = (0, 1))
    ry = shift_matrix[(i, 0)] - ly
    rx = shift_matrix[(i, 1)] - lx
    return (1 - rx) * (1 - ry) * lxly_subset + rx * (1 - ry) * hxly_subset + (1 - rx) * ry * lxhy_subset + rx * ry * hxhy_subset


def process_3d_v1(subset, tmat, index):
    i = index
    t = tmat[i,0:2,0:2]
    u = tmat[i,0:2,2].reshape(2,1) 
    # inv_t = np.linalg.inv(t)
    sz = np.shape(subset)
    x, y = np.meshgrid(np.linspace(0,sz[1],sz[1],False),np.linspace(0,sz[0],sz[0],False))
    p = np.stack([x.ravel(),y.ravel()],0)
#     p = p + u
    trans_p = t@p + u
    trans_x = np.reshape(trans_p[0,:],x.shape)
    trans_y = np.reshape(trans_p[1,:],y.shape)

    ly = np.clip(np.floor(trans_y),0,sz[0]-1).astype(int)
    hy = np.clip(np.ceil(trans_y),0,sz[0]-1).astype(int)
    lx = np.clip(np.floor(trans_x),0,sz[1]-1).astype(int)
    hx = np.clip(np.ceil(trans_x),0,sz[1]-1).astype(int)
    lxly_subset = subset[ly,lx]
    lxhy_subset = subset[hy,lx]
    hxly_subset = subset[ly,hx]
    hxhy_subset = subset[hy,hx]
    ry = trans_y - ly
    rx = trans_x - lx
    return (1 - rx) * (1 - ry) * lxly_subset + rx * (1 - ry) * hxly_subset + (1 - rx) * ry * lxhy_subset + rx * ry * hxhy_subset

def process_4d_v1(subset, tmat, index):
    i = index
    t = tmat[i,0,0]
    u = tmat[i,0,1]
    # inv_t = np.linalg.inv(t)
    sz = np.shape(subset)
    x = np.linspace(0,sz[0],sz[0],False)
    trans_x = t*x + u

    lx = np.clip(np.floor(trans_x),0,sz[0]-1).astype(int)
    hx = np.clip(np.ceil(trans_x),0,sz[0]-1).astype(int)
    rx = (trans_x - lx).reshape((-1,) + (1,)*(subset.ndim-1))

    return (1 - rx) * subset[lx] + rx * subset[hx]


def process_5d_v1(subset, tmat, index):
    i = index
    t = tmat[i,0:2,0:2]
    u = tmat[i,0:2,2].reshape(2,1)
    # inv_t = np.linalg.inv(t)
    sz = np.shape(subset)
    
    x, y = np.meshgrid(np.linspace(0,sz[1],sz[1],False),np.linspace(0,sz[0],sz[0],False))
    p = np.stack([x.ravel(),y.ravel()],0)
#     p = p + u
    trans_p = t@p + u
    trans_x = np.reshape(trans_p[0,:],x.shape)
    trans_y = np.reshape(trans_p[1,:],y.shape)

    ly = np.clip(np.floor(trans_y),0,sz[0]-1).astype(int)
    hy = np.clip(np.ceil(trans_y),0,sz[0]-1).astype(int)
    lx = np.clip(np.floor(trans_x),0,sz[1]-1).astype(int)
    hx = np.clip(np.ceil(trans_x),0,sz[1]-1).astype(int)
    lxly_subset = subset[ly,lx,:,:]
    lxhy_subset = subset[hy,lx,:,:]
    hxly_subset = subset[ly,hx,:,:]
    hxhy_subset = subset[hy,hx,:,:]
    ry = trans_y - ly
    rx = trans_x - lx
    v00 = (1 - rx) * (1 - ry)
    v10 = rx * (1 - ry)
    v01 = (1 - rx) * ry 
    v11 = rx * ry 

    return v00[:,:,np.newaxis,np.newaxis] * lxly_subset + v10[:,:,np.newaxis,np.newaxis] * hxly_subset \
        + v01[:,:,np.newaxis,np.newaxis] * lxhy_subset + v11[:,:,np.newaxis,np.newaxis] * hxhy_subset

def _get_executor(executor_type, max_workers):
    if executor_type == 'thread':
        return ThreadPoolExecutor(max_workers=max_workers)
    if executor_type == 'process':
        return ProcessPoolExecutor(max_workers=max_workers)
    raise ValueError("executor_type must be 'thread' or 'process'.")

def interp_sub_pix(data, shift_matrix, max_workers = 10, executor_type = 'thread'):
    '''
    Parallelized function to apply sub-pixel shifts to 3D, 4D, or 5D data.

    Parameters:
        data (ndarray): Input data of shape (N, ...) where N is the first axis.
        shift_matrix (ndarray): Shift values corresponding to each slice.
        max_workers (int): Number of parallel workers (default: 10).
        executor_type (str): 'thread' (default) or 'process'.

    Returns:
        ndarray: Shifted data of the same shape as input.
    '''
    sz = np.shape(data)
    sz_len = np.size(sz)
    # results = np.zeros_like(data)
    process_func = None
    if sz_len == 3:
        process_func = process_3d
    elif sz_len == 4:
        process_func = process_4d
    elif sz_len == 5:
        process_func = process_5d
    else:
        raise ValueError('Dimension of the data must be 3D, 4D, or 5D.')
    
    available_cores = os.cpu_count() or 1
    max_workers = min(max_workers, available_cores)
    with _get_executor(executor_type, max_workers) as executor:
        # Submit a task for each slice
        futures = {executor.submit(process_func, subset, shift_matrix, index): index for index, subset in enumerate(data)}

        for future in tqdm(as_completed(futures), total=sz[0], desc="Parallel interp_sub_pix"):
            idx = futures[future]
            data[idx] = future.result()
    
    return data

def interp_sub_pix_v1(data, tmat, max_workers = 10, executor_type = 'thread'):
    '''
    Parallelized function to apply sub-pixel shifts to 3D, 4D, or 5D data.

    Parameters:
        data (ndarray): Input data of shape (N, ...) where N is the first axis.
        tmat (ndarray): Transform matrices corresponding to each slice.
        max_workers (int): Number of parallel workers (default: 10).
        executor_type (str): 'thread' (default) or 'process'.

    Returns:
        ndarray: Shifted data of the same shape as input.
    '''
    sz = np.shape(data)
    sz_len = np.size(sz)
    # results = np.zeros_like(data)
    process_func = None
    if sz_len == 3:
        process_func = process_3d_v1
    elif sz_len == 4:
        process_func = process_4d_v1
    elif sz_len == 5:
        process_func = process_5d_v1
    else:
        raise ValueError('Dimension of the data must be 3D, 4D, or 5D.')
    
    available_cores = os.cpu_count() or 1
    max_workers = min(max_workers, available_cores)
    with _get_executor(executor_type, max_workers) as executor:
        # Submit a task for each slice
        futures = {executor.submit(process_func, subset, tmat, index): index for index, subset in enumerate(data)}

        for future in tqdm(as_completed(futures), total=sz[0], desc="Parallel interp_sub_pix"):
            idx = futures[future]
            data[idx] = future.result()
    
    return data

def save_aligned_stack(im_stack, trans_matrix, output_path, name):
    """Apply the alignment in place, save the stack as <output_path><name>.tiff, and return (stack, summed image)."""
    im_stack = interp_sub_pix_v1(im_stack, trans_matrix)
    imp = np.sum(im_stack, axis=0)
    tifffile.imwrite(f"{output_path}{name}.tiff", im_stack.astype(np.float32), imagej=True)
    print(f"save {name} to {output_path}{name}.tiff")
    return im_stack, imp

def save_ims(file_list, trans_matrix, output_path, name, flip_axis=None, target_shape=None):
    """
    Load the images in file_list, optionally flip along flip_axis and downsample to target_shape,
    then align and save them with save_aligned_stack. Returns (aligned stack, summed image).
    """
    im_stack = load_ims(file_list)
    if flip_axis is not None:
        im_stack = np.flip(im_stack, axis=flip_axis)
    if target_shape is not None:
        im_stack = downsample_image_stack(im_stack, target_shape)
    return save_aligned_stack(im_stack, trans_matrix, output_path, name)

def trans_coor3D(X, Y, Z, M):
    '''Transform coordinates (X,Y,Z) by 3x3 matrix M (e.g., rotation/transform).'''
    sz = X.shape
    coords = np.vstack([
        X.ravel(),
        Y.ravel(),
        Z.ravel()])
    transformed = np.asarray(M @ coords)
    cX = np.reshape(transformed[0, :], sz)
    cY = np.reshape(transformed[1, :], sz)
    cZ = np.reshape(transformed[2, :], sz)
    return (cX, cY, cZ)


def create_grid(X, Y, Z, M):
    '''Create a 3D grid in the transformed coordinate system defined by M.'''
    (cX, cY, cZ) = trans_coor3D(X, Y, Z, M)
    x_coords = np.unique(np.round(cX, 2))
    y_coords = np.unique(np.round(cY, 2))
    z_coords = np.unique(np.round(cZ, 2))
    vec = M[:, 0] + M[:, 1] + M[:, 2]
    dz = abs(vec[2])
    dy = abs(vec[1])
    dx = abs(vec[0])
    x_range = np.arange(x_coords.min(), x_coords.max(), dx)
    y_range = np.arange(y_coords.min(), y_coords.max(), dy)
    z_range = np.arange(z_coords.min(), z_coords.max(), dz)
    (Vx, Vy, Vz) = np.meshgrid(x_range, y_range, z_range)
    pix_sz = np.array([
        dx,
        dy,
        dz]).reshape(3, 1)
    return (pix_sz, Vx, Vy, Vz)


def interp3_oblique(X, Y, Z, V, M, Vx, Vy, Vz):
    '''
    Trilinear interpolation of 3D volume V defined on coordinates X, Y, Z 
    at the points given by Vx, Vy, Vz (transformed by matrix M).
    '''
    sz = V.shape
    minX = X.min()
    maxX = X.max()
    minY = Y.min()
    maxY = Y.max()
    minZ = Z.min()
    maxZ = Z.max()
    dX = (maxX - minX) / (sz[1] - 1)
    dY = (maxY - minY) / (sz[0] - 1)
    dZ = (maxZ - minZ) / (sz[2] - 1)
    (qX, qY, qZ) = trans_coor3D(Vx, Vy, Vz, M)
    (nX, rX) = np.divmod((qX - minX) / dX, 1)
    (nY, rY) = np.divmod((qY - minY) / dY, 1)
    (nZ, rZ) = np.divmod((qZ - minZ) / dZ, 1)
    nX = nX.astype(np.int32).ravel()
    nY = nY.astype(np.int32).ravel()
    nZ = nZ.astype(np.int32).ravel()
    rX = rX.ravel()
    rY = rY.ravel()
    rZ = rZ.ravel()
    mask = np.ones(nX.shape, dtype = np.float32)
    mask[(nX < 0) | (nX >= sz[1] - 1) | (nY < 0) | (nY >= sz[0] - 1) | (nZ < 0) | (nZ >= sz[2] - 1)] = 0
    nX = np.clip(nX, 0, sz[1] - 2)
    nY = np.clip(nY, 0, sz[0] - 2)
    nZ = np.clip(nZ, 0, sz[2] - 2)
    V000 = V[(nY, nX, nZ)]
    V100 = V[(nY, nX + 1, nZ)]
    V010 = V[(nY + 1, nX, nZ)]
    V110 = V[(nY + 1, nX + 1, nZ)]
    V001 = V[(nY, nX, nZ + 1)]
    V101 = V[(nY, nX + 1, nZ + 1)]
    V011 = V[(nY + 1, nX, nZ + 1)]
    V111 = V[(nY + 1, nX + 1, nZ + 1)]
    inv_rx = 1 - rX
    inv_ry = 1 - rY
    inv_rz = 1 - rZ
    Vq_flat = V000 * inv_rx * inv_ry * inv_rz + V100 * rX * inv_ry * inv_rz + V010 * inv_rx * rY * inv_rz + V110 * rX * rY * inv_rz + V001 * inv_rx * inv_ry * rZ + V101 * rX * inv_ry * rZ + V011 * inv_rx * rY * rZ + V111 * rX * rY * rZ
    Vq_flat *= mask
    return Vq_flat.reshape(Vx.shape)


def interp3_oblique_reduced(X, Y, Z, V, M, Vx, Vy, Vz):
    vq = interp3_oblique(X, Y, Z, V, M, Vx, Vy, Vz)
    qxz = np.sum(vq, axis=0)
    qyz = np.sum(vq, axis=1)
    del vq
    return qxz, qyz


class RSM:
    def __init__(self, det_data, energy, delta, gamma, num_angle, th_step, pix, det_dist, offset, fluo_stack=None, fluo_names=None):
        # input det_data [angle,position, det_row,det_col]
        # output rsm [position,q_y,q_x,q_z]
        self.energy = energy
        self.delta = -delta * np.pi/180
        self.gamma = -gamma * np.pi/180
        self.num_angle = num_angle
        self.th_step = th_step * np.pi/180
        self.pix = pix
        self.det_dist = det_dist
        self.offset = offset
        self.det_data = det_data
        self.k = 1e4 / (12.398/energy)
        self.fluo_stack = fluo_stack
        self.fluo_names = fluo_names
    def calcRSM(self, coor, data_store='reduced', desired_workers=60, parallel_mode='thread', memory_safety=0.7, save_det_data=None):
    
        '''
        Calculate reciprocal space mapping (RSM) from the detector data.
        The bulk of the work is done by interpolating the transformed detector data
        onto a grid. Parallel processing is used to speed up the per-scan interpolation.
        parallel_mode: 'thread' (ThreadPoolExecutor) or 'process' (ProcessPoolExecutor).
        memory_safety: fraction of available memory the outputs and workers may use; outside [0, 1] disables the guard.
        save_det_data: .npy path; if given, the input det_data is saved there before it is reshaped and later deleted.
        '''
        if parallel_mode == 'thread':
            Executor = ThreadPoolExecutor
        elif parallel_mode == 'process':
            Executor = ProcessPoolExecutor
        else:
            raise ValueError("parallel_mode must be 'thread' or 'process'")
        
        # --- Standard computations to set up matrices and grids ---
        sz = np.shape(self.det_data)
        data_type = np.float32
        det_row = sz[-2]
        det_col = sz[-1]
        Mx = np.matrix([[1., 0., 0.],
                        [0., np.cos(self.delta), -np.sin(self.delta)],
                        [0., np.sin(self.delta), np.cos(self.delta)]])
        My = np.matrix([[np.cos(self.gamma), 0., np.sin(self.gamma)],
                        [0., 1., 0.],
                        [-np.sin(self.gamma), 0., np.cos(self.gamma)]])
        M_D2L = My @ Mx
        M_L2D = np.linalg.inv(M_D2L)
        kx_lab = M_D2L @ np.array([[1.], [0.], [0.]]) * self.k * (self.pix/self.det_dist)
        ky_lab = M_D2L @ np.array([[0.], [1.], [0.]]) * self.k * (self.pix/self.det_dist)
        k_0 = self.k * M_L2D @ np.array([[0.], [0.], [1.]])
        h = self.k * np.array([[0.], [0.], [1.]]) - k_0
        rock_z = np.cross((M_L2D @ np.array([[0.], [1.], [0.]])).T, h.T).T
        kz = -rock_z * self.th_step
        kz_lab = M_D2L @ kz
        M_O2L = np.concatenate([kx_lab, ky_lab, kz_lab], axis=1)
        M_L2O = np.linalg.inv(M_O2L)
        x_rng = np.linspace(1 - round(det_col/2), det_col - round(det_col/2), det_col, dtype=data_type)
        y_rng = np.linspace(1 - round(det_row/2), det_row - round(det_row/2), det_row,dtype=data_type)
        z_rng = np.linspace(1 - round(self.num_angle/2), self.num_angle - round(self.num_angle/2), self.num_angle,dtype=data_type)
        X, Y, Z = np.meshgrid(x_rng, y_rng, z_rng)
        X = X + self.offset[1]
        Y = Y + self.offset[0]
        ux_cryst = kz_lab / np.linalg.norm(kz_lab)
        uz_cryst = M_D2L @ h / np.linalg.norm(M_D2L @ h)
        uy_cryst = np.cross(uz_cryst.T, ux_cryst.T).T
        M_C2L = np.concatenate([ux_cryst, uy_cryst, uz_cryst], axis=1)
        M_C2O = M_L2O @ M_C2L
        M_O2C = np.linalg.inv(M_C2O)
        self.M_O2L = M_O2L
        self.M_L2O = M_L2O
        self.M_O2C = M_O2C
        self.M_C2O = M_C2O
        self.M_C2L = M_C2L
        self.M_L2C = np.linalg.inv(M_C2L)

        if coor == 'lab':
            M = M_O2L
            M_inv = M_L2O
        elif coor == 'cryst':
            M = M_O2C
            M_inv = M_C2O
        elif coor == 'cryst_beam_integrated':
            M = M_O2L
            M_inv = M_L2O
            orig_store = data_store
            data_store = 'full'
        else:
            print('coor must be lab or cryst')
            return

        self.coor = coor
        M = np.asarray(M, dtype=data_type)
        pix_sz, xq, yq, zq = create_grid(X, Y, Z, M)
        trans_sz = np.shape(xq)

        # --- Reshape and prepare the detector data ---
        # self.det_data = np.squeeze(np.swapaxes(np.expand_dims(self.det_data, axis=-1), 0, -1), 0)
        if save_det_data is not None:
            np.save(save_det_data, self.det_data)  # raw [angle, ..., det_row, det_col] layout, as attach_det_data expects
        self.det_data = np.moveaxis(self.det_data, 0, -1)

        sz = np.shape(self.det_data)
        self.det_data = np.reshape(self.det_data, [-1, sz[-3], sz[-2], sz[-1]])
        new_sz = np.shape(self.det_data)

        # --- Memory guard: preallocated outputs plus per-task interpolation temporaries ---
        n_vox = int(np.prod(trans_sz))
        if data_store == 'full':
            output_bytes = new_sz[0] * n_vox * 4
        else:
            output_bytes = new_sz[0] * (trans_sz[0] + trans_sz[1]) * trans_sz[2] * 4
        per_task_bytes = 25 * n_vox * 4  # about 25 float32/int32 arrays of grid size live in interp3_oblique
        if 0 <= memory_safety <= 1:
            budget = psutil.virtual_memory().available * memory_safety
            if output_bytes > budget:
                raise MemoryError(f"calcRSM output needs ~{output_bytes/1e9:.1f} GB but only {budget/1e9:.1f} GB "
                                  f"is allowed ({memory_safety:.0%} of available). Use data_store='reduced', "
                                  f"a larger bins, or a smaller roi.")

        if data_store == 'full':
            self.full_data = np.zeros((new_sz[0], trans_sz[0], trans_sz[1], trans_sz[2]),
                                       dtype=data_type)
        else:
            self.qxz_data = np.zeros((new_sz[0], trans_sz[1], trans_sz[2]), dtype=data_type)
            self.qyz_data = np.zeros((new_sz[0], trans_sz[0], trans_sz[2]), dtype=data_type)

        # --- Parallelize the per-scan interpolation ---
        # Determine number of workers; if desired_workers is unspecified, use available cores.
        available_cores = os.cpu_count() or 1
        max_workers = min(desired_workers, available_cores)
        max_workers = _cap_workers_by_memory(max_workers, per_task_bytes, safety=memory_safety,
                                             overhead=1.0 if parallel_mode == 'thread' else 2.0,
                                             reserve_bytes=output_bytes)

        # Threads share memory (numpy releases the GIL); processes pickle inputs/results per task.
        chunk_size = max_workers
        with Executor(max_workers=max_workers) as executor:
            with tqdm(total=new_sz[0], desc="Processing scans in parallel") as pbar:
                for batch_start in range(0, new_sz[0], chunk_size):
                    batch_end = min(batch_start + chunk_size, new_sz[0])
                    if data_store == 'full':
                        futures = {executor.submit(interp3_oblique, X, Y, Z, self.det_data[i, :, :, :],
                                     M_inv, xq, yq, zq): i for i in range(batch_start, batch_end)}
                    else:
                        futures = {executor.submit(interp3_oblique_reduced, X, Y, Z, self.det_data[i, :, :, :],
                                     M_inv, xq, yq, zq): i for i in range(batch_start, batch_end)}
                    for future in as_completed(futures):
                        idx = futures[future]
                        if data_store == 'full':
                            vq = future.result()
                            self.full_data[idx, :, :, :] = vq
                            del vq
                        else:
                            qxz, qyz = future.result()
                            self.qxz_data[idx, :, :] = qxz
                            self.qyz_data[idx, :, :] = qyz
                            del qxz, qyz
                        del futures[future]
                        pbar.update(1)
                    del futures
                    gc.collect()

        # --- Save grid and metadata ---
        self.xq = xq
        self.yq = yq
        self.zq = zq
        self.h = h
        self.X = X
        self.Y = Y
        self.Z = Z

        # --- Optionally reshape and clean up data after processing ---
        if data_store != 'full':
            del self.det_data
            gc.collect()
            # The reshaping below reintroduces any higher-order dimensions from the
            # original detector data array.
            self.qxz_data = np.reshape(self.qxz_data,
                                       np.concatenate((sz[0:-3], [trans_sz[1], trans_sz[2]]), axis=0))
            self.qyz_data = np.reshape(self.qyz_data,
                                       np.concatenate((sz[0:-3], [trans_sz[0], trans_sz[2]]), axis=0))
            print("raw det_data is deleted")
            print("qxz_data: [pos,qx,qz] with dimensions of {}".format(self.qxz_data.shape))
            print("qyz_data: [pos,qy,qz] with dimensions of {}".format(self.qyz_data.shape))
        else:
            self.full_data = np.reshape(self.full_data,
                                        np.concatenate((sz[0:-3], trans_sz[0:3]), axis=0))
            print("det_data: raw aligned det data, [pos,det_row,det_col,angles] with dimensions of {}"
                  .format(self.det_data.shape))
            print("full_data: 3D rsm, [pos,qy,qx,qz] with dimensions of {}"
                  .format(self.full_data.shape))
        self.data_store = data_store
        

    def calcSTRAIN(self, method):
        # calculate lattice strain and tilting angles based on method selected

        self.method = method
        if self.data_store == 'full':
            qz_pos = np.sum(np.sum(self.full_data, -2), -2)
            qx_pos = np.sum(np.sum(self.full_data, -1), -2)
            qy_pos = np.sum(np.sum(self.full_data, -1), -1)
        else:
            qz_pos = np.sum(self.qxz_data, -2)
            qx_pos = np.sum(self.qxz_data, -1)
            qy_pos = np.sum(self.qyz_data, -1)
        sz = np.shape(qz_pos)
        qz_pos = np.reshape(qz_pos, (-1,qz_pos.shape[-1]))
        qy_pos = np.reshape(qy_pos, (-1,qy_pos.shape[-1]))
        qx_pos = np.reshape(qx_pos, (-1,qx_pos.shape[-1]))

        if method['fit_type'] == 'com':   
            shift_qz = np.zeros(qz_pos.shape[0])
            shift_qy = np.zeros(qy_pos.shape[0])
            shift_qx = np.zeros(qx_pos.shape[0])
            for i in tqdm(range(qz_pos.shape[0])):
                shift_qz[i] = cen_of_mass(qz_pos[i,:])
                shift_qy[i] = cen_of_mass(qy_pos[i,:])
                shift_qx[i] = cen_of_mass(qx_pos[i,:])
            self.strain = -shift_qz * (self.zq[0, 0, 1] - self.zq[0, 0, 0]) / np.linalg.norm(self.h)
            self.tilt_x = shift_qx * (self.xq[0, 1, 0] - self.xq[0, 0, 0]) / np.linalg.norm(self.h)
            self.tilt_y = shift_qy * (self.yq[1, 0, 0] - self.yq[0, 0, 0]) / np.linalg.norm(self.h)
            self.tot = np.sum(qz_pos, -1)
            self.strain = np.reshape(self.strain, sz[:-1])
            self.tilt_x = np.reshape(self.tilt_x, sz[:-1])
            self.tilt_y = np.reshape(self.tilt_y, sz[:-1])
            self.tot = np.reshape(self.tot, sz[:-1])
        elif method['fit_type'] == 'peak':
            peak = method['shape']
            n_peaks = method['n_peaks']
            shift_qz = np.zeros((qz_pos.shape[0],n_peaks[2]))
            shift_qy = np.zeros((qy_pos.shape[0],n_peaks[1]))
            shift_qx = np.zeros((qx_pos.shape[0],n_peaks[0]))
            x = self.xq[0,:,0]
            y = self.yq[:,0,0]
            z = self.zq[0,0,:]
            for i in tqdm(range(qz_pos.shape[0])):
                
                x_param,_ = fit_peaks(x, qx_pos[i,:],peak,n_peaks[0])
                y_param,_ = fit_peaks(y, qy_pos[i,:],peak,n_peaks[1])
                z_param,_ = fit_peaks(z, qz_pos[i,:],peak,n_peaks[2])

                if peak == 'voigt':
                    shift_qx[i,:] = [x_param[j] for j in range(1,4*n_peaks[0],4)]
                    shift_qy[i,:] = [y_param[j] for j in range(1,4*n_peaks[1],4)]
                    shift_qz[i,:] = [z_param[j] for j in range(1,4*n_peaks[2],4)]
                else:
                    shift_qx[i,:] = [x_param[j] for j in range(1,3*n_peaks[0],3)]
                    shift_qy[i,:] = [y_param[j] for j in range(1,3*n_peaks[1],3)]
                    shift_qz[i,:] = [z_param[j] for j in range(1,3*n_peaks[2],3)]
            self.strain = -shift_qz/np.linalg.norm(self.h)
            self.tilt_x = shift_qx / np.linalg.norm(self.h)
            self.tilt_y = shift_qy / np.linalg.norm(self.h)
            self.tot = np.sum(qz_pos, -1)
            self.strain = np.reshape(self.strain, (sz[:-1]+(n_peaks[2],)))
            self.tilt_x = np.reshape(self.tilt_x, (sz[:-1]+(n_peaks[0],)))
            self.tilt_y = np.reshape(self.tilt_y, (sz[:-1]+(n_peaks[1],)))
            self.tot = np.reshape(self.tot, sz[:-1])

            # move peaks to the first axis
            self.strain = np.squeeze(np.moveaxis(self.strain, -1, 0))
            self.tilt_x = np.squeeze(np.moveaxis(self.tilt_x, -1, 0))
            self.tilt_y = np.squeeze(np.moveaxis(self.tilt_y, -1, 0))


        if method['mask'] is not None:
            ref_name = method['mask']
            mask = np.zeros_like(self.fluo_stack[0])
            threshold = method['mask threshold']
            if ref_name == 'tot':
                maxV = np.max(self.tot)
                mask[self.tot > threshold*maxV] = 1
            else:
                ref_im = self.fluo_stack[self.fluo_names.index(ref_name)]
                maxV = np.max(ref_im)
                mask[ref_im > threshold*maxV] = 1
            if self.strain.ndim == 3: 
                self.strain *= mask[np.newaxis,:,:]
            else:
                self.strain *= mask
            if self.tilt_x.ndim == 3: 
                self.tilt_x *= mask[np.newaxis,:,:]
            else:
                self.tilt_x *= mask    
            if self.tilt_y.ndim == 3: 
                self.tilt_y *= mask[np.newaxis,:,:]
            else:
                self.tilt_y *= mask

        # if np.size(sz) == 3:
        #     shift_qz = np.zeros((sz[0], sz[1]))
        #     shift_qx = np.zeros((sz[0], sz[1]))
        #     shift_qy = np.zeros((sz[0], sz[1]))
        #     for i in tqdm(range(sz[0])):
        #         for j in range(sz[1]):
        #             if method['fit_type'] == 'com':
        #                 shift_qz[i, j] = cen_of_mass(qz_pos[i, j, :])
        #                 shift_qy[i, j] = cen_of_mass(qy_pos[i, j, :])
        #                 shift_qx[i, j] = cen_of_mass(qx_pos[i, j, :])
        # elif np.size(sz) == 2:
        #     shift_qz = np.zeros((sz[0]))
        #     shift_qx = np.zeros((sz[0]))
        #     shift_qy = np.zeros((sz[0]))
        #     for i in tqdm(range(sz[0])):
        #         if method['fit_type'] == 'com':
        #             shift_qz[i] = cen_of_mass(qz_pos[i, :])
        #             shift_qy[i] = cen_of_mass(qy_pos[i, :])
        #             shift_qx[i] = cen_of_mass(qx_pos[i, :])
        # self.strain = -shift_qz * (self.zq[0, 0, 1] - self.zq[0, 0, 0]) / np.linalg.norm(self.h)
        # self.tilt_x = shift_qx * (self.xq[0, 1, 0] - self.xq[0, 0, 0]) / np.linalg.norm(self.h)
        # self.tilt_y = shift_qy * (self.yq[1, 0, 0] - self.yq[0, 0, 0]) / np.linalg.norm(self.h)
        # self.tot = np.sum(qz_pos, -1) 
    def disp(self):
        
        fig = plt.figure()
        fig.set_size_inches(8, 6)
        fig.set_dpi(160)
        
        sz = np.shape(self.strain)
        if np.size(sz) == 2:
            ax = plt.subplot(2,2,1)
            im = ax.imshow(self.tot)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('total intensity (cts)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)
        
            ax = plt.subplot(2,2,2)
            im = ax.imshow(self.strain*100)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('strain (%)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)
        
            ax = plt.subplot(2,2,3)
            im = ax.imshow(self.tilt_x*180/np.pi)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('tilt_x (degree)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)
         
            ax = plt.subplot(2,2,4)
            im = ax.imshow(self.tilt_y*180/np.pi)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('tilt_y (degree)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)
        elif np.size(sz) == 1:
            ax = plt.subplot(2,2,1)
            im = ax.plot(self.tot)
            ax.set_xlabel('x')
            ax.set_ylabel('total intensity (cts)')
             
            ax = plt.subplot(2,2,2)
            im = ax.plot(self.strain*100)
            ax.set_xlabel('x')
            ax.set_ylabel('strain (%)')
             
            ax = plt.subplot(2,2,3)
            im = ax.plot(self.tilt_x*180/np.pi)
            ax.set_xlabel('x')
            ax.set_ylabel('tilt_x (degree)')
         
            ax = plt.subplot(2,2,4)
            im = ax.imshow(self.tilt_y*180/np.pi)
            ax.set_xlabel('x')
            ax.set_ylabel('tilt_y (degree)')
        plt.tight_layout()
        plt.show(block=False)

        #plt.savefig('./result.png')
              
    def save(self,output_path,file_name=None):
        # file_name is a prefix (e.g. 'scan123' -> 'scan123_strain_map.tif'); None keeps the default names
        prefix = '' if file_name is None else f'{file_name}_'

        if not os.path.exists(output_path):
            os.mkdir(output_path)
            print("Directory '%s' created" %output_path)
        
        file_name = ''.join([output_path,prefix,'tot_intensity_map.tif'])
        tifffile.imwrite(file_name,np.asarray(self.tot,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,prefix,'strain_map.tif'])
        tifffile.imwrite(file_name,np.asarray(self.strain,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,prefix,'tilt_x_map.tif'])
        tifffile.imwrite(file_name,np.asarray(self.tilt_x,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,prefix,'tilt_y_map.tif'])
        tifffile.imwrite(file_name,np.asarray(self.tilt_y,dtype=np.float32),imagej=True)

        # file_name = ''.join([output_path,'pos_rsm_xz.obj'])
        # if self.data_store == 'full':
        #     pickle.dump(np.sum(self.full_data,-3),open(file_name,'wb'), protocol = 4)
        # else:
        #     pickle.dump(self.qxz_data,open(file_name,'wb'), protocol = 4)

        # file_name = ''.join([output_path,'pos_rsm_yz.obj'])
        # if self.data_store == 'full':
        #     pickle.dump(np.sum(self.full_data,-2),open(file_name,'wb'),protocol = 4)
        # else:
        #     pickle.dump(self.qyz_data,open(file_name,'wb'),protocol = 4)

        # file_name = ''.join([output_path,'xq.obj'])
        # pickle.dump(self.xq,open(file_name,'wb'),protocol = 4)

        # file_name = ''.join([output_path,'yq.obj'])
        # pickle.dump(self.yq,open(file_name,'wb'),protocol = 4)

        # file_name = ''.join([output_path,'zq.obj'])
        # pickle.dump(self.zq,open(file_name,'wb'),protocol = 4)

        # file_name = ''.join([output_path,'h.obj'])
        # pickle.dump(self.h,open(file_name,'wb'),protocol = 4)

        if plt.fignum_exists(1):
            file_name = ''.join([output_path,'results.png'])
            plt.savefig(file_name)
        file_name = f"{output_path}all.obj"
        pickle.dump(self,open(file_name,'wb'),protocol = 4)

    @classmethod
    def load(cls, file_name=None, det_data=None, mmap=False):
        # file_name may be an all.obj path or a directory containing it; None opens a file dialog
        # det_data: array, .npy path, or True to also load diff_data.npy next to all.obj (see attach_det_data)
        if file_name is None:
            file_name = select_file()
        elif os.path.isdir(file_name):
            file_name = os.path.join(file_name, 'all.obj')
        with open(file_name, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{file_name} does not contain an RSM object.")
        obj.source_dir = os.path.dirname(os.path.abspath(file_name))
        if det_data is not None:
            obj.attach_det_data(det_data, mmap=mmap)
        return obj

    def attach_det_data(self, det_data=True, mmap=False):
        """
        Restore the raw detector data [angle, ..., det_row, det_col] that calcRSM deletes.
        det_data: array, path to a .npy file, or True for diff_data.npy in the folder all.obj was loaded from.
        mmap: memory-map the .npy file (read-only) instead of reading it into memory.
        """
        if det_data is True:
            if not hasattr(self, 'source_dir'):
                raise ValueError("det_data=True needs an object created by RSM.load(); pass a path or array instead.")
            det_data = os.path.join(self.source_dir, 'diff_data.npy')
        if isinstance(det_data, (str, os.PathLike)):
            det_data = np.load(det_data, mmap_mode='r' if mmap else None)
        if det_data.shape[0] != self.num_angle or tuple(det_data.shape[-2:]) != tuple(self.X.shape[:2]):
            raise ValueError(f"det_data shape {det_data.shape} does not match num_angle={self.num_angle} "
                             f"and detector size {tuple(self.X.shape[:2])}.")
        self.det_data = det_data
        return self

    def run_interactive(self,scale='linear'):
        ims = [self.tot,self.strain,self.tilt_x,self.tilt_y]
        ims = np.stack(ims,axis=0)
        im_stack = np.concatenate((self.fluo_stack,ims),axis = 0)
        names = self.fluo_names + ['tot','strain','tilt_x','tilt_y']
        if self.data_store == 'full':
            disp_data = [np.swapaxes(np.sum(self.full_data,-3),-1,-2),np.swapaxes(np.sum(self.full_data,-2),-1,-2)]
            
        else:
            disp_data = [np.swapaxes(self.qxz_data,-1,-2),np.swapaxes(self.qyz_data,-1,-2)]
        transform = np.log if scale == 'log' else None  # applied per clicked slice to avoid full-size copies
        interactive_map(names,im_stack,['qx vs qz','qy vs qz'],disp_data,transform=transform)
        return
    
    def generate_movie(self, desc, path = None, scale='linear', fixed_row=None, fixed_col=None):
                  
        ims = [self.tot,self.strain,self.tilt_x,self.tilt_y]
        ims = np.stack(ims,axis=0)
        im_stack = np.concatenate((self.fluo_stack,ims),axis = 0)
        names = self.fluo_names + ['tot','strain','tilt_x','tilt_y']
        if self.data_store == 'full':
            disp_data = [np.swapaxes(np.sum(self.full_data,-3),-1,-2),np.swapaxes(np.sum(self.full_data,-2),-1,-2)]
            
        else:
            disp_data = [np.swapaxes(self.qxz_data,-1,-2),np.swapaxes(self.qyz_data,-1,-2)]
        transform = np.log if scale == 'log' else None  # applied per frame to avoid full-size copies
        
        sz = np.shape(self.tot) 
        if path is None:
            if fixed_row is None:
                fixed_row = sz[0]//2
            if fixed_col is None:
                fixed_col = sz[1]//2
            path = generate_row_col_list(0, sz[0], 0, sz[1], fixed_row=fixed_row, fixed_col=fixed_col)
        create_movie(desc, names,im_stack,['qx vs qz','qy vs qz'],disp_data,path,transform=transform)
        return 
    
def cen_of_mass(c):
    c = c.ravel()
    tot = np.sum(c)
    n = np.size(c)
    a = 0
    idx = n//2
    for i in range(n):
        a = a + c[i]
        if a > tot/2:
            idx = i - (a-tot/2)/c[i]
            break
    return idx-n//2
def rsm_cen_x_y(data):
    sz = np.shape(data)
    new_data = np.zeros(data.shape,dtype=data.dtype)
    im_xz = np.sum(data,0)
    im_yz = np.sum(data,1)
    
    if len(sz) == 3:
        for i in range(sz[2]):
            
            x_cen = cen_of_mass(im_xz[:,i])
            y_cen = cen_of_mass(im_yz[:,i])
            tot = np.sum(im_xz[:,i])
            row = int(np.floor(y_cen))
            col = int(np.floor(x_cen))
            new_data[row,col,i] = (1 - (y_cen - row))*(1 - (x_cen - col))*tot
            new_data[row,col+1,i] = (1 - (y_cen - row))*((x_cen - col))*tot
            new_data[row+1,col,i] = ((y_cen - row))*(1 - (x_cen - col))*tot
            new_data[row+1,col+1,i] = ((y_cen - row))*((x_cen - col))*tot
    else:
        print("must be 3D rsm data in lab (beam) coordinates")
    return new_data

# def get_path(scan_id, key_name='merlin1', db=db):
#     """Return file path with given scan id and keyname.
#     """
    
#     h = db[int(scan_id)]
#     e = list(db.get_events(h, fields=[key_name]))
#     #id_list = [v.data[key_name] for v in e]
#     id_list = [v['data'][key_name] for v in e]
#     rootpath = db.reg.resource_given_datum_id(id_list[0])['root']
#     flist = [db.reg.resource_given_datum_id(idv)['resource_path'] for idv in id_list]
#     flist = set(flist)
#     fpath = [os.path.join(rootpath, file_path) for file_path in flist]
#     return fpath

def interactive_map(
    names,
    im_stack,
    label,
    data_4D,
    cmap='jet',
    clim=None,
    marker_color='black',
    transform=None
):
    n = min(len(names), im_stack.shape[0])
    m = min(len(label),len(data_4D))

    def slice_at(j, row, col):
        img = data_4D[j][row, col]
        return img if transform is None else transform(img)

    total = n + m
    ncols, nrows = 3, int(np.ceil(total / 3))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(3 * ncols, 3 * nrows),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2}
    )
    axs = axs.ravel()

    # initial plotting
    
    for ax, name, img in zip(axs, names[:n], im_stack[:n]):
        ax.imshow(img, cmap=cmap, aspect='auto')
        ax.set_title(name, fontsize=9)
        ax.axis('off')
    for i in range(m):
        diff_ax = axs[n+i]
        im = diff_ax.imshow(slice_at(i, 0, 0), cmap=cmap, clim=clim, aspect='auto')
        diff_ax.set_title(label, fontsize=9)
        diff_ax.axis('off')
        plt.show(block=False)

    for ax in axs[total:]:
        ax.axis('off')

    cbar = fig.colorbar(im, ax=diff_ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # tighten BEFORE connecting
    fig.tight_layout()

    def onclick(event):
        if event.inaxes not in axs:
            return
        col, row = int(round(event.xdata)), int(round(event.ydata))
        if not (0 <= col < im_stack.shape[2] and 0 <= row < im_stack.shape[1]):
            return

        # redraw each panel
        for i, ax in enumerate(axs[:total]):
            ax.clear()
            if i < n:
                ax.imshow(im_stack[i], cmap=cmap, aspect='auto')
                ax.plot(col, row, 'o', color=marker_color, ms=5)
                ax.set_title(names[i], fontsize=9)
            else:
                j = i - n
                ax.imshow(slice_at(j, row, col), cmap=cmap, clim=clim, aspect='auto')
                ax.set_title(label[j], fontsize=9)
            ax.axis('off')

        # immediate redraw
        fig.canvas.draw()

    def on_close(event):
        # the callbacks keep the closure alive, so release the data explicitly
        nonlocal data_4D, im_stack
        data_4D = None
        im_stack = None
        for ax in axs:
            ax.clear()

    # connect *after* layout
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('close_event', on_close)

    fig.show()

def create_movie(desc, names,im_stack,label,data_4D,path,cmap='jet',marker_color='white',clim=None,transform=None):
    # desc: a dictionary. Example,
    # desc ={
    #    'title':'Movie',
    #    'artist': 'hyan',
    #    'comment': 'Blanket film',
    #    'save_file': 'movie_blanket_film.mp4',
    #    'fps': 15,
    #    'dpi': 100
    # }
    # names: names of the individual im in im_stack
    # label: name of the 4D dataset
    # data_4D: the 4D dataset, for example, [row, col, qx, qz]
    # path: sampled positions for the movie, a list of [row, col]
    # cmap: color scheme of the plot
    # color: color of the marker
    # transform: optional function applied to each displayed slice, e.g. np.log
    
    n = min(len(names), im_stack.shape[0])
    m = min(len(label),len(data_4D))

    def slice_at(j, row, col):
        img = data_4D[j][row, col]
        return img if transform is None else transform(img)

    total = n + m
    ncols, nrows = 3, int(np.ceil(total / 3))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(3 * ncols, 3 * nrows),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2}
    )
    axs = axs.ravel()

    # initial plotting
    
    for ax, name, img in zip(axs, names[:n], im_stack[:n]):
        ax.imshow(img, cmap=cmap, aspect='auto')
        ax.set_title(name, fontsize=9)
        ax.axis('off')
    for i in range(m):
        diff_ax = axs[n+i]
        im = diff_ax.imshow(slice_at(i, 0, 0), cmap=cmap, clim=clim, aspect='auto')
        diff_ax.set_title(label, fontsize=9)
        diff_ax.axis('off')
        plt.show(block=False)

    for ax in axs[total:]:
        ax.axis('off')

    cbar = fig.colorbar(im, ax=diff_ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    
    fig.tight_layout()
    
    def update_fig(row,col,cmap=cmap,color=marker_color):
        
        plt.cla()
        for i, ax in enumerate(axs[:total]):
            ax.clear()
            if i < n:
                ax.imshow(im_stack[i], cmap=cmap, aspect='auto')
                ax.plot(col, row, 'o', color=marker_color, ms=5)
                ax.set_title(names[i], fontsize=9)
            else:
                j = i - n
                ax.imshow(slice_at(j, row, col), cmap=cmap, clim=clim, aspect='auto')
                ax.set_title(label[j], fontsize=9)
            ax.axis('off')
        fig.tight_layout()
        #fig.canvas.draw_idle()
        return 
     
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=desc['title'], artist=desc['artist'],
                comment=desc['comment'])
    writer = FFMpegWriter(fps=desc['fps'], metadata=metadata)
     
    with writer.saving(fig, desc['save_file'], dpi=desc['dpi']):
        writer.grab_frame()
         
        for j in tqdm(range(len(path)),desc='Progress'):       
            update_fig(path[j][0],path[j][1],cmap=cmap,color=marker_color)
            writer.grab_frame()
    writer.finish()

def get_file_creation_time(file_path):
    try:
        return os.path.getctime(file_path)
    except OSError:
        # If there is an error (e.g., file not found), return 0
        return 0

def sort_files_by_creation_time(file_list):
    # Sort the file list based on their creation time
    return sorted(file_list, key=lambda file: get_file_creation_time(file))

def block_mask(data, pos1, pos2, axes_swap=True,region = 0):
    # define a line specified by pos1 and pos2, set either the bottom part to be zero, region = 0, or the upper part, region = 1
    sz = np.shape(data)
    data_new = np.reshape(data,[-1,sz[-2],sz[-1]])
    sz_new = np.shape(data_new)
    mask = np.ones((sz_new[-2],sz_new[-1]))
    if axes_swap:
        data_new = np.swapaxes(data_new,-2,-1)
    if pos2[0]-pos1[0] == 0:
        a_inv = 0
        b_inv = pos1[0]
        x, y = np.meshgrid(np.linspace(0,sz_new[-1],sz_new[-1]),np.linspace(0,sz_new[-2],sz_new[-2]))
        y_x = a_inv*y+b_inv
        if region == 0:
            mask[y_x < x] = 0
        else:
            mask[y_x > x] = 0
        for i in range(sz_new[0]):
            data_new[i,:,:] = data_new[i,:,:]*mask
    
    
    else:
        a = (pos2[1]-pos1[1])/(pos2[0]-pos1[0])
        b = pos1[1]-a*pos1[0]
    
        x, y = np.meshgrid(np.linspace(0,sz_new[-1],sz_new[-1]),np.linspace(0,sz_new[-2],sz_new[-2]))
        x_y = a*x+b
        if region == 0:
            mask[x_y < y] = 0
        else:
            mask[x_y > y] = 0
        for i in range(sz_new[0]):
            data_new[i,:,:] = data_new[i,:,:]*mask
    
    if axes_swap:
        data_new = np.reshape(np.swapaxes(data_new,-2,-1),sz)
        data_new = np.swapaxes(data_new,-2,-1)
    else:
        data_new = np.reshape(data_new,sz)
    return data_new

# Define peak shapes
def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

def lorentzian(x, A, x0, gamma):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)

def voigt(x, A, x0, sigma, gamma):
    z = ((x - x0) + 1j*gamma) / (sigma * np.sqrt(2))
    return A * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

# General fitting function for N peaks
def build_model(x, peak_type, n_peaks):
    def model(x, *params):
        y = np.zeros_like(x)
        for i in range(n_peaks):
            offset = i * shape_param_count[peak_type]
            p = params[offset : offset + shape_param_count[peak_type]]
            y += peak_funcs[peak_type](x, *p)
        return y
    return model

# Parameter counts per peak shape
shape_param_count = {
    'gaussian': 3,
    'lorentzian': 3,
    'voigt': 4
}

# Mapping of shape to function
peak_funcs = {
    'gaussian': gaussian,
    'lorentzian': lorentzian,
    'voigt': voigt
}

def fit_peaks(x, y, peak_type='gaussian', n_peaks=1, p0=None, bounds=(-np.inf, np.inf)):
    """
    Fits single or multiple peaks using specified shape.

    Parameters:
        x (array): x-data
        y (array): y-data
        peak_type (str): 'gaussian', 'lorentzian', or 'voigt'
        n_peaks (int): number of peaks to fit
        p0 (list or None): initial parameter guess [A1, x01, sigma1, A2, x02, sigma2, ...]
        bounds (2-tuple): (lower_bounds, upper_bounds)

    Returns:
        popt: fitted parameters
        fit_y: fitted curve
    """
    if peak_type not in peak_funcs:
        raise ValueError(f"Unsupported peak_type '{peak_type}'")

    model = build_model(x, peak_type, n_peaks)

    if p0 is None:
        # Default guess: equally spaced peaks
        step = (x[-1] - x[0]) / (n_peaks + 1)
        p0 = []
        for i in range(n_peaks):
            A_guess = max(y)
            x0_guess = x[0] + (i+1) * step
            sigma_guess = (x[-1] - x[0]) / 10
            if peak_type == 'voigt':
                gamma_guess = sigma_guess / 2
                p0.extend([A_guess, x0_guess, sigma_guess, gamma_guess])
            else:
                p0.extend([A_guess, x0_guess, sigma_guess])
    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds,maxfev=5000)
        fit_y = model(x, *popt)
        return popt, fit_y
    except Exception as e:
        print(f"Fit failed: {e}")
        return np.full_like(p0,np.nan), np.full_like(y, np.nan)
    
def select_roi(image, output_path='', percentile=(1, 99.5), scale='log'):
    """
    Display an image and allow user to draw a rectangle to select ROI.
    
    Parameters:
        image (ndarray): 2D image array
        output_path (str): directory prefix for the saved ROI images
        percentile (tuple): (low, high) percentiles used as color limits so outlier pixels do not saturate the display;
                            None uses the full min/max range
        scale (str): 'linear' or 'log'; 'log' uses a log color scale (colorbar stays in image units),
                     percentiles are taken over positive pixels and pixels <= 0 are drawn black

    Returns:
        roi (list): [row_start, col_start, row_width, col_width]
                    or None if no selection was made.
    """
    if scale not in ('linear', 'log'):
        raise ValueError("scale must be 'linear' or 'log'")
    roi_holder = {}
    cmap = plt.get_cmap('jet').copy()
    finite = image[np.isfinite(image)]
    if scale == 'log':
        cmap.set_bad('black')  # LogNorm masks values <= 0
        finite = finite[finite > 0]
    if percentile is None:
        vmin, vmax = (finite.min(), finite.max()) if scale == 'log' else (None, None)
    else:
        vmin, vmax = np.percentile(finite, percentile)
    if scale == 'log':
        show_kw = dict(cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        show_kw = dict(cmap=cmap, vmin=vmin, vmax=vmax)

    def on_select(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        row_start = min(y1, y2)
        col_start = min(x1, x2)
        row_width = abs(y2 - y1)
        col_width = abs(x2 - x1)

        roi_holder['roi'] = [row_start, col_start, row_width, col_width]
        print("Selected ROI:", roi_holder['roi'])
        
    fig, ax = plt.subplots()
    im = ax.imshow(image, **show_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title("Draw a rectangle to select ROI, then close the window")

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5,
        minspany=5,
        spancoords='pixels',
        interactive=True
    )
    selector.set_active(True)
    plt.show(block=True)  # Waits for user interaction and window close

    roi = roi_holder.get('roi', None)
    if roi is not None:
        row_start, col_start, row_width, col_width = roi

        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)
            print("Directory '%s' created" % output_path)

        fig_full, ax_full = plt.subplots()
        ax_full.imshow(image, **show_kw)
        ax_full.add_patch(plt.Rectangle((col_start, row_start), col_width, row_width,
                                        edgecolor='white', facecolor='none', linewidth=2))
        ax_full.set_title(f"ROI: {roi}")
        fig_full.savefig(f"{output_path}roi_{roi}.png", dpi=150, bbox_inches='tight')
        plt.close(fig_full)

        fig_crop, ax_crop = plt.subplots()
        ax_crop.imshow(image[row_start:row_start + row_width,
                             col_start:col_start + col_width], **show_kw)
        ax_crop.set_title(f"ROI crop: {roi}")
        fig_crop.savefig(f"{output_path}roi_crop_{roi}.png", dpi=150, bbox_inches='tight')
        plt.close(fig_crop)

    return roi

def slider_view(im_stack):
    # Create figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)  # Leave space for slider

    # Display the first image
    im = ax.imshow(im_stack[0], cmap='jet')
    ax.set_title("Image 0")
    ax.axis('off')

    # Add slider below the image
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, 'Frame', 0, len(im_stack)-1, valinit=0, valstep=1)

    # Update function for slider
    def update(val):
        idx = int(slider.val)
        im.set_data(im_stack[idx])
        ax.set_title(f"Image {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show(block=True)

def load_json_file():
    # Create a hidden Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to select JSON file
    file_path = filedialog.askopenfilename(
        title="Select a JSON file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )

    if not file_path:
        print("No file selected.")
        return None

    # Load the JSON data
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print("Loaded data:")
        print(json.dumps(data, indent=4))
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None
    
def get_baseline_fields(run, field_list):
    """
    Extract baseline fields from a Bluesky run using DataBroker v1, suppressing pandas fragmentation warnings.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

            df = run.table(stream_name='baseline')

        if df.empty:
            print("Baseline stream exists but has no data.")
            return {}

        first_row = df.iloc[0]
        result = {}
        for field in field_list:
            if field in first_row:
                result[field] = first_row[field]
            else:
                print(f"Field '{field}' not found in baseline.")
        return result

    except Exception as e:
        print(f"Error accessing baseline: {e}")
        return {}

def read_params_db(sid_list,microscope='mll',det='merlin1'):
    
    det_param = {'merlin1':55, "merlin2":55, "eiger2_images":75}
    num_scans = len(sid_list)
    if microscope == 'mll':
        th_name = 'dsth'
    elif microscope =='zp':
        th_name = 'zpsth'
    else:
        print('microscope name needs to be either mll or zp')
    
    keys = ['energy','diff_gamma','diff_delta','diff_r']
    values = get_baseline_fields(db[sid_list[0]],keys)

    th_step = get_baseline_fields(db[sid_list[1]],[th_name])[th_name] - get_baseline_fields(db[sid_list[0]],[th_name])[th_name]

    if np.abs(th_step) < 1e-3:
        print(f"{th_name} doesn't seem to change over scans.Please switch to another microscope for the correct theta stage")

    det_pix = det_param[det]

    params={"number of angles":num_scans,
            "angle step":th_step,
            "detector name": det,
            "energy": values['energy'],
            "gamma": values['diff_gamma'],
            "delta": values['diff_delta'],
            "detector distance": values['diff_r']*1000,
            "pixel size": det_pix
            }
    print(json.dumps(params, indent=4))
    return params

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj
    
def get_path_old(scan_id, key_name='merlin1', db=db):
    """Return file path with given scan id and keyname.
    """
    
    h = db[int(scan_id)]
    e = list(db.get_events(h, fields=[key_name]))
    #id_list = [v.data[key_name] for v in e]
    id_list = [v['data'][key_name] for v in e]
    rootpath = db.reg.resource_given_datum_id(id_list[0])['root']
    flist = [db.reg.resource_given_datum_id(idv)['resource_path'] for idv in id_list]
    flist = set(flist)
    fpath = [os.path.join(rootpath, file_path) for file_path in flist]
    return fpath

def get_path(scan_id, key_name='merlin1', db=db):
    global db_key
    if db_key == 'old':
        try:
            return get_path_old(scan_id,key_name,db)
        except Exception as e:
            return get_path_new(scan_id,key_name,db)
    else:
        try:
            return get_path_new(scan_id,key_name,db)
        except Exception as e:
            return get_path_old(scan_id,key_name,db)
        
def generate_row_col_list(row_start, row_end, col_start, col_end, fixed_row=None, fixed_col=None):
    pairs = []

    # Case 1: vary row with fixed column
    if fixed_col is not None:
        pairs.extend([[r, fixed_col] for r in range(row_start, row_end )])

    # Case 2: vary column with fixed row
    if fixed_row is not None:
        pairs.extend([[fixed_row, c] for c in range(col_start, col_end )])

    return pairs  
def bin_image(img, bins = 2):
    h, w = img.shape
    h_new = h // bins
    w_new = w // bins
    img = img[:h_new*bins, :w_new*bins]  # crop to fit
    img = img.reshape(h_new, bins, w_new, bins)
    return img.sum(axis=(1, 3))
    
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a .obj file",
        filetypes=[("obj Files", "*.obj"), ("All Files", "*.*")]
    )
    return file_path