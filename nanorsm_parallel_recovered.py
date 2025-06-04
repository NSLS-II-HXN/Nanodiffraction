# Source Generated with Decompyle++
# File: nanorsm_parallel.cpython-312.pyc (Python 3.12)

import numpy as np
from pystackreg import StackReg
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import tifffile
import h5py
import csv
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import os
from databroker import db
import sys
sys.path.insert(0, '/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
from hxntools.CompositeBroker import db
from hxntools.scan_info import get_scan_positions
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import concurrent.futures as concurrent

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


def load_ims(file_list):
    # stacking is along the first axis
    num_ims = np.size(file_list)
    for i in tqdm(range(num_ims),desc="Progress"):
        file_name = file_list[i]
        im = tifffile.imread(file_name)
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


def load_txts(file_list):
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
    file_list = []
    for sid in sid_list:
        # tmp = ''.join([data_path, prefix,'{}'.format(sid),postfix])
        tmp = f"{data_path}{prefix}{sid}{postfix}"
        file_list.append(tmp)
    return file_list


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


def load_h5_data(file_list, roi, mask):
    # load a list of scans, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    data_type = 'float32'
    
    num_scans = np.size(file_list)
    det = '/entry/instrument/detector/data'
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

def load_h5_data_single(file_name,data_name,roi,mask,threshold):
    # Each thread will open a file, process it and return the index, data, and raw-size.
    # Using a context manager ensures the file is closed.
    with h5py.File(file_name[0], 'r') as f:
        dset = np.asarray(f[data_name])
        # Read the full dataset (or use slices if desired)
        # Multiply by the mask if one is provided.
        if mask is None:
            data = dset[...]
        else:
            data = dset[...] * mask
        # Apply ROI if provided; else flip the full dataset.
        if roi is None:
            data = np.flip(data[:,:,:], axis=1)
        else:
            data = np.flip(data[:, roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]], axis=1)
        raw_size = np.shape(dset)
    if threshold is not None:
        data[data < threshold[0]] = 0
        data[data > threshold[1]] = 0
    return data

def load_h5_data_parallel(file_list, roi=None, mask=None, threshold=None):
    """
    Load a list of HDF5 (h5) files, apply optional ROI and mask,
    flip the data along axis=1, and stack the results along the first axis.
    
    file_list: list of filenames (paths) to load
    roi: [row_start, col_start, row_size, col_size] or None
    mask: an array of the same shape as each image data, or None.
    threshold: set pixel outside the bound [low, high] to zero, or None 
    """
    data_type = 'float32'
    num_scans = len(file_list)
    data_name = '/entry/instrument/detector/data'



    # Create a list to hold results (preserving file order)
    results = [None] * num_scans

    # Launch threads to process each file concurrently.
    with ProcessPoolExecutor() as executor:
        # Create a dictionary mapping futures to index
        futures = {executor.submit(load_h5_data_single, file_list[i],data_name,roi,mask,threshold): i for i in range(num_scans)}
        for future in tqdm(as_completed(futures), total=num_scans, desc="Progress"):
            idx = futures[future]
            results[idx] = future.result()
    results = np.stack(results)

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

def load_h5_data_db_parallel(sid_list, det, mon=None, roi=None, mask=None, threshold=None, max_workers = 10):
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
      
    Returns:
      diff_data: stacked diffraction data with shape (num_scans, frames, row, col)
                 (or, if only one scan exists, a rocking curve with angle moved to the first axis).
    """
    num_scans = np.size(sid_list)
    
    diff_data = [None]*num_scans
    # === Process scans 1...num_scans-1 concurrently ===
    num_cores = os.cpu_count() or 1
    max_workers = min(max_workers, num_cores)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_h5_data_db, sid_list[i],det, mon,roi,mask,threshold): i for i in range(num_scans)}
        for future in tqdm(as_completed(futures), total=num_scans, desc="Progress"):
            idx = futures[future]
            try:
                diff_data[idx] = future.result()
            except Exception as e:
                print(f"Error processing scan index {i}: {e}")
    
    # If only one scan was loaded, assume a rocking curve scan: swap axes.
    diff_data = np.asarray(diff_data)
    if num_scans == 1:
        diff_data = np.swapaxes(diff_data, 0, 1)
        print("Assume it is a rocking curve scan; number of angles = {}".format(diff_data.shape[0]))
    
    return diff_data

def load_and_sum_db(sid, det):
        sid = int(sid)
        data_type  = 'float32'
        data_name  = '/entry/instrument/detector/data'
        file_names = get_path(sid, det)  # get_path() must be defined in your environment
        num_subscan = len(file_names)
        
        # If there is only one subscan, load it directly.
        if num_subscan == 1:
            with h5py.File(file_names[0], 'r') as f:
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
    print(f"Using {max_workers} workers (desired: {max_workers}, available cores: {num_cores}).")
    
    # List to hold per-scan summed data (preserving original order).
    scan_sum_list = [None] * num_scans

    # Process scans concurrently.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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


def load_and_sum(file_name):
    dataset_path='/entry/instrument/detector/data'
    with h5py.File(file_name[0], 'r') as f:
        data = np.asarray(f[dataset_path])
    return np.sum(data, axis=0)  # sum along time or scan axis


def sum_all_h5_data_parallel(file_list):
    num_scans = len(file_list)
    results = [None]*num_scans
    
    with ProcessPoolExecutor(max_workers = 10) as executor:
        # Submit all tasks to the pool
        futures = {executor.submit(load_and_sum, file_list[i]): i for i in range(num_scans)}
        for future in tqdm(as_completed(futures), total=num_scans, desc="Processing Scans"):           
            idx = futures[future]
            results[idx] = future.result()  

    # Sum all partial results
    results = np.stack(results)
    sum_all_data = np.sum(results, axis=0)
    return sum_all_data


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


def interp_sub_pix(data, shift_matrix, max_workers = 10):
    '''
    Parallelized function to apply sub-pixel shifts to 3D, 4D, or 5D data.

    Parameters:
        data (ndarray): Input data of shape (N, ...) where N is the first axis.
        shift_matrix (ndarray): Shift values corresponding to each slice.
        max_workers (int): Number of parallel workers (default: 8).

    Returns:
        ndarray: Shifted data of the same shape as input.
    '''
    sz = np.shape(data)
    sz_len = np.size(sz)
    results = np.zeros_like(data)
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
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit a task for each slice
        futures = {executor.submit(process_func, subset, shift_matrix, index): index for index, subset in enumerate(data)}
        for future in tqdm(as_completed(futures), total=sz[0], desc="Parallel interp_sub_pix"):
            idx = futures[future]
            results[idx] = future.result()
    return results


def trans_coor3D(X, Y, Z, M):
    '''Transform coordinates (X,Y,Z) by 3x3 matrix M (e.g., rotation/transform).'''
    sz = X.shape
    coords = np.vstack([
        X.ravel(),
        Y.ravel(),
        Z.ravel()])
    transformed = np.array(M @ coords)
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
    (Vx, Vy, Vz) = np.meshgrid(x_range, y_range, z_range, indexing = 'ij')
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


class RSM:
    def __init__(self, det_data, energy, delta, gamma, num_angle, th_step, pix, det_dist, offset):
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

    def calcRSM(self, coor, data_store='reduced', desired_workers=10):
        """
        Calculate reciprocal space mapping (RSM) from the detector data.
        The bulk of the work is done by interpolating the transformed detector data
        onto a grid. Parallel processing is used to speed up the per-scan interpolation.
        """
        # --- Standard computations to set up matrices and grids ---
        sz = np.shape(self.det_data)
        data_type = self.det_data.dtype
        sz_len = np.size(sz)
        det_row = sz[sz_len - 2]
        det_col = sz[sz_len - 1]
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
        x_rng = np.linspace(1 - round(det_col/2), det_col - round(det_col/2), det_col)
        y_rng = np.linspace(1 - round(det_row/2), det_row - round(det_row/2), det_row)
        z_rng = np.linspace(1 - round(self.num_angle/2), self.num_angle - round(self.num_angle/2), self.num_angle)
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
        pix_sz, xq, yq, zq = create_grid(X, Y, Z, M)
        trans_sz = np.shape(xq)

        # --- Reshape and prepare the detector data ---
        self.det_data = np.squeeze(np.swapaxes(np.expand_dims(self.det_data, axis=-1), 0, -1), 0)
        sz = np.shape(self.det_data)
        self.det_data = np.reshape(self.det_data, [-1, sz[-3], sz[-2], sz[-1]])
        new_sz = np.shape(self.det_data)

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
        

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(interp3_oblique, X, Y, Z, self.det_data[i, :, :, :],
                                 M_inv, xq, yq, zq): i for i in range(new_sz[0])}
            for future in tqdm(as_completed(futures), total=new_sz[0],
                               desc="Processing scans in parallel"):
                idx = futures[future]
                vq = future.result()
                if data_store == 'full':
                    self.full_data[idx, :, :, :] = vq
                else:
                    self.qxz_data[idx, :, :] = np.sum(vq, axis=0)
                    self.qyz_data[idx, :, :] = np.sum(vq, axis=1)

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
        # ... (remaining part of calcSTRAIN unchanged) ...
        if self.data_store == 'full':
            qz_pos = np.sum(np.sum(self.full_data, -2), -2)
            qx_pos = np.sum(np.sum(self.full_data, -1), -2)
            qy_pos = np.sum(np.sum(self.full_data, -1), -1)
        else:
            qz_pos = np.sum(self.qxz_data, -2)
            qx_pos = np.sum(self.qxz_data, -1)
            qy_pos = np.sum(self.qyz_data, -1)
        sz = np.shape(qz_pos)
        if np.size(sz) == 3:
            shift_qz = np.zeros((sz[0], sz[1]))
            shift_qx = np.zeros((sz[0], sz[1]))
            shift_qy = np.zeros((sz[0], sz[1]))
            for i in tqdm(range(sz[0])):
                for j in range(sz[1]):
                    if method == 'com':
                        shift_qz[i, j] = cen_of_mass(qz_pos[i, j, :])
                        shift_qy[i, j] = cen_of_mass(qy_pos[i, j, :])
                        shift_qx[i, j] = cen_of_mass(qx_pos[i, j, :])
        elif np.size(sz) == 2:
            shift_qz = np.zeros((sz[0]))
            shift_qx = np.zeros((sz[0]))
            shift_qy = np.zeros((sz[0]))
            for i in tqdm(range(sz[0])):
                if method == 'com':
                    shift_qz[i] = cen_of_mass(qz_pos[i, :])
                    shift_qy[i] = cen_of_mass(qy_pos[i, :])
                    shift_qx[i] = cen_of_mass(qx_pos[i, :])
        self.strain = -shift_qz * (self.zq[0, 0, 1] - self.zq[0, 0, 0]) / np.linalg.norm(self.h)
        self.tilt_x = shift_qx * (self.xq[0, 1, 0] - self.xq[0, 0, 0]) / np.linalg.norm(self.h)
        self.tilt_y = shift_qy * (self.yq[1, 0, 0] - self.yq[0, 0, 0]) / np.linalg.norm(self.h)
        self.tot = np.sum(qz_pos, -1)
        
    def disp(self):
        
        fig = plt.figure(1)
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
        #plt.savefig('./result.png')
        
    def save(self,output_path):

        if not os.path.exists(output_path):
            os.mkdir(output_path)
            print("Directory '%s' created" %output_path)
        
        file_name = ''.join([output_path,'tot_intensity_map.tif'])
        tifffile.imsave(file_name,np.asarray(self.tot,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,'strain_map.tif'])
        tifffile.imwrite(file_name,np.asarray(self.strain,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,'tilt_x_map.tif'])
        tifffile.imwrite(file_name,np.asarray(self.tilt_x,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,'tilt_y_map.tif'])
        tifffile.imwrite(file_name,np.asarray(self.tilt_y,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,'pos_rsm_xz.obj'])
        if self.data_store == 'full':
            pickle.dump(np.sum(self.full_data,-3),open(file_name,'wb'), protocol = 4)
        else:
            pickle.dump(self.qxz_data,open(file_name,'wb'), protocol = 4)

        file_name = ''.join([output_path,'pos_rsm_yz.obj'])
        if self.data_store == 'full':
            pickle.dump(np.sum(self.full_data,-2),open(file_name,'wb'),protocol = 4)
        else:
            pickle.dump(self.qyz_data,open(file_name,'wb'),protocol = 4)

        file_name = ''.join([output_path,'xq.obj'])
        pickle.dump(self.xq,open(file_name,'wb'),protocol = 4)

        file_name = ''.join([output_path,'yq.obj'])
        pickle.dump(self.yq,open(file_name,'wb'),protocol = 4)

        file_name = ''.join([output_path,'zq.obj'])
        pickle.dump(self.zq,open(file_name,'wb'),protocol = 4)

        file_name = ''.join([output_path,'h.obj'])
        pickle.dump(self.h,open(file_name,'wb'),protocol = 4)

        if plt.fignum_exists(1):
            file_name = ''.join([output_path,'results.png'])
            plt.savefig(file_name)
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
    return idx
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

def get_path(scan_id, key_name='merlin1', db=db):
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

def interactive_map(names,im_stack,label,data_4D, cmap='jet', clim=None, marker_color = 'black'):

    l = len(names)
    im_sz = np.shape(im_stack)
    l = np.fmin(l,im_sz[0])

    num_maps = l + 1
    layout_row = np.round(np.sqrt(num_maps))
    layout_col = np.ceil(num_maps/layout_row)
    layout_row = int (layout_row)
    layout_col = int (layout_col)

    fig, axs = plt.subplots(layout_row,layout_col)
    size_y = layout_row*4
    size_x = layout_col*6
    if size_x < 8:
        size_y = size_y*8/size_x
        size_x = 8
    if size_y < 6:
        size_x = size_x*6/size_y
        size_y = 6
    fig.set_size_inches(size_x,size_y)
    for i in range(l):
        axs[np.unravel_index(i,[layout_row,layout_col])].imshow(im_stack[i,:,:],cmap=cmap)
        axs[np.unravel_index(i,[layout_row,layout_col])].set_title(names[i])
    im_diff = axs[np.unravel_index(l,[layout_row,layout_col])].imshow(data_4D[0,0,:,:],cmap=cmap,clim=clim)
    axs[np.unravel_index(l,[layout_row,layout_col])].set_title(label)
    if layout_col*layout_row > num_maps:
        for i in range(num_maps,layout_row*layout_col):
            axs[np.unravel_index(i,[layout_row,layout_col])].axis('off')
    fig.tight_layout()
    fig.colorbar(im_diff, ax=axs[np.unravel_index(l,[layout_row,layout_col])])

    def onclick(event):
        global row, col
        col, row = event.xdata, event.ydata
        if col is not None and row is not None and col <= im_sz[2] and row <= im_sz[1]:
            row = int(np.round(row))
            col = int(np.round(col))
            for i in range(l):
                axs[np.unravel_index(i,[layout_row,layout_col])].clear()
                axs[np.unravel_index(i,[layout_row,layout_col])].imshow(im_stack[i,:,:],cmap=cmap)
                axs[np.unravel_index(i,[layout_row,layout_col])].set_title(names[i])
                axs[np.unravel_index(i,[layout_row,layout_col])].plot(col,row,marker='o',markersize=4, color=marker_color)
            axs[np.unravel_index(l,[layout_row,layout_col])].imshow(data_4D[row,col,:,:],cmap=cmap,clim=clim)
            axs[np.unravel_index(l,[layout_row,layout_col])].set_title(label)
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

def load_diff_data(sid,scaler_names,det_name, mon = None):
    h = db[sid]
    df = h.table()
    s = scan_command(sid)
    x_mot = s.split()[0]
    y_mot = s.split()[4]
    scan_col = int (s.split()[3])
    scan_row = int (s.split()[7])
    diff_data = list(h.data(det_name))
    im_stack = []
    print(f'row = {scan_row} col = {scan_col}')
    for name in scaler_names:

        if name in df:
            tmp = df[name]
        else:
            tmp = (df['Det1_{}'.format(name)] + df['Det2_{}'.format(name)] + df['Det3_{}'.format(name)])
        #print(np.shape(tmp))
        tmp = np.reshape(np.asarray(tmp),(1,scan_row, scan_col))

        if len(im_stack) == 0 :
            im_stack = tmp
        else:
            im_stack = np.concatenate([im_stack,tmp],axis=0)
    if mon is not None:
        mon_var = df[mon]
        im_stack = im_stack/np.expand_dims(mon_var,0)
    sz = np.shape(diff_data)
    return im_stack, np.reshape(diff_data,(scan_row,scan_col,sz[2],sz[3]))

def create_movie(desc, names,im_stack,label,data_4D,path,cmap='jet',color='white'):
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
    
    l = len(names)
    im_sz = np.shape(im_stack)
    l = np.fmin(l,im_sz[0])

    num_maps = l + 1
    layout_row = np.round(np.sqrt(num_maps))
    layout_col = np.ceil(num_maps/layout_row)
    layout_row = int (layout_row)
    layout_col = int (layout_col)
    #plt.figure()
    fig, axs = plt.subplots(layout_row,layout_col)
    size_y = layout_row*4
    size_x = layout_col*6
    if size_x < 8:
        size_y = size_y*8/size_x
        size_x = 8
    if size_y < 6:
        size_x = size_x*6/size_y
        size_y = 6
    fig.set_size_inches(size_x,size_y)
    for i in range(l):
        axs[np.unravel_index(i,[layout_row,layout_col])].imshow(im_stack[i,:,:],cmap=cmap)
        axs[np.unravel_index(i,[layout_row,layout_col])].set_title(names[i])
    axs[np.unravel_index(l,[layout_row,layout_col])].imshow(data_4D[0,0,:,:],cmap=cmap)
    axs[np.unravel_index(l,[layout_row,layout_col])].set_title(label)
    if layout_col*layout_row > num_maps:
        for i in range(num_maps,layout_row*layout_col):
            axs[np.unravel_index(i,[layout_row,layout_col])].axis('off')
    fig.tight_layout()
    
    def update_fig(row,col,cmap=cmap,color=color):

        plt.cla()
        for i in range(l):
            axs[np.unravel_index(i,[layout_row,layout_col])].imshow(im_stack[i,:,:],cmap=cmap)
            axs[np.unravel_index(i,[layout_row,layout_col])].plot(col,row,marker='o',markersize=2, color=color)
            axs[np.unravel_index(i,[layout_row,layout_col])].set_title(names[i])
        axs[np.unravel_index(l,[layout_row,layout_col])].imshow(data_4D[row,col,:,:],cmap=cmap)
        axs[np.unravel_index(l,[layout_row,layout_col])].set_title(label)
        if layout_col*layout_row > num_maps:
            for i in range(num_maps,layout_row*layout_col):
                axs[np.unravel_index(i,[layout_row,layout_col])].axis('off')
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
            update_fig(path[j,0],path[j,1],cmap=cmap,color=color)
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