#  Copyright (C) 2012 Daniel Maturana
#  This file is part of binvox-rw-py.
#
#  binvox-rw-py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  binvox-rw-py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with binvox-rw-py. If not, see <http://www.gnu.org/licenses/>.
#

"""
Binvox to Numpy and back.


>>> import numpy as np
>>> import binvox_rw
>>> with open('chair.binvox', 'rb') as f:
...     m1 = binvox_rw.read_as_3d_array(f)
...
>>> m1.dims
[32, 32, 32]
>>> m1.scale
41.133000000000003
>>> m1.translate
[0.0, 0.0, 0.0]
>>> with open('chair_out.binvox', 'wb') as f:
...     m1.write(f)
...
>>> with open('chair_out.binvox', 'rb') as f:
...     m2 = binvox_rw.read_as_3d_array(f)
...
>>> m1.dims==m2.dims
True
>>> m1.scale==m2.scale
True
>>> m1.translate==m2.translate
True
>>> np.all(m1.data==m2.data)
True

>>> with open('chair.binvox', 'rb') as f:
...     md = binvox_rw.read_as_3d_array(f)
...
>>> with open('chair.binvox', 'rb') as f:
...     ms = binvox_rw.read_as_coord_array(f)
...
>>> data_ds = binvox_rw.dense_to_sparse(md.data)
>>> data_sd = binvox_rw.sparse_to_dense(ms.data, 32)
>>> np.all(data_sd==md.data)
True
>>> # the ordering of elements returned by numpy.nonzero changes with axis
>>> # ordering, so to compare for equality we first lexically sort the voxels.
>>> np.all(ms.data[:, np.lexsort(ms.data)] == data_ds[:, np.lexsort(data_ds)])
True
"""

import numpy as np
from skan import Skeleton, summarize, csr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
from voxel_tool import Voxel
"""
import ij.process as process
from ij import IJ, ImagePlus
from sc.fiji.skeletonize3D import Skeletonize3D_
from sc.fiji.analyzeSkeleton import AnalyzeSkeleton_, Point
"""

class VoxelModel(object):
    """ Holds a binvox model.
    data is either a three-dimensional numpy boolean array (dense representation)
    or a two-dimensional numpy float array (coordinate representation).

    dims, translate and scale are the model metadata.

    dims are the voxel dimensions, e.g. [32, 32, 32] for a 32x32x32 model.

    scale and translate relate the voxels to the original model coordinates.

    To translate voxel coordinates i, j, k to original coordinates x, y, z:

    x_n = (i+.5)/dims[0]
    y_n = (j+.5)/dims[1]
    z_n = (k+.5)/dims[2]
    x = scale*x_n + translate[0]
    y = scale*y_n + translate[1]
    z = scale*z_n + translate[2]

    """

    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)

    def write(self, fp):
        write_binvox(self, fp)

def read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale

def read_as_3d_array(fp, fix_coords=True):
    """ Read binary binvox format as array.

    Returns the model with accompanying metadata.

    Voxels are stored in a three-dimensional numpy array, which is simple and
    direct, but may use a lot of memory for large models. (Storage requirements
    are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
    boolean arrays use a byte per element).

    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    return VoxelModel(data, dims, translate, scale, axis_order)

def read_as_coord_array(fp, fix_coords=True):
    """ Read binary binvox format as coordinates.

    Returns binvox model with voxels in a "coordinate" representation, i.e.  an
    3 x N array where N is the number of nonzero voxels. Each column
    corresponds to a nonzero voxel and the 3 rows are the (x, z, y) coordinates
    of the voxel.  (The odd ordering is due to the way binvox format lays out
    data).  Note that coordinates refer to the binvox voxels, without any
    scaling or translation.

    Use this to save memory if your model is very sparse (mostly empty).

    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)

    values, counts = raw_data[::2], raw_data[1::2]

    sz = np.prod(dims)
    index, end_index = 0, 0
    end_indices = np.cumsum(counts)
    indices = np.concatenate(([0], end_indices[:-1])).astype(end_indices.dtype)

    values = values.astype(np.bool)
    indices = indices[values]
    end_indices = end_indices[values]

    nz_voxels = []
    for index, end_index in zip(indices, end_indices):
        nz_voxels.extend(range(index, end_index))
    nz_voxels = np.array(nz_voxels)
    # TODO are these dims correct?
    # according to docs,
    # index = x * wxh + z * width + y; // wxh = width * height = d * d

    x = nz_voxels / (dims[0]*dims[1])
    zwpy = nz_voxels % (dims[0]*dims[1]) # z*w + y
    z = zwpy / dims[0]
    y = zwpy % dims[0]
    if fix_coords:
        data = np.vstack((x, y, z))
        axis_order = 'xyz'
    else:
        data = np.vstack((x, z, y))
        axis_order = 'xzy'

    #return Voxels(data, dims, translate, scale, axis_order)
    return VoxelModel(np.ascontiguousarray(data), dims, translate, scale, axis_order)

def dense_to_sparse(voxel_data, dtype=np.int):
    """ From dense representation to sparse (coordinate) representation.
    No coordinate reordering.
    """
    if voxel_data.ndim!=3:
        raise ValueError('voxel_data is wrong shape; should be 3D array.')
    return np.asarray(np.nonzero(voxel_data), dtype)

def sparse_to_dense(voxel_data, dims, dtype=np.bool):
    if voxel_data.ndim!=2 or voxel_data.shape[0]!=3:
        raise ValueError('voxel_data is wrong shape; should be 3xN array.')
    if np.isscalar(dims):
        dims = [dims]*3
    dims = np.atleast_2d(dims).T
    # truncate to integers
    xyz = voxel_data.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dims), 0)
    xyz = xyz[:,valid_ix]
    out = np.zeros(dims.flatten(), dtype=dtype)
    out[tuple(xyz)] = True
    return out

#def get_linear_index(x, y, z, dims):
    #""" Assuming xzy order. (y increasing fastest.
    #TODO ensure this is right when dims are not all same
    #"""
    #return x*(dims[1]*dims[2]) + z*dims[1] + y

def bwrite(fp,s):
    fp.write(s.encode())
    

def write_pair(fp,state, ctr):
    fp.write(struct.pack('B',state))
    fp.write(struct.pack('B',ctr))
    
    

def write_binvox(voxel_model, fp):
    """ Write binary binvox format.

    Note that when saving a model in sparse (coordinate) format, it is first
    converted to dense format.

    Doesn't check if the model is 'sane'.

    """
    if voxel_model.data.ndim==2:
        # TODO avoid conversion to dense
        dense_voxel_data = sparse_to_dense(voxel_model.data, voxel_model.dims)
    else:
        dense_voxel_data = voxel_model.data

    bwrite(fp,'#binvox 1\n')
    bwrite(fp,'dim '+' '.join(map(str, voxel_model.dims))+'\n')
    bwrite(fp,'translate '+' '.join(map(str, voxel_model.translate))+'\n')
    bwrite(fp,'scale '+str(voxel_model.scale)+'\n')
    bwrite(fp,'data\n')
    if not voxel_model.axis_order in ('xzy', 'xyz'):
        raise ValueError('Unsupported voxel model axis order')

    if voxel_model.axis_order=='xzy':
        voxels_flat = dense_voxel_data.flatten()
    elif voxel_model.axis_order=='xyz':
        voxels_flat = np.transpose(dense_voxel_data, (0, 2, 1)).flatten()

    # keep a sort of state machine for writing run length encoding
    state = voxels_flat[0]
    ctr = 0
    for c in voxels_flat:
        if c==state:
            ctr += 1
            # if ctr hits max, dump
            if ctr==255:
                write_pair(fp, state, ctr)
                ctr = 0
        else:
            # if switch state, dump
            write_pair(fp, state, ctr)
            state = c
            ctr = 1
    # flush out remainders
    if ctr > 0:
        write_pair(fp, state, ctr)

def skeleton_properties(imp):
    """ Retrieves lists of endpoints, junction points, junction
        voxels and total length from a skeletonized image
    """
    skel_analyzer = AnalyzeSkeleton_()
    skel_analyzer.setup("", imp)
    skel_result = skel_analyzer.run()

    avg_lengths = skel_result.getAverageBranchLength()
    n_branches = skel_result.getBranches()
    lengths = [n*avg for n, avg in zip(n_branches, avg_lengths)]
    total_length = sum(lengths)

    return (skel_result.getListOfEndPoints(), skel_result.getJunctions(),
            skel_result.getListOfJunctionVoxels(), total_length)


import numpy as np

from utils import divide_nonzero
from hessian import absolute_hessian_eigenvalues


def frangi(nd_array, scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=500, black_vessels=True):

    if not nd_array.ndim == 3:
        raise(ValueError("Only 3 dimensions is currently supported"))

    # from https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/_frangi.py#L74
    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    filtered_array = np.zeros(sigmas.shape + nd_array.shape)

    for i, sigma in enumerate(sigmas):
        eigenvalues = absolute_hessian_eigenvalues(nd_array, sigma=sigma, scale=True)
        filtered_array[i] = compute_vesselness(*eigenvalues, alpha=alpha, beta=beta, c=frangi_c,
                                               black_white=black_vessels)

    return np.max(filtered_array, axis=0)


def compute_measures(eigen1, eigen2, eigen3):
    """
    RA - plate-like structures
    RB - blob-like structures
    S - background
    """
    Ra = divide_nonzero(np.abs(eigen2), np.abs(eigen3))
    Rb = divide_nonzero(np.abs(eigen1), np.sqrt(np.abs(np.multiply(eigen2, eigen3))))
    S = np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))
    return Ra, Rb, S


def compute_plate_like_factor(Ra, alpha):
    return 1 - np.exp(np.negative(np.square(Ra)) / (2 * np.square(alpha)))


def compute_blob_like_factor(Rb, beta):
    return np.exp(np.negative(np.square(Rb) / (2 * np.square(beta))))


def compute_background_factor(S, c):
    return 1 - np.exp(np.negative(np.square(S)) / (2 * np.square(c)))


def compute_vesselness(eigen1, eigen2, eigen3, alpha, beta, c, black_white):
    Ra, Rb, S = compute_measures(eigen1, eigen2, eigen3)
    plate = compute_plate_like_factor(Ra, alpha)
    blob = compute_blob_like_factor(Rb, beta)
    background = compute_background_factor(S, c)
    return filter_out_background(plate * blob * background, black_white, eigen2, eigen3)


def filter_out_background(voxel_data, black_white, eigen2, eigen3):
    """
    Set black_white to true if vessels are darker than the background and to false if
    vessels are brighter than the background.
    """
    if black_white:
        voxel_data[eigen2 < 0] = 0
        voxel_data[eigen3 < 0] = 0
    else:
        voxel_data[eigen2 > 0] = 0
        voxel_data[eigen3 > 0] = 0
    voxel_data[np.isnan(voxel_data)] = 0
    return voxel_data

def get_boundry(data):
    import scipy.ndimage as ndimage
    from scipy.spatial import cKDTree
    data = data.copy(order='C')
    mask = data != 0

    struct = ndimage.generate_binary_structure(3, 3)
    erode = ndimage.binary_erosion(mask, struct)
    edges = mask ^ erode
    
    new = np.zeros(data.shape, dtype=int)
    new[edges] = 1
    new = np.transpose(new, (2, 0, 1))
    return new
    

def denoise(img_file, write_file):
    #img_file = '/Users/kimihirochin/Desktop/mesh/test_1_thinned.binvox'
    with open(img_file, 'rb') as f:
        model = read_as_3d_array(f)
        vol = model.data
        size = vol.shape
        
        from voxel_tool import Voxel
        idx = np.argmax(vol)  # find the index of the maximum intensity pixel
        #idx = np.unravel_index(np.argmax(vol, axis=None), vol.shape)
        for idx in range(np.prod(size)):
            idx_arr = np.unravel_index(idx, size)
            if idx % 1000000 == 0:
                print(idx)
            if not vol[idx_arr]:
                continue
            isolated = True
            voxel = Voxel(idx, size)
            for i in voxel.get_neighbors():
                if vol[i]:
                    isolated = False
                    break
            if isolated:
                vol[idx_arr] = False
        
        #new_f = open('/Users/kimihirochin/Desktop/mesh/test_1_thinned_denoised.binvox', "xb")
        new_f = open(new_file, "xb")
        new_model = VoxelModel(vol, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)

def get_coords_file(img_file, fid):
    img_file = '/Users/kimihirochin/Desktop/mesh/test_1_main_structure_>10.binvox'
    with open(img_file, 'rb') as f:
        model = read_as_3d_array(f)
        g, coords, degimg = csr.skeleton_to_csgraph(np.array(model.data), spacing=[1, 1, 1])
        mask1 = degimg == 1
        print("%d" %np.count_nonzero(mask1 != 0))
        mask2 = degimg == 2
        print("%d" %np.count_nonzero(mask2 != 0))
        mask3 = degimg >= 3
        print("%d" %np.count_nonzero(mask3 != 0))

        branch_end = mask1 | mask3
        edge = mask2

        vol = branch_end
        size = vol.shape
        for idx in range(np.prod(size)):
            idx_arr = np.unravel_index(idx, size)
            if idx % 10000000 == 0:
                print(idx)
            if not vol[idx_arr]:
                continue
            voxel = Voxel(idx, size)
            for i in voxel.get_neighbors(square_size=4):
                if vol[i]:
                    vol[i] = False
        print("%d" %np.count_nonzero(vol != 0))         
        new_f = open('/Users/kimihirochin/Desktop/mesh/test_1_branch_end.binvox', "xb")
        new_model = VoxelModel(vol, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)

        vol = edge
        size = vol.shape
        for idx in range(np.prod(size)):
            idx_arr = np.unravel_index(idx, size)
            if idx % 10000000 == 0:
                print(idx)
            if not vol[idx_arr]:
                continue
            voxel = Voxel(idx, size)
            for i in voxel.get_neighbors(square_size=5):
                if vol[i]:
                    vol[i] = False
        print("%d" %np.count_nonzero(vol != 0))  
        new_f = open('/Users/kimihirochin/Desktop/mesh/test_1_edge_points.binvox', "xb")
        new_model = VoxelModel(vol, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)
        exit()
        new_f = open('/Users/kimihirochin/Desktop/mesh/test_1_deg1.binvox', "xb")
        new_model = VoxelModel(mask3, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)
        #print(np.max(degimg))
        exit()
        degrees = np.diff(g.indptr)
        branching_pts = coords[np.where(degrees >= 3)]
        turning_pts = coords[np.where(degrees == 2)]
        end_pts = coords[np.where(degrees == 1)]
        np.save('/Users/kimihirochin/Desktop/mesh/test_%d_branching_points.npy' % fid, branching_pts)
        np.save('/Users/kimihirochin/Desktop/mesh/test_%d_turning_points.npy' % fid, turning_pts)
        np.save('/Users/kimihirochin/Desktop/mesh/test_%d_end_points.npy' % fid, end_pts)
        stats = csr.branch_statistics(g)
        return stats

def get_main_struct(img_file, fid):
    img_file = '/Users/kimihirochin/Desktop/mesh/test_1_thinned_denoised.binvox'
    with open(img_file, 'rb') as f:
        model = read_as_3d_array(f)
        import cc3d
        labels_in = np.array(model.data)
        labels_out = cc3d.connected_components(labels_in) # 26-connected

        N = np.max(labels_out)
        main = np.zeros(labels_in.shape, dtype=int)
        count = 1
        for segid in range(1, N+1):
            extracted_image = labels_out * (labels_out == segid)
            extracted_image = np.array(np.array(extracted_image, dtype=bool), dtype=int)
            if np.count_nonzero(extracted_image != 0) > 10:
                # new_f = open('/Users/kimihirochin/Desktop/mesh/test_1_main_structure_%d.binvox' % (fid, count), "xb")
                # new_model = VoxelModel(extracted_image, model.dims, model.translate, model.scale, model.axis_order)
                # write_binvox(new_model, new_f)
                main = main + extracted_image
                print("%d: %d" %(segid, np.count_nonzero(extracted_image != 0)))
                count += 1
        main = np.array(np.array(main, dtype=bool), dtype=int)
        new_f = open('/Users/kimihirochin/Desktop/mesh/test_1_main_structure_>10.binvox', "xb")
        new_model = VoxelModel(main, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)


if __name__ == '__main__':
    import scipy, nibabel
    filename = '/Users/kimihirochin/Desktop/mesh/IXI002-Guys-0828-ANGIOSENS_-s256_-0701-00007-000001-01.skull.label.nii.gz'
    # get_main_struct(filename,1)
    get_coords_file(filename,1)
    path = '/Users/kimihirochin/Desktop/mesh/test_1_skull_boundry_1.nii.gz'
    path2 = '/Users/kimihirochin/Desktop/mesh/test_1_skull_boundry_2.nii.gz'
    image_array = nibabel.load(filename).get_data()
    aff = nibabel.load(filename).affine
    new = get_boundry(image_array)
    img = nibabel.Nifti1Image(new, aff)
   # img2 = nibabel.Nifti1Image(data_new, aff)
    nibabel.save(img, path)
    #nibabel.save(img2, path2)
    exit()
    """
    img_before = '/Users/kimihirochin/Desktop/mesh/sample1_2.binvox'
    with open(img_before, 'rb') as f:
        model = read_as_3d_array(f)
        vol = np.array(model.data, dtype=int) * 255
        size = vol.shape
        ret = frangi(vol)
        print(ret.shape)
        print(np.max(ret))
        print(np.count_nonzero(ret != 0))
    """
    """
    img_file = '/Users/kimihirochin/Desktop/mesh/test_1_thinned.binvox'
    with open(img_file, 'rb') as f:
        model = read_as_3d_array(f)
        vol = model.data
        size = vol.shape
        print(np.sum(np.array(~vol, dtype=int)))
        
        from voxel_tool import Voxel
        idx = np.argmax(vol)  # find the index of the maximum intensity pixel
        #idx = np.unravel_index(np.argmax(vol, axis=None), vol.shape)
        for idx in range(np.prod(size)):
            idx_arr = np.unravel_index(idx, size)
            if idx % 1000000 == 0:
                print(idx)
            if not vol[idx_arr]:
                continue
            isolated = True
            voxel = Voxel(idx, size)
            for i in voxel.get_neighbors():
                if vol[i]:
                    isolated = False
                    break
            if isolated:
                vol[idx_arr] = False
        
        new_f = open('/Users/kimihirochin/Desktop/mesh/test_1_thinned_denoised.binvox', "xb")
        new_model = VoxelModel(vol, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)
        #model.write(new_f)
        exit()
    """
    """
    """
    img_file = '/Users/kimihirochin/Desktop/mesh/test_1_thinned_denoised.binvox'
    with open(img_file, 'rb') as f:
        model = read_as_3d_array(f)
        # g, coords, degimg = csr.skeleton_to_csgraph(np.array(model.data), spacing=[1, 1, 1])
        # degrees = np.diff(g.indptr)
        # branching_pts = coords[np.where(degrees >= 3)]
        # turning_pts = coords[np.where(degrees == 2)]
        # end_pts = coords[np.where(degrees == 1)]
        # np.save('/Users/kimihirochin/Desktop/mesh/test_1_branching_points.npy', branching_pts)
        # np.save('/Users/kimihirochin/Desktop/mesh/test_1_turning_points.npy', turning_pts)
        # np.save('/Users/kimihirochin/Desktop/mesh/test_1_end_points.npy', end_pts)
        # print(branching_pts)
        # print(branching_pts.shape)
        # stats = csr.branch_statistics(g)
        # print(stats)

        import cc3d
        labels_in = np.array(model.data)
        labels_out = cc3d.connected_components(labels_in) # 26-connected

        N = np.max(labels_out)
        print(N)
        main = np.zeros(labels_in.shape, dtype=int)
        count = 1
        for segid in range(1, N+1):
            extracted_image = labels_out * (labels_out == segid)
            #print(np.max(extracted_image))
            extracted_image = np.array(np.array(extracted_image, dtype=bool), dtype=int)
            #print(np.max(extracted_image))
            if np.count_nonzero(extracted_image != 0) > 100:
                #new_f = open('/Users/kimihirochin/Desktop/mesh/test_1_main_structure_%d.binvox' % count, "xb")
                #new_model = VoxelModel(extracted_image, model.dims, model.translate, model.scale, model.axis_order)
                #write_binvox(new_model, new_f)
                main = main + extracted_image
                print("%d: %d" %(segid, np.count_nonzero(extracted_image != 0)))
                count += 1
        main = np.array(np.array(main, dtype=bool), dtype=int)
        new_f = open('/Users/kimihirochin/Desktop/mesh/test_1_main_structure.binvox', "xb")
        new_model = VoxelModel(main, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)

    """
    print("open image...")
    skel= IJ.openImage(img_file)
    height= skel.getHeight()
    width= skel.getWidth()

    print("skeletonizing image...")
    IJ.run(skel, "Skeletonize (2D/3D)", "")
    skel.show()

    end_points, junctions, junction_voxels, total_len = skeleton_properties(impSkel)
    print(end_points)
    print(junctions)
    print(junction_voxels)
    """