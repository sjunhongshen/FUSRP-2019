################################################################
# Helper functions for generating the nodes to be used in GCO
#
# Author: Junhong Shen (jhshen@g.ucla.edu)
################################################################

def get_boundry(data):
    """
    Get the boundry voxels of a volume.
    """

    import scipy.ndimage as ndimage
    data = data.copy(order='C')
    mask = data != 0

    struct = ndimage.generate_binary_structure(10, 10)
    erode = ndimage.binary_erosion(mask, struct)
    edges = mask ^ erode
    
    new = np.zeros(data.shape, dtype=int)
    new[edges] = 1
    new = np.transpose(new, (2, 0, 1))
    return new
    

def denoise(img_file, new_file):
    """
    Remove isolated voxels in a volume.
    """

    with open(img_file, 'rb') as f:
        model = read_as_3d_array(f)
        vol = model.data
        size = vol.shape
        
        from voxel_tool import Voxel
        for idx in range(np.prod(size)):
            idx_arr = np.unravel_index(idx, size)
            if idx % 1000000 == 0:
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
        
        new_f = open(new_file, "xb")
        new_model = VoxelModel(vol, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)


def get_coords_file(img_file, write_file):
    """
    For a voxel representation of vessels, return the coordinates of points sampled along the vessels.
    """

    with open(img_file, 'rb') as f:
        model = read_as_3d_array(f)
        g, coords, degimg = csr.skeleton_to_csgraph(np.array(model.data), spacing=[1, 1, 1])
        mask1 = degimg == 1
        mask2 = degimg == 2
        mask3 = degimg >= 3

        branch_end = mask1 | mask3
        edge = mask2

        vol1 = branch_end
        vol2 = edge
        size = vol2.shape
        for idx in range(np.prod(size)):
            idx_arr = np.unravel_index(idx, size)
            if not vol2[idx_arr]:
                continue
            voxel = Voxel(idx, size)
            for i in voxel.get_neighbors(square_size=10):
                if vol2[i]:
                    vol2[i] = False

        vol = vol1 | vol2
        size = vol.shape
        for idx in range(np.prod(size)):
            idx_arr = np.unravel_index(idx, size)
            if not vol[idx_arr]:
                continue
            voxel = Voxel(idx, size)
            for i in voxel.get_neighbors(square_size=5):
                if vol[i]:
                    vol[i] = False

        new_f = open(write_file, "xb")
        new_model = VoxelModel(vol, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)


def get_main_struct(img_file, write_file):
    """
    For a voxel representation of vascular network, return the main structure of the vessels using 
    connected component analysis.
    """

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
                main = main + extracted_image
                print("%d: %d" %(segid, np.count_nonzero(extracted_image != 0)))
                count += 1

        main = np.array(np.array(main, dtype=bool), dtype=int)
        new_f = open(write_file, "xb")
        new_model = VoxelModel(main, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)


def sample_points(img_file, write_file):
    """
    Uniformly sample points in a volume.
    """

    with open(img_file, 'rb') as f:
        model = read_as_3d_array(f)
        vol = model.data
        size = vol.shape

        for idx in range(np.prod(size)):
            idx_arr = np.unravel_index(idx, size)
            if vol[idx_arr]:
                voxel = Voxel(idx, size)
                for i in voxel.get_neighbors(square_size = 3):
                    vol[i] = False

        new_f = open(write_file, "xb")
        new_model = VoxelModel(np.array(vol, dtype=int), model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)
        

def get_edges(img_file, pt_file, write_file):
    """
    Analyze the vessels and the points sampled from the vessels, give the connection information.
    """

    with open(img_file, 'rb') as f, open(pt_file, 'rb') as p1:
        model = read_as_3d_array(f)
        pts = read_as_3d_array(p1).data
        pts_coords = np.transpose(dense_to_sparse(pts), (1, 0))
        import cc3d
        labels_in = np.array(model.data)
        labels_out = cc3d.connected_components(labels_in)

        N = np.max(labels_out)
        edge_list = []
        for segid in range(1, N+1):
            extracted_image = labels_out == segid
            skel = Skeleton(extracted_image)
            for i in range(skel.n_paths):
                coords = skel.path_coordinates(i)
                prev = None
                for c in coords:
                    c = np.array(c, dtype=int)
                    for j in range(len(pts_coords)):
                        if (c == pts_coords[j]).all():
                            if prev == None:
                                prev = j
                            else:
                                cur = j
                                edge_list.append([prev, cur])
                                prev = cur
                            break

        np.save(write_file, edge_list)



#######################################
# Operations on binvox files
#######################################

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
"""

import numpy as np
from skan import Skeleton, summarize, csr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
from voxel_tool import Voxel


class VoxelGrid(object):

    def __init__(self, points, x_y_z=[1, 1, 1], bb_cuboid=True, build=True):
        """
        Parameters
        ----------         
        points: (N,3) ndarray
                The point cloud from wich we want to construct the VoxelGrid.
                Where N is the number of points in the point cloud and the second
                dimension represents the x, y and z coordinates of each point.
        
        x_y_z:  list
                The segments in wich each axis will be divided.
                x_y_z[0]: x axis 
                x_y_z[1]: y axis 
                x_y_z[2]: z axis

        bb_cuboid(Optional): bool
                If True(Default):   
                    The bounding box of the point cloud will be adjusted
                    in order to have all the dimensions of equal lenght.                
                If False:
                    The bounding box is allowed to have dimensions of different sizes.
        """
        self.points = points

        xyzmin = np.min(points, axis=0) - 0.001
        xyzmax = np.max(points, axis=0) + 0.001

        if bb_cuboid:
            #: adjust to obtain a  minimum bounding box with all sides of equal lenght 
            diff = max(xyzmax-xyzmin) - (xyzmax-xyzmin)
            xyzmin = xyzmin - diff / 2
            xyzmax = xyzmax + diff / 2
        
        self.xyzmin = xyzmin
        self.xyzmax = xyzmax

        segments = []
        shape = []

        for i in range(3):
            # note the +1 in num 
            if type(x_y_z[i]) is not int:
                raise TypeError("x_y_z[{}] must be int".format(i))
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
            segments.append(s)
            shape.append(step)
        
        self.segments = segments

        self.shape = shape

        self.n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
        self.n_x = x_y_z[0]
        self.n_y = x_y_z[1]
        self.n_z = x_y_z[2]
        
        self.id = "{},{},{}-{}".format(x_y_z[0], x_y_z[1], x_y_z[2], bb_cuboid)

        if build:
            self.build()


    def build(self):

        structure = np.zeros((len(self.points), 4), dtype=int)

        structure[:,0] = np.searchsorted(self.segments[0], self.points[:,0]) - 1

        structure[:,1] = np.searchsorted(self.segments[1], self.points[:,1]) - 1

        structure[:,2] = np.searchsorted(self.segments[2], self.points[:,2]) - 1

        # i = ((y * n_x) + x) + (z * (n_x * n_y))
        structure[:,3] = ((structure[:,1] * self.n_x) + structure[:,0]) + (structure[:,2] * (self.n_x * self.n_y)) 
        
        self.structure = structure

        vector = np.zeros(self.n_voxels)
        count = np.bincount(self.structure[:,3])
        vector[:len(count)] = count

        self.vector = vector.reshape(self.n_z, self.n_y, self.n_x)


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

