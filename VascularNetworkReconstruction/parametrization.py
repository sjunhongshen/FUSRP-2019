import numpy as np
from voxel_tool import Voxel
from matplotlib import pyplot as plt

############################################################################
#### Helper Functions
############################################################################
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


############################################################################
### Main Functions
############################################################################

# TODO: find the minimum set of points that cover the whole surface
def choose_surface_points(path):
    with open(path, 'rb') as f:
        model = read_as_3d_array(f)
        surface_points = model.data
        size = surface_points.shape
        pts_coords = np.transpose(dense_to_sparse(pts), (1, 0))

        chosen_points = np.zeros(size, dtype=int)
        for idx in range(np.prod(size)):
            idx_arr = np.unravel_index(idx, size)
            print("!")
            exit()
            if not surface_points[idx_arr]:
                continue
            voxel = Voxel(idx, size)
            count = 0

            # get the 26 (3 ^ 3 - 1) neighbors; if square_size = 2, then get 5 ^ 3 -1 neighbors
            for i in voxel.get_neighbors(square_size=1):
                if not surface_points[i]:
                    count += 1
            if count >= 10:
                chosen_points[idx_arr] = 1

        new_filename = '/Users/kimihirochin/Desktop/mesh/test_1_chosen_surface_points.binvox'
        new_file = open(new_filename, "xb")
        new_model = VoxelModel(chosen_points, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)

# TODO: parametrize a slice
def paramerize_slice(path):
    with open(path, 'rb') as f:
        model = read_as_3d_array(f)
        surface_points = model.data
        size = surface_points.shape
        x, y, z = size
        pts_coords = np.transpose(dense_to_sparse(pts), (1, 0))

        paramerized_surface = np.zeros(size, dtype=int)

        for i in range(x):
            pass

        new_filename = '/Users/kimihirochin/Desktop/mesh/test_1_parametrized_surface.binvox'
        new_file = open(new_filename, "xb")
        new_model = VoxelModel(paramerized_surface, model.dims, model.translate, model.scale, model.axis_order)
        write_binvox(new_model, new_f)


if __name__ == '__main__':
    path = '/Users/kimihirochin/Desktop/mesh/test_1_hemi_surface.nii.gz'
    choose_surface_points(path)
    
