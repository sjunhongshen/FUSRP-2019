#!/Users/edelsonc/miniconda3/bin/python
"""
Test class for python neighbor tracking

author: edelsonc
created: 02/11/2017
"""
import numpy
import itertools

class Voxel(object):
    """
    Voxel object that tracks the nearest neighbors or a given voxel
    
    Example
    -------
    >>> import numpy
    >>> z, y, x = 3, 5, 5
    >>> test = numpy.arange(75).reshape((z, y, x))
    >>> voxel = Voxel(65, (z, y, x))
    >>> voxel.get_neighbors(flat=True)
    [34, 35, 36, 39, 40, 41, 44, 45, 46, 59, 60, 61, 64, 66, 69, 70, 71, 84, 85, 86, 89, 90, 91, 94, 95, 96]
    >>> voxel.get_neighbors(flat=False) #doctest: +ELLIPSIS
    [(1, 2, -1),...]
    """
    def __init__(self, f_index, dim):
        """
        Arguments
        ---------
        f_index -- voxel index in the flattened array
        dim -- tuple of original 3D dimension (depth, row, column)
        """
        self.dim = dim
        self.f_idx = f_index
        self.idx = numpy.unravel_index(f_index, dim)  # get 3D index
        self.z, self.y, self.x = ( i - 1 for i in dim )

    def get_neighbors(self, square_size=1, flat=False):
        """
        Returns a list of the indexes of the neighboring voxels

        Arguments
        ---------
        square_size -- number of pixels to go out in each direction
        flat -- if the indexes should be flat indexes or not; False default
        """
        zs = ( self.idx[0]+i for i in range(-square_size, square_size + 1) )
        ys = ( self.idx[1]+i for i in range(-square_size, square_size + 1) )
        xs = ( self.idx[2]+i for i in range(-square_size, square_size + 1) )

        neighbors = []
        for idx in itertools.product(zs, ys, xs):
            c0 = idx != self.idx
            c1 = not ( idx[0] < 0 or idx[0] > self.z )
            c2 = not ( idx[1] < 0 or idx[1] > self.y )
            c3 = not ( idx[2] < 0 or idx[2] > self.x )
        
            if c0 and c1 and c2 and c3:
                neighbors.append(idx)

        if flat == True:
            return [self.flatten_index(i, self.dim) for i in neighbors]
        
        return neighbors

    @staticmethod
    def flatten_index(idx, dim):
        """
        Returns a flatten index for a given 3D index
        """
        return dim[1]*dim[2]*idx[0] + dim[2]*idx[1] + idx[2]

if __name__ == '__main__':
    import doctest
    doctest.testmod()