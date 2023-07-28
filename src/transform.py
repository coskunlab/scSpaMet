from scipy.ndimage import map_coordinates
import numpy as np 

def position_grid(shape):
    """
    Create a matrix of grid for all points
    """

    coords = np.meshgrid(*[range(x) for x in shape], indexing='ij')
    coords = np.array(coords).astype(np.int16)
    return np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def affine_to_grid(matrix, grid, displacement=True):
    """
    Transform affine transformation matrix to grid
    """

    mm = matrix[:2, :2]
    tt = matrix[:2, -1]
    result = np.einsum('...ij,...j->...i', mm, grid) + tt
    if displacement:
        result = result - grid
    return result


def interpolate_image(image, X, order=1):
    """
    Map the input image to new coordinates by interpolation
    """

    X = np.moveaxis(X, -1, 0)
    return map_coordinates(image, X, order=order, mode='constant')


def apply_global_affine(
    fix, mov,
    affine,
    fix_spacing=1, mov_spacing=1,
    order=1,
):
    """
    Apply global affine transform
    """

    grid = position_grid(fix.shape) * fix_spacing
    coords = affine_to_grid(affine, grid, displacement=False) / mov_spacing
    return interpolate_image(mov, coords, order=order)
