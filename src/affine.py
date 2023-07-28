from fishspot.filter import white_tophat
from fishspot.detect import detect_spots_log
import numpy as np 
import cv2 

# Modified version from big stream for 2D
def blob_detection(
    image,
    min_blob_radius,
    max_blob_radius,
    **kwargs,
):
    """
    """

    wth = white_tophat(image, max_blob_radius)
    spots = detect_spots_log(
        wth,
        min_blob_radius,
        max_blob_radius,
        **kwargs,
    ).astype(int)
    intensities = image[spots[:, 0], spots[:, 1]]
    return np.hstack((spots[:, :2], intensities[..., None]))

def get_spot_context(image, spots, vox, radius):
    """
    """

    output = []
    for spot in spots:
        s = (spot/vox).astype(int)
        w = image[s[0]-radius:s[0]+radius+1,
                  s[1]-radius:s[1]+radius+1]
        output.append( [spot, w] )
    return output   

def _stats(arr):
    """
    """

    # compute mean and standard deviation along columns
    arr = arr.astype(np.float64)
    means = np.mean(arr, axis=1)
    sqr_means = np.mean(np.square(arr), axis=1)
    stddevs = np.sqrt( sqr_means - np.square(means) )
    return means, stddevs

def pairwise_correlation(A, B):
    """
    """

    # grab and flatten context
    a_con = np.array( [a[1].flatten() for a in A] )
    b_con = np.array( [b[1].flatten() for b in B] )

    # get means and std for all contexts, center contexts
    a_mean, a_std = _stats(a_con)
    b_mean, b_std = _stats(b_con)
    a_con = a_con - a_mean[..., None]
    b_con = b_con - b_mean[..., None]

    # compute pairwise correlations
    corr = np.matmul(a_con, b_con.T)
    corr = corr / a_std[..., None]
    corr = corr / b_std[None, ...]
    corr = corr / a_con.shape[1]

    # contexts with no variability are nan, set to 0
    corr[np.isnan(corr)] = 0
    return corr

def match_points(A, B, scores, threshold):
    """
    """

    # split positions from context
    a_pos = np.array( [a[0] for a in A] )
    b_pos = np.array( [b[0] for b in B] )

    # get highest scores above threshold
    best_indcs = np.argmax(scores, axis=1)
    a_indcs = range(len(a_pos))
    keeps = scores[(a_indcs, best_indcs)] > threshold

    # return positions of corresponding points
    return a_pos[keeps, :2], b_pos[best_indcs[keeps], :2]

def ransac_align_points(
    pA, pB, 
    threshold,
    diagonal_constraint=0.75,
    default=np.eye(3),
):
    """
    """

    # sensible requirement of 50 or more spots to compute ransac affine
    if len(pA) < 50 or len(pB) < 50:
        if default is not None:
            print("Insufficient spot matches for ransac")
            print("Returning default")
            return default
        else:
            message = "Insufficient spot matches for ransac"
            message += ", need 50 or more"
            raise ValueError(message)

    # compute the affine
    Aff, inline = cv2.estimateAffine2D(
        pA, pB,
        ransacReprojThreshold=threshold,
        confidence=0.999,
    )

    # rarely ransac just doesn't work (depends on data and parameters)
    # sensible choices for hard constraints on the affine matrix
    if np.any( np.diag(Aff) < diagonal_constraint ):
        if default is not None:
            print("Degenerate affine produced")
            print("Returning default")
            return default
        else:
            message = "Degenerate affine produced"
            message += ", ransac failed"
            raise ValueError(message)

    # augment affine to 4x4 matrix
    affine = np.eye(3)
    affine[:2, :] = Aff

    return affine

def ransac_affine(
    fix, mov,
    min_radius,
    max_radius,
    match_threshold,
    fix_spacing=1, mov_spacing=1,
    cc_radius=12,
    nspots=5000,
    align_threshold=2.0,
    num_sigma_max=15,
    verbose=True,
    fix_spots=None,
    mov_spots=None,
    default=np.eye(4),
    **kwargs,
):
    """
    """

    if verbose:
        print('Getting key points')

    # get spots
    if fix_spots is None:
        fix_spots = blob_detection(
            fix, min_radius, max_radius,
            num_sigma=min(max_radius-min_radius, num_sigma_max),
            threshold=0, exclude_border=cc_radius,
        )
        if fix_spots.shape[0] < 50:
            print('Fewer than 50 spots found in fixed image, returning default')
            return default
        if verbose:
            ns = fix_spots.shape[0]
            print(f'FIXED image: found {ns} key points')

    if mov_spots is None:
        mov_spots = blob_detection(
            mov, min_radius, max_radius,
            num_sigma=min(max_radius-min_radius, num_sigma_max),
            threshold=0, exclude_border=cc_radius,
        )
        if mov_spots.shape[0] < 50:
            print('Fewer than 50 spots found in moving image, returning default')
            return default
        if verbose:
            ns = mov_spots.shape[0]
            print(f'MOVING image: found {ns} key points')

    # sort
    sort_idx = np.argsort(fix_spots[:, 2])[::-1]
    fix_spots = fix_spots[sort_idx, :2][:nspots]
    sort_idx = np.argsort(mov_spots[:, 2])[::-1]
    mov_spots = mov_spots[sort_idx, :2][:nspots]

    # convert to physical units
    fix_spots = fix_spots * fix_spacing
    mov_spots = mov_spots * mov_spacing

    # get contexts
    fix_spots = get_spot_context(
        fix, fix_spots, fix_spacing, cc_radius,
    )
    mov_spots = get_spot_context(
        mov, mov_spots, mov_spacing, cc_radius,
    )

    # get point correspondences
    correlations = pairwise_correlation(
        fix_spots, mov_spots,
    )
    fix_spots, mov_spots = match_points(
        fix_spots, mov_spots,
        correlations, match_threshold,
    )
    if verbose:
        ns = fix_spots.shape[0]
        print(f'MATCHED points: found {ns} matched points')

    # align
    return ransac_align_points(
        fix_spots, mov_spots, align_threshold, **kwargs,
    )
