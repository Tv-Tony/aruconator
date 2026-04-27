import numpy as np

def get_perspective_transform(src_pts, dst_pts):
    """
    Calculates the 3x3 perspective transformation matrix.
        
    Returns:
        3x3 transformation matrix.
    """
    A = []
    for i in range(4):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    
    A = np.array(A)
    # Solve A * h = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    # The solution is the last row of V (or last column of Vt)
    h = Vt[-1, :]
    return h.reshape(3, 3) / h[-1]

def custom_warp_perspective(img, M, dsize):
    """
    Custom implementation of perspective warping using bilinear interpolation.
    
    Returns:
        Warped image.
    """
    dst_w, dst_h = dsize
    src_h, src_w, channels = img.shape
    
    # Create a grid of coordinates in the destination image
    x, y = np.meshgrid(np.arange(dst_w), np.arange(dst_h))
    ones = np.ones_like(x)
    dst_coords = np.stack([x, y, ones], axis=-1).reshape(-1, 3).T
    
    # Calculate inverse transformation matrix
    M_inv = np.linalg.inv(M)
    
    # Map destination coordinates back to source coordinates
    src_coords = M_inv @ dst_coords
    src_coords /= src_coords[2, :]  
    
    src_x = src_coords[0, :].reshape(dst_h, dst_w)
    src_y = src_coords[1, :].reshape(dst_h, dst_w)
    
    # Bilinear interpolation
    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int)
    y1 = y0 + 1
    
    mask = (src_x >= 0) & (src_x < src_w - 1) & (src_y >= 0) & (src_y < src_h - 1)
    
    output = np.zeros((dst_h, dst_w, channels), dtype=img.dtype)
    
    # Weights for interpolation
    wa = (x1 - src_x) * (y1 - src_y)
    wb = (x1 - src_x) * (src_y - y0)
    wc = (src_x - x0) * (y1 - src_y)
    wd = (src_x - x0) * (src_y - y0)
    
    # Apply weights to each channel
    for c in range(channels):
        img_c = img[:, :, c]
        
        v00 = img_c[y0[mask], x0[mask]]
        v01 = img_c[y1[mask], x0[mask]]
        v10 = img_c[y0[mask], x1[mask]]
        v11 = img_c[y1[mask], x1[mask]]
        
        # Compute interpolated value
        interp_val = (v00 * wa[mask] + v01 * wb[mask] +
                      v10 * wc[mask] + v11 * wd[mask])
        
        output[mask, c] = interp_val.astype(img.dtype)
        
    return output
