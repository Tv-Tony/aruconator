import numpy as np

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    print("PyTorch CUDA detected — custom warp running on GPU.")
    DEVICE = torch.device("cuda")
else:
    print("PyTorch CUDA not available — custom warp running on CPU.")
    DEVICE = torch.device("cpu") if True else None


def get_perspective_transform(src_pts, dst_pts):
    """
    Calculates the 3x3 perspective transformation matrix via SVD.
    Returns:
        3x3 numpy transformation matrix.
    """
    A = []
    for i in range(4):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    return h.reshape(3, 3) / h[-1]


def custom_warp_perspective(img, M, dsize):
    """
    Custom perspective warping with bilinear interpolation.
    Runs on GPU via PyTorch if available, otherwise falls back to NumPy CPU.
    """
    dst_w, dst_h = dsize
    src_h, src_w, channels = img.shape

    if TORCH_AVAILABLE:
        return _warp_torch(img, M, dst_w, dst_h, src_w, src_h, channels)
    else:
        return _warp_numpy(img, M, dst_w, dst_h, src_w, src_h, channels)


def _warp_torch(img, M, dst_w, dst_h, src_w, src_h, channels):
    # Upload image and inverse homography to GPU
    img_t = torch.from_numpy(img.astype(np.float32)).to(DEVICE)  # H x W x C
    M_inv = torch.from_numpy(np.linalg.inv(M).astype(np.float32)).to(DEVICE)  # 3x3

    # Build destination coordinate grid on GPU
    xs = torch.arange(dst_w, dtype=torch.float32, device=DEVICE)
    ys = torch.arange(dst_h, dtype=torch.float32, device=DEVICE)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(x)
    dst_coords = torch.stack([x, y, ones], dim=0).reshape(3, -1)  # 3 x N

    # Map destination -> source via inverse homography
    src_coords = M_inv @ dst_coords          # 3 x N
    src_coords = src_coords / src_coords[2]  # normalize homogeneous

    src_x = src_coords[0].reshape(dst_h, dst_w)
    src_y = src_coords[1].reshape(dst_h, dst_w)

    # Bilinear interpolation
    x0 = torch.floor(src_x).long()
    x1 = x0 + 1
    y0 = torch.floor(src_y).long()
    y1 = y0 + 1

    valid = (src_x >= 0) & (src_x < src_w - 1) & (src_y >= 0) & (src_y < src_h - 1)

    wa = ((x1 - src_x) * (y1 - src_y))
    wb = ((x1 - src_x) * (src_y - y0))
    wc = ((src_x - x0) * (y1 - src_y))
    wd = ((src_x - x0) * (src_y - y0))

    output = torch.zeros((dst_h, dst_w, channels), dtype=torch.float32, device=DEVICE)

    for c in range(channels):
        img_c = img_t[:, :, c]
        v00 = img_c[y0[valid], x0[valid]]
        v01 = img_c[y1[valid], x0[valid]]
        v10 = img_c[y0[valid], x1[valid]]
        v11 = img_c[y1[valid], x1[valid]]

        output[valid, c] = (v00 * wa[valid] + v01 * wb[valid] +
                            v10 * wc[valid] + v11 * wd[valid])

    return output.byte().cpu().numpy()


def _warp_numpy(img, M, dst_w, dst_h, src_w, src_h, channels):
    M_inv = np.linalg.inv(M)

    x, y = np.meshgrid(np.arange(dst_w, dtype=np.float32),
                        np.arange(dst_h, dtype=np.float32))
    ones = np.ones_like(x)
    dst_coords = np.stack([x, y, ones], axis=-1).reshape(-1, 3).T

    src_coords = M_inv @ dst_coords
    src_coords /= src_coords[2, :]

    src_x = src_coords[0, :].reshape(dst_h, dst_w)
    src_y = src_coords[1, :].reshape(dst_h, dst_w)

    x0 = np.floor(src_x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(np.int32)
    y1 = y0 + 1

    mask = (src_x >= 0) & (src_x < src_w - 1) & (src_y >= 0) & (src_y < src_h - 1)
    output = np.zeros((dst_h, dst_w, channels), dtype=img.dtype)

    wa = (x1 - src_x) * (y1 - src_y)
    wb = (x1 - src_x) * (src_y - y0)
    wc = (src_x - x0) * (y1 - src_y)
    wd = (src_x - x0) * (src_y - y0)

    for c in range(channels):
        img_c = img[:, :, c]
        v00 = img_c[y0[mask], x0[mask]]
        v01 = img_c[y1[mask], x0[mask]]
        v10 = img_c[y0[mask], x1[mask]]
        v11 = img_c[y1[mask], x1[mask]]
        output[mask, c] = (v00 * wa[mask] + v01 * wb[mask] +
                           v10 * wc[mask] + v11 * wd[mask]).astype(img.dtype)

    return output