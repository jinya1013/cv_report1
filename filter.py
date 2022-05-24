import numpy as np

def im2col(img, flt_shape=3):
    """
    画像imgをフィルタの形状flt_shapeに合わせて, 行列に変換する関数.

    Parameters
    --------------------------
    img: ndarray [in_h, in_w, C]
        入力RGB画像データ
    flt_shape: int, tuple(int, int)
        作用させるフィルタの形状

    Returns
    --------------------------
    col: ndarray [C, flt_h x flt_w, out_h x out_w]
        変換後の行列
    """
    in_c, in_h, in_w = img.shape

    if type(flt_shape) == int:
        flt_h, flt_w = flt_shape, flt_shape
    else:
        flt_h, flt_w = flt_shape

    out_h, out_w = in_h - flt_h + 1, in_w - flt_w + 1

    col = np.zeros((in_c, flt_h, flt_w, out_h, out_w))

    for h in range(flt_h):
        h_max = h + out_h
        for w in range(flt_w):
            w_max = w + out_w
            col[:, h, w, :, :] = img[:, h:h_max, w:w_max]

    col = col.reshape(in_c, flt_h*flt_w, out_h*out_w)

    return col

def col2im(col, out_shape):
    """
    フィルタをかけた後の行列colを形状がout_shapeの画像に戻す関数

    Parameters
    --------------------------
    col: ndarray [C, 1, out_h x out_w]
        フィルタ作用後の行列データ
    out_shape: tuple(int, int)
        出力画像の形状

    Returns
    --------------------------
    col: ndarray [C, flt_h x flt_w, (in_h - flt_h + 1) x (in_w - flt_w + 1)]
        変換後の行列
    """
    
    return col.transpose(2, 0, 1).reshape(out_shape)


def bilateral_filter(img, flt_size=3, sigma_d=1, sigma_r=1, padding=True):
    """
    画像imgにBilateral Filterをかける関数.

    Parameters
    --------------------------
    img: ndarray [in_h, in_w, C] ([in_h, in_w])
        入力RGB画像データ
    flt_size: int
        Bilateralのサイズ, フィルタは正方形でサイズは奇数である必要がある.
    sigma_d: float
        距離依存のガウス項の分散
    sigma_r: float
        輝度依存のガウス項の分散
    padding: bool
        フィルタの形状に合わせたパディングを行うか否か

    Returns
    --------------------------
    filtered: ndarray [out_h, out_w, C]
        フィルタ作用後の画像データ
    """

    if flt_size % 2 == 0:
        raise ValueError("Filter size should be an odd number.\n")

    if img.ndim == 2:
        img = img[:, :, None]

    if padding:
        pad = (flt_size - 1) // 2
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), "constant")

    out_shape = img.shape[0] - flt_size + 1, img.shape[1] - flt_size + 1, img.shape[2]

    # 距離依存のガウスカーネル
    kernel_d = np.empty((flt_size, flt_size))

    center = flt_size // 2
    for i in range(-center, center+1):
        for j in range(-center, center+1):
            kernel_d[center+i][center+j] = np.exp(-(i**2 + j**2)/(2 * sigma_d**2))

    kernel_d = kernel_d.reshape(1, -1, 1)

    # 輝度依存のカーネル
    col = im2col(img.transpose(2, 0, 1), flt_size)
    center = flt_size**2 // 2
    kernel_r = np.exp(-(col[:, center, :][:, None] - col)**2/(2 * sigma_r**2))

    # 2つを掛け合わせてカーネルを作る
    kernel = kernel_d * kernel_r

    # フィルタをかける
    filtered_col = np.sum(col * kernel, axis=1, keepdims=True) / np.sum(kernel, axis=1, keepdims=True)

    filtered_img = col2im(filtered_col, out_shape)

    return filtered_img

def guided_filter(img, guide=None, flt_size=3, eps=5.0, padding=True):
    """
    画像imgにGuided Filterをかける関数.

    Parameters
    --------------------------
    img: ndarray [in_h, in_w, C] ([in_h, in_w])
        入力RGB画像データ
    guide: ndarray [in_h, in_w, C]
        ガイドRGB画像データ, guide=None のときはガイド画像として入力画像imgを用いる.   
    flt_size: int
        Guided Filterのサイズ.
    eps: float
        正則化の係数
    padding: bool
        フィルタの形状に合わせたパディングを行うか否か

    Returns
    --------------------------
    filtered: ndarray [out_h, out_w, C]
        フィルタ作用後の画像データ
    """

    if guide is None:
        guide = np.copy(img)

    if img.ndim == 2:
        img = img[:, :, None]

    if guide.ndim == 2:
        guide = guide[:, :, None]

    if padding:
        pad = (flt_size - 1) // 2
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), "constant")
        guide = np.pad(guide, ((pad, pad), (pad, pad), (0, 0)), "constant")


    out_shape = img.shape[0] - flt_size + 1, img.shape[1] - flt_size + 1, img.shape[2]

    center = flt_size**2 // 2

    guide_col = im2col(guide.transpose(2, 0, 1), flt_size)
    guide_mu = np.mean(guide_col, axis=1, keepdims=True)
    guide_sigma = np.mean((guide_col - guide_mu)**2, axis=1, keepdims=True)

    col = im2col(img.transpose(2, 0, 1), flt_size)
    mu = np.mean(col, axis=1, keepdims=True)

    a = np.mean(guide_col*col - guide_mu*mu, axis=1, keepdims=True) / (guide_sigma**2 + eps)
    b = mu - a * guide_mu
    q = np.mean(a, axis=1, keepdims=True) * guide_col[:, center, None] + np.mean(b, axis=1, keepdims=True)

    return col2im(q, out_shape)

def non_local_means_filter(img, flt_size=3, sigma=1, h=0.6, padding=True):
    """
    画像imgにNLM Filterをかける関数.

    Parameters
    --------------------------
    img: ndarray [in_h, in_w, C] ([in_h, in_w])
        入力RGB画像データ
    flt_size: int
        NLM Filterのサイズ.
    sigma: float
        画素どうしの輝度距離を計算するときのパラメタ
    h: float
        画素どうしの輝度距離を計算するときのパラメタ
    padding: bool
        フィルタの形状に合わせたパディングを行うか否か

    Returns
    --------------------------
    filtered: ndarray [out_h, out_w, C]
        フィルタ作用後の画像データ
    """
    if img.ndim == 2:
        img = img[:, :, None]

    img_shape = img.shape

    if padding:
        pad_ud = flt_size - 1
        img_ud = np.pad(img, ((pad_ud, pad_ud), (pad_ud, pad_ud), (0, 0)), "constant")
        pad = (flt_size - 1) // 2
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), "constant")

    col = im2col(img.transpose(2, 0, 1), (flt_size, flt_size))

    center = flt_size**2 // 2
    r = flt_size // 2

    col_ud = im2col(img_ud.transpose(2, 0, 1), (flt_size, flt_size))

    d2 = np.zeros_like(col)

    for y in range(r+1, r+img_shape[1]+1):
        for x in range(r+1, r+img_shape[0]+1):
            ul = img_ud[(x-1)-r:(x-1)+r+1, (y-1)-r:(y-1)+r+1]
            u = img_ud[x-r:x+r+1, (y-1)-r:(y-1)+r+1]
            ur = img_ud[(x+1)-r:(x+1)+r+1, (y-1)-r:(y-1)+r+1]
            cl = img_ud[(x-1)-r:(x-1)+r+1, y-r:y+r+1]
            c = img_ud[x-r:x+r+1, y-r:y+r+1]
            cr = img_ud[(x+1)-r:(x+1)+r+1, y-r:y+r+1]
            ll = img_ud[(x-1)-r:(x-1)+r+1, (y+1)-r:(y+1)+r+1]
            l = img_ud[x-r:x+r+1, (y+1)-r:(y+1)+r+1]
            lr = img_ud[(x+1)-r:(x+1)+r+1, (y+1)-r:(y+1)+r+1]

            q = np.stack([ul, u, ur, cl, c, cr, ll, l, lr])

            d2[:, :, (y-r-1)*img_shape[0]+x-r-1] = np.mean((q - c[None])**2, axis=(1, 2, 3))


    kernel = np.exp(-(np.maximum(d2 - sigma**2, 0) / h**2))
    
    filtered_col = np.mean(kernel * col, axis=1, keepdims=True) / np.mean(kernel, axis=1, keepdims=True)
    
    return col2im(filtered_col, img_shape)