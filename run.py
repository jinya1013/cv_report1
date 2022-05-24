from email.policy import default
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from filter import bilateral_filter, guided_filter, non_local_means_filter
from score import psnr

def run_exp(imgs, noises, bilateral_iters=10, guided_eps=5, nlm_sigma=10, nlm_h=5):
    """
    画像のリストimgsおよびノイズのリストnoisesに対して, Bilateral Filter, Guided Filter, NLM Filterをかけてその性能を比較する関数.

    Parameters
    --------------------------
    imgs: list( ndarray [in_h, in_w, C] ([in_h, in_w]) )
        入力RGB画像データのリスト
    noises: list( float )
        画像データに対して加えるノイズの標準偏差のリスト
    bilateral_iters: int
        Bilateral Filterを作用させる回数
    guided_eps: float
        Guided Filterの正則化項の係数
    nlm_sigma: float
        NLM Filterで画素どうしの輝度距離を計算するときのパラメタ
    nlm_h: float
        NLM Filterで画素どうしの輝度距離を計算するときのパラメタ

    Returns
    --------------------------
    score: ndarray [len(imgs), len(noises), 4]
        各フィルタpsnr値
    filtered: list( ndarray [len(noises), in_h, in_w, 3] )
    """
    score = []
    filtered = []
    for img in tqdm(imgs):
        score.append([])
        filtered.append([])
        for noise in tqdm(noises):
            score[-1].append([])
            filtered[-1].append([])
            noised = img + np.random.normal(loc=0, scale=noise, size=img.shape)

            score[-1][-1].append(psnr(img, noised))
            filtered[-1][-1].append(noised)

            # Bilateral Filter
            bilateral_filtered = np.copy(noised) 
            for _ in range(bilateral_iters):
                bilateral_filtered = bilateral_filter(bilateral_filtered)
            score[-1][-1].append(psnr(img, bilateral_filtered))
            filtered[-1][-1].append(bilateral_filtered)

            # Guided Filter
            guided_filtered = guided_filter(noised, eps=guided_eps)
            score[-1][-1].append(psnr(img, guided_filtered))
            filtered[-1][-1].append(guided_filtered)

            # NLM Filter
            nlm_filtered = non_local_means_filter(noised, sigma=nlm_sigma, h=nlm_h)
            score[-1][-1].append(psnr(img, nlm_filtered))
            filtered[-1][-1].append(nlm_filtered)

    return np.array(score), filtered

def run_exp(imgs, noises, bilateral_dict={"iters":5, "flt_size":3, "sigma_d":1, "sigma_r":1},
            guided_dict={"guide":None, "flt_size":3, "eps":5.0},
            nlm_dict={"flt_size":3, "sigma":10.0, "h":5.0}, padding=True):
    """
    画像のリストimgsおよびノイズのリストnoisesに対して, Bilateral Filter, Guided Filter, NLM Filterをかけてその性能を比較する関数.

    Parameters
    --------------------------
    imgs: list( ndarray [in_h, in_w, C] ([in_h, in_w]) )
        入力RGB画像データのリスト
    noises: list( float )
        画像データに対して加えるノイズの標準偏差のリスト
    bilateral_dict: int
        Bilateral Filterのパラメタ
    guided_dict: dict
        Guided Filterのパラメタ
    nlm_dict: dict
        NLM Filterのパラメタ

    Returns
    --------------------------
    score: ndarray [len(imgs), len(noises), 4]
        各フィルタpsnr値
    filtered: list( ndarray [len(noises), in_h, in_w, 3] )
    """




    score = []
    filtered = []
    for img in tqdm(imgs):
        score.append([])
        filtered.append([])
        for noise in tqdm(noises):
            score[-1].append([])
            filtered[-1].append([])
            noised = img + np.random.normal(loc=0, scale=noise, size=img.shape)

            score[-1][-1].append(psnr(img, noised))
            filtered[-1][-1].append(noised)

            # Bilateral Filter
            bilateral_filtered = np.copy(noised) 
            for _ in range(bilateral_dict["iters"]):
                bilateral_filtered = bilateral_filter(bilateral_filtered, bilateral_dict["flt_size"], bilateral_dict["sigma_d"], bilateral_dict["sigma_r"], padding)
            score[-1][-1].append(psnr(img, bilateral_filtered))
            filtered[-1][-1].append(bilateral_filtered)

            # Guided Filter
            guided_filtered = guided_filter(noised, guided_dict["guide"], guided_dict["flt_size"], guided_dict["eps"], padding)
            score[-1][-1].append(psnr(img, guided_filtered))
            filtered[-1][-1].append(guided_filtered)

            # NLM Filter
            nlm_filtered = non_local_means_filter(noised, nlm_dict["flt_size"], nlm_dict["sigma"], nlm_dict["h"], padding)
            score[-1][-1].append(psnr(img, nlm_filtered))
            filtered[-1][-1].append(nlm_filtered)

    return np.array(score), filtered

if __name__ == "__main__":

    import os
    # ディレクトリがない場合、作成する
    if not os.path.exists("./outputs"):
        print("ディレクトリを作成します")
        os.makedirs("./outputs")

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}) #桁を揃える

    import argparse

    bilateral_dict = {"iters":10, "flt_size":3, "sigma_d":1, "sigma_r":1}
    guided_dict = {"guide":None, "flt_size":3, "eps":5.0}
    nlm_dict = {"flt_size":3, "sigma":10.0, "h":5}

    parser = argparse.ArgumentParser()

    parser.add_argument("--b_iters", type=int, default=5)
    parser.add_argument("--b_fsize", type=int, default=3)
    parser.add_argument("--b_sigmad", type=float, default=1.0)
    parser.add_argument("--b_sigmar", type=float, default=1.0)
    parser.add_argument("--g_fsize", type=int, default=3)
    parser.add_argument("--g_eps", type=float, default=5.0)
    parser.add_argument("--n_fsize", type=int, default=3)
    parser.add_argument("--n_sigma", type=float, default=10.0)
    parser.add_argument("--n_h", type=float, default=5.0)
    parser.add_argument("--not_padding", action='store_false')


    args = parser.parse_args() 
    bilateral_dict["iters"] = args.b_iters
    bilateral_dict["flt_size"] = args.b_fsize
    bilateral_dict["sigma_d"] = args.b_sigmad
    bilateral_dict["sigma_r"] = args.b_sigmar
    guided_dict["flt_size"] = args.g_fsize
    guided_dict["eps"] = args.g_eps
    nlm_dict["flt_size"] = args.n_fsize
    nlm_dict["sigma"] = args.n_sigma
    nlm_dict["h"] = args.n_h
    padding = args.not_padding


    ramen = np.array(Image.open("/content/drive/MyDrive/CV/img/ramen.jpeg"), dtype="float32")
    ramen = ramen / 255

    pudding = np.array(Image.open("/content/drive/MyDrive/CV/img/pudding.jpeg"), dtype="float32")
    pudding = pudding / 255

    onsen = np.array(Image.open("/content/drive/MyDrive/CV/img/onsen.jpeg"), dtype="float32")
    onsen = onsen / 255

    psnr, filtered = run_exp([ramen, pudding, onsen], [0.01, 0.05, 0.1, 0.2, 0.5], bilateral_dict, guided_dict, nlm_dict, padding)

    ramens = np.array(filtered[0])
    title = ["No Filter", "Bilateral FIlter", "Guided Filter", "NLM FIlter"]
    fig = plt.figure(figsize=(36, 24))
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1)
        ax.imshow(ramens[2][i].transpose(1, 0, 2)[::-1, ::-1, :]) #画像の向きがなぜか90度傾いていたので修正
        ax.set_title(title[i], {"fontsize": 60})
        ax.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    fig.savefig("./outputs/ramens.png")

    puddings = np.array(filtered[1])
    title = ["No Filter", "Bilateral FIlter", "Guided Filter", "NLM FIlter"]
    fig = plt.figure(figsize=(36, 24))
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        ax.imshow(puddings[3][i])
        ax.set_title(title[i], {"fontsize": 60})
        ax.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    fig.savefig("./outputs/puddings.png")

    onsens = np.array(filtered[2])
    title = ["No Filter", "Bilateral FIlter", "Guided Filter", "NLM FIlter"]
    fig = plt.figure(figsize=(36, 24))
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1)
        ax.imshow(onsens[4][i])
        ax.set_title(title[i], {"fontsize": 60})
        ax.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    fig.savefig("./outputs/onsens.png")

    print("\n")
    print("ramen")
    print("No Noise | Bilateral | Guided | NLM")
    print("______________________________")
    print(psnr[0])

    print("pudding")
    print("No Noise | Bilateral | Guided | NLM")
    print("______________________________")
    print(psnr[1])

    print("onsen")
    print("No Noise | Bilateral | Guided | NLM")
    print("______________________________")
    print(psnr[2])




