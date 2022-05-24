## コンピュータビジョンレポート1

- 実行方法
`$ python run.py`

    `./outputs`にノイズ除去画像が生成されていればOK

- コマンドライン引数

    `"--b_iters", type=int, default=5, help="bilateral filterの作用回数"`

    `"--b_fsize", type=int, default=3, help="bilateral filterのフィルタサイズ"`

    `"--b_sigmad", type=float, default=1.0, help="bilateral filterの距離分散パラメタ"`

    `"--b_sigmar", type=float, default=1.0, help="bilateral filterの輝度分散パラメタ"`

    `"--g_fsize", type=int, default=3, help="guided filterのフィルタサイズ"`

    `"--g_eps", type=float, default=5.0, help="guided filterの正則化パラメタ"`

    `"--n_fsize", type=int, default=3, help="guided filterのフィルタサイズ"`

    `"--n_sigma", type=float, default=10.0, help="nlm filterの分散パラメタ"`

    `"--n_h", type=float, default=5.0, help="nlm filterのhパラメタ"`
