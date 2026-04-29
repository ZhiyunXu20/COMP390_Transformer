translate_cli — 终端英->法翻译（checkpoint 须为本仓库 train.py 保存的 best.pt / last.pt）

用法（在仓库根目录执行；checkpoint 相对路径相对于仓库根）:
  python translate_cli/interactive_translate.py -c runs/<run_name>/best.pt --pkg small_try

  small_try / small_head / small_swap / base_improve / base_1 对应训练时使用的代码目录，必须与 checkpoint 一致。

未指定 -c 时默认加载 runs/best.pt。

交互: 直接运行，按提示输入英文；空行退出。
管道: echo 'your sentence .' | python translate_cli/interactive_translate.py -c runs/fast_dot/best.pt --pkg small_try
