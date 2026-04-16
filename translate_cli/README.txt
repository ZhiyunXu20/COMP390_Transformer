translate_cli — 终端英->法翻译

用法:
  python interactive_translate.py -c /root/autodl-tmp/small_try/runs/fast_dot/best.pt

交互: 直接运行，按提示输入英文；空行退出。
管道: echo 'your sentence .' | python interactive_translate.py -c <checkpoint>

checkpoint 须为本仓库 train.py 保存的 best.pt / last.pt，且路径中需包含
small_try、small_head、small_swap、base_improve 或 base_1 以便自动加载对应 model/config。
