import numpy as np
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Resident Set Size: the non-swapped physical memory the process is using

n_sample = 25689
max_length_x = 128
max_length_t = 130
n_char = 1782

# メモリ使用量測定前
t1 = get_memory_usage()
print(t1/ (1024 * 1024))
# 配列作成
x_encoder = np.zeros((n_sample, max_length_x, n_char), dtype=np.bool_)  # encoderへの入力
x_decoder = np.zeros((n_sample, max_length_t, n_char), dtype=np.bool_)  # decoderへの入力
t_decoder = np.zeros((n_sample, max_length_t, n_char), dtype=np.bool_)  # decoderの正解

# メモリ使用量測定後
t2 = get_memory_usage()
print(t2/ (1024 * 1024))

# メモリ消費の差を出力
print(f"Memory used: {t2 - t1} bytes")
