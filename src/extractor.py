import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
from tqdm import tqdm
import os
import gc
from . import config  # 引用同目录下的 config.py

class FeatureExtractor:
    def __init__(self):
        # 初始化时，预先计算好所有卫星的 FFT 码，存入内存
        self.fft_code_matrix = self._precompute_codes()

    def _precompute_codes(self):
        """ 预计算 C/A 码 FFT 矩阵 (只运行一次，极大提升后续速度) """
        print(f"⚡️ [Init] 预生成 C/A 码库 (FS={config.WORK_FS/1e6:.2f}MHz)...")
        
        # 计算工作采样率下的采样点数
        samples_per_ms = int(config.WORK_FS / 1000)
        total_samples = samples_per_ms * config.INT_TIME_MS
        
        # 完整的 32 颗卫星 G2 Taps 表
        g2_taps_dict = {
            1:[2,6], 2:[3,7], 3:[4,8], 4:[5,9], 5:[1,9], 6:[2,10], 7:[1,8], 8:[2,9],
            9:[3,10], 10:[2,3], 11:[3,4], 12:[5,6], 13:[6,7], 14:[7,8], 15:[8,9], 16:[9,10],
            17:[1,4], 18:[2,5], 19:[3,6], 20:[4,7], 21:[5,8], 22:[6,9], 23:[1,3], 24:[4,6],
            25:[5,7], 26:[6,8], 27:[7,9], 28:[8,10], 29:[1,6], 30:[2,7], 31:[3,8], 32:[4,9]
        }
        
        # 创建一个空矩阵 [32, N]
        code_matrix = np.zeros((32, total_samples), dtype=np.float32)
        
        # 时间轴与重采样索引
        ts = 1.0 / config.WORK_FS
        tc = 1.0 / 1.023e6
        # 生成 1ms 的索引模板
        idx_1ms = np.ceil(ts * np.arange(1, samples_per_ms + 1) / tc).astype(int) - 1
        idx_1ms = np.clip(idx_1ms, 0, 1022)

        # 循环生成 32 颗卫星的码
        for prn in range(1, 33):
            taps = g2_taps_dict.get(prn, [2,6])
            g1 = np.ones(10, dtype=int); g2 = np.ones(10, dtype=int)
            code = []
            for _ in range(1023):
                out = (g1[9] + (g2[taps[0]-1] + g2[taps[1]-1]) % 2) % 2
                code.append(out)
                new_g1 = (g1[2]+g1[9])%2; g1=np.roll(g1,1); g1[0]=new_g1
                new_g2 = (g2[1]+g2[2]+g2[5]+g2[7]+g2[8]+g2[9])%2; g2=np.roll(g2,1); g2[0]=new_g2
            
            # 0/1 转 -1/1
            raw_code = np.array(code) * 2 - 1
            # 重采样并重复 INT_TIME_MS 次
            full_code = np.tile(raw_code[idx_1ms], config.INT_TIME_MS)
            
            # 长度强制对齐
            if len(full_code) > total_samples: full_code = full_code[:total_samples]
            elif len(full_code) < total_samples: full_code = np.pad(full_code, (0, total_samples-len(full_code)))
            code_matrix[prn-1, :] = full_code

        # 直接返回频域共轭矩阵，方便后续做相关运算
        return fft(code_matrix, axis=1).conj()

    def process_single_file(self, filename):
        """ 主处理函数: 读取 bin -> 运算 -> 生成 csv """
        # 拼接完整路径
        raw_path = os.path.join(config.DATA_RAW_DIR, filename)
        # 输出文件名 (例如 ds4.bin -> ds4_features.csv)
        out_csv = os.path.join(config.DATA_PROC_DIR, filename.replace('.bin', '_features.csv'))
        
        # 检查文件是否存在
        if not os.path.exists(raw_path):
            print(f"⚠️ 文件缺失: {filename} (请检查 data/raw/ 文件夹)")
            return None

        # 如果已经跑过了，就跳过（省时间）
        if os.path.exists(out_csv):
            print(f"✅ 已存在: {out_csv} (跳过提取)")
            return out_csv

        print(f"🚀 [Extracting] 正在处理: {filename} ...")
        
        # === 修复 BUG 的区域 ===
        # 1. 计算原始数据(25MHz)每次要读多少个点
        # 公式: 25000 samples/ms * 4ms = 100,000 samples
        raw_chunk_samples = int(config.FS / 1000 * config.INT_TIME_MS) 
        
        # 2. 计算降采样后(Work FS)用于计算的长度
        work_samples = self.fft_code_matrix.shape[1]
        
        # 3. CN0 的积分增益补偿 (10*log10(1/0.004) ≈ 24dB)
        cn0_gain = 10 * np.log10(1 / (config.INT_TIME_MS / 1000.0))
        
        # 4. 计算步长: 每 0.5 秒读一次
        # 采样率 * 0.5秒 * (I+Q 2路) * (int16 2字节) = 字节数
        step_bytes = int(0.5 * config.FS * 2 * 2) 
        
        file_size = os.path.getsize(raw_path)
        total_snaps = file_size // step_bytes
        
        results = []
        
        # 打开二进制文件
        with open(raw_path, 'rb') as f:
            # 进度条
            for i in tqdm(range(total_snaps), desc=filename, unit="snap"):
                # 跳至指定位置
                f.seek(i * step_bytes)
                
                # 读取 raw_chunk_samples 个复数点 (所以 count * 2)
                raw = np.fromfile(f, dtype=np.int16, count=raw_chunk_samples * 2)
                
                if len(raw) < raw_chunk_samples * 2: break
                
                # I/Q 复数化 + 降采样 (DECIMATION=6)
                if len(raw) % 2 != 0: raw = raw[:-1]
                sig = (raw[0::2] + 1j * raw[1::2])[::config.DECIMATION]
                
                # 长度再次强制对齐 (防止降采样除不尽)
                if len(sig) > work_samples: sig = sig[:work_samples]
                elif len(sig) < work_samples: sig = np.pad(sig, (0, work_samples-len(sig)))
                
                # === Turbo Engine Core (矩阵运算核心) ===
                N = len(sig)
                ts_vec = np.arange(N) * (1.0/config.WORK_FS)
                best_peaks = np.zeros(32)
                best_noise = np.zeros(32)
                best_dopplers = np.zeros(32)

                # 循环多普勒频点
                for doppler in range(-config.SEARCH_BAND, config.SEARCH_BAND, config.STEP_HZ):
                    # 载波剥离
                    baseband = sig * np.exp(-1j * 2 * np.pi * doppler * ts_vec)
                    # 并行相关: 一次算出32颗星
                    corr = np.abs(ifft(fft(baseband) * self.fft_code_matrix, axis=1)) ** 2
                    
                    # 提取每颗星的峰值
                    peaks = np.max(corr, axis=1)
                    
                    # 更新最佳结果
                    updates = peaks > best_peaks
                    if np.any(updates):
                        for idx in np.where(updates)[0]:
                            best_peaks[idx] = peaks[idx]
                            best_dopplers[idx] = doppler
                            # 粗略计算噪声底 (剔除峰值后的均值)
                            best_noise[idx] = (np.sum(corr[idx]) - peaks[idx]) / (N-1)

                # === 结果筛选与保存 ===
                for prn in range(32):
                    if best_noise[prn] > 0:
                        # 计算线性信噪比
                        snr = (best_peaks[prn] - best_noise[prn]) / best_noise[prn]
                        if snr > 0:
                            # 转换为 dB-Hz
                            cn0 = 10 * np.log10(snr) + cn0_gain
                            
                            # 阈值判定 (只保留有效卫星)
                            if cn0 > config.ACQ_THRESHOLD:
                                results.append({
                                    "Time": round(i * 0.5, 2),
                                    "PRN": prn + 1,
                                    "CN0_dBHz": round(cn0, 2),
                                    "Doppler": int(best_dopplers[prn])
                                })
                
                # 定期清理内存
                if i % 100 == 0: gc.collect()

        # 保存为 CSV
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        return out_csv