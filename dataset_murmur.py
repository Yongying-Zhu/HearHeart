import torch
import torchaudio
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from torch.utils.data import Dataset

class AudioUtil:
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if sr != newsr:
            resig = torchaudio.transforms.Resample(sr, newsr)(sig)
            return resig, newsr
        return sig, sr

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms
        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = (max_len - sig_len) // 2
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return sig, sr

    @staticmethod
    def simple_hmm_segment(sig, sr):
        """简化版HMM分割：基于包络检测模拟心动周期"""
        # 计算包络
        sig_np = sig.numpy()[0]
        analytic_signal = signal.hilbert(sig_np)
        envelope = np.abs(analytic_signal)
        
        # 平滑包络
        window_size = int(0.05 * sr)  # 50ms窗口
        envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # 检测峰值（S1/S2候选）
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(envelope_smooth, distance=int(0.2*sr), prominence=0.3*np.max(envelope_smooth))
        
        # 如果峰值太少，返回原始信号
        if len(peaks) < 2:
            return [sig]
        
        # 按心动周期切分（每两个峰之间为一个周期）
        segments = []
        for i in range(len(peaks)-1):
            start = peaks[i]
            end = peaks[i+1]
            if end - start > int(0.3 * sr):  # 至少300ms
                segment = sig[:, start:end]
                segments.append(segment)
        
        return segments if len(segments) > 0 else [sig]

    @staticmethod
    def spectro_gram(aud, n_mels=128, n_fft=512, hop_len=None):
        sig, sr = aud
        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec)
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec)
        return aug_spec

class SoundDS_Murmur(Dataset):
    """支持HMM分割的数据集"""
    def __init__(self, df, data_path, mode='train', df_wide=None, use_segmentation=True):
        self.df = df
        self.data_path = data_path
        self.mode = mode
        self.duration = 15000
        self.sr = 4000
        self.df_wide = df_wide
        self.use_segmentation = use_segmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.data_path + '/' + self.df.iloc[idx]['relative_path']
        class_id = self.df.iloc[idx]['label']
        
        # 加载音频
        aud = AudioUtil.open(audio_file)
        reaud = AudioUtil.resample(aud, self.sr)
        
        # HMM分割（训练时使用）
        if self.mode == 'train' and self.use_segmentation:
            segments = AudioUtil.simple_hmm_segment(reaud[0], self.sr)
            # 随机选择一个片段
            segment = segments[np.random.randint(len(segments))]
            rechan = (segment, self.sr)
        else:
            rechan = (reaud[0][:1], self.sr)
        
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=128, n_fft=512, hop_len=None)
        
        if self.mode == 'train':
            sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        
        wide_features = torch.tensor(self.df_wide.iloc[idx].values, dtype=torch.float32)
        
        return sgram, wide_features, class_id

class SoundDS_Patch_Murmur(Dataset):
    """验证集：多片段推理"""
    def __init__(self, df, data_path, df_wide=None):
        self.df = df
        self.data_path = data_path
        self.duration = 15000
        self.sr = 4000
        self.df_wide = df_wide

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.data_path + '/' + self.df.iloc[idx]['relative_path']
        class_id = self.df.iloc[idx]['label']
        
        aud = AudioUtil.open(audio_file)
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = (reaud[0][:1], self.sr)
        
        # 滑动窗口
        sig, sr = rechan
        total_len = sig.shape[1]
        patch_len = self.duration * sr // 1000
        hop = patch_len // 2
        
        patches = []
        for start in range(0, total_len, hop):
            end = min(start + patch_len, total_len)
            patch = sig[:, start:end]
            if patch.shape[1] < patch_len:
                pad = torch.zeros((1, patch_len - patch.shape[1]))
                patch = torch.cat([patch, pad], dim=1)
            sgram = AudioUtil.spectro_gram((patch, sr), n_mels=128, n_fft=512)
            patches.append(sgram)
        
        patches_tensor = torch.stack(patches)
        wide_features = torch.tensor(self.df_wide.iloc[idx].values, dtype=torch.float32)
        
        return patches_tensor, wide_features, class_id
