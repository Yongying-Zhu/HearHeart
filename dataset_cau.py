import torch
import torchaudio
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from torch.utils.data import Dataset
import random

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
    def multi_spectro_gram(aud, n_mels=128, n_fft=512, hop_len=256):
        """生成三种时频表示：Mel + STFT + CQT"""
        sig, sr = aud
        sig_np = sig.numpy()[0]
        
        # 明确指定hop_length（修复None问题）
        if hop_len is None:
            hop_len = n_fft // 4  # 默认值
        
        # 1. Mel频谱图
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels
        )(sig)
        mel_spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spec)
        
        # 2. STFT频谱图
        stft = librosa.stft(sig_np, n_fft=n_fft, hop_length=hop_len)
        stft_mag = np.abs(stft)
        stft_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
        if stft_db.shape[0] != n_mels:
            from scipy.ndimage import zoom
            zoom_factor = n_mels / stft_db.shape[0]
            stft_db = zoom(stft_db, (zoom_factor, 1), order=1)
        stft_tensor = torch.tensor(stft_db, dtype=torch.float32).unsqueeze(0)
        
        # 3. CQT（修复：明确hop_length，n_bins=72）
        cqt = librosa.cqt(sig_np, sr=sr, hop_length=hop_len, 
                          bins_per_octave=12, n_bins=72, fmin=25)
        cqt_mag = np.abs(cqt)
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
        if cqt_db.shape[0] != n_mels:
            from scipy.ndimage import zoom
            zoom_factor = n_mels / cqt_db.shape[0]
            cqt_db = zoom(cqt_db, (zoom_factor, 1), order=1)
        cqt_tensor = torch.tensor(cqt_db, dtype=torch.float32).unsqueeze(0)
        
        # 时间维度对齐
        min_time = min(mel_spec.shape[2], stft_tensor.shape[2], cqt_tensor.shape[2])
        mel_spec = mel_spec[:, :, :min_time]
        stft_tensor = stft_tensor[:, :, :min_time]
        cqt_tensor = cqt_tensor[:, :, :min_time]
        
        # 拼接成3通道
        multi_spec = torch.cat([mel_spec, stft_tensor, cqt_tensor], dim=0)
        
        return multi_spec

    @staticmethod
    def cutmix(spec, label, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        _, H, W = spec.shape
        cut_h = int(H * np.sqrt(1 - lam))
        cut_w = int(W * np.sqrt(1 - lam))
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        spec_aug = spec.clone()
        spec_aug[:, y1:y2, x1:x2] = 0
        
        return spec_aug, label

    @staticmethod
    def mixup(spec, label, alpha=0.4):
        lam = np.random.beta(alpha, alpha)
        noise = torch.randn_like(spec) * 0.1
        spec_mixed = lam * spec + (1 - lam) * noise
        return spec_mixed, label

    @staticmethod
    def cutout(spec, n_holes=3, length=20):
        _, H, W = spec.shape
        spec_aug = spec.clone()
        
        for _ in range(n_holes):
            y = np.random.randint(H)
            x = np.random.randint(W)
            
            y1 = np.clip(y - length // 2, 0, H)
            y2 = np.clip(y + length // 2, 0, H)
            x1 = np.clip(x - length // 2, 0, W)
            x2 = np.clip(x + length // 2, 0, W)
            
            spec_aug[:, y1:y2, x1:x2] = spec.mean()
        
        return spec_aug

class SoundDS_CAU(Dataset):
    def __init__(self, df, data_path, mode='train', df_wide=None):
        self.df = df
        self.data_path = data_path
        self.mode = mode
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
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        
        # 明确指定hop_len=256
        multi_sgram = AudioUtil.multi_spectro_gram(dur_aud, n_mels=128, n_fft=512, hop_len=256)
        
        if self.mode == 'train':
            rand = random.random()
            if rand < 0.3:
                multi_sgram, class_id = AudioUtil.cutmix(multi_sgram, class_id)
            elif rand < 0.6:
                multi_sgram, class_id = AudioUtil.mixup(multi_sgram, class_id)
            elif rand < 0.9:
                multi_sgram = AudioUtil.cutout(multi_sgram)
        
        wide_features = torch.tensor(self.df_wide.iloc[idx].values, dtype=torch.float32)
        
        return multi_sgram, wide_features, class_id

class SoundDS_Patch_CAU(Dataset):
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
            multi_sgram = AudioUtil.multi_spectro_gram((patch, sr), n_mels=128, n_fft=512, hop_len=256)
            patches.append(multi_sgram)
        
        patches_tensor = torch.stack(patches)
        wide_features = torch.tensor(self.df_wide.iloc[idx].values, dtype=torch.float32)
        
        return patches_tensor, wide_features, class_id
