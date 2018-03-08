import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lombscargle, welch
from scipy.interpolate import splrep, splev
from dateutil import parser
from pytz import timezone

def main():
    df = pd.read_csv('~/Dropbox/export4.csv', names=["time", "hr"])
    df = df.drop(0)
    start_time = df.iloc[0, 0]
    st = int(parser.parse(start_time).timestamp())
    df["elapsed_time"] = [int(parser.parse(t).timestamp()) - st for t in df["time"]]
    df["time"] = [parser.parse(t) for t in df["time"]]
    df["time_jp"] = [t.astimezone(timezone('Asia/Tokyo')) for t in df["time"]]
    df["rri"] = [(60 * 1000 / int(d)) for d in df["hr"]]
    df.set_index("time", inplace=True)

    lf_hf = RRI_2_LFHF(df)
    lf_hf_ls = RRI_2_LFHF_ls(df)
    print("LF/HF = " + str(lf_hf) + " (welch)")
    print("LF/HF = " + str(lf_hf_ls) + " (lombscargle)")

def RRI_2_LFHF_ls(dataset):

    time = dataset["elapsed_time"]
    t = np.array(time)

    rri = dataset["rri"]
    ibi = np.array(rri)

    phi = 4.0 * np.pi

    f = np.linspace(0.001, phi, 1000)

    Pgram = lombscargle(t, ibi, f, normalize=True)

    vlf = 0.04
    lf = 0.15
    hf = 0.4
    Fs = 250

    lf_freq_band = (f >= vlf) & (f <= lf)
    hf_freq_band = (f >= lf) & (f <= hf)

    dy = 1.0 / Fs
    LF = np.trapz(y=abs(Pgram[lf_freq_band]), x=None, dx=dy)
    HF = np.trapz(y=abs(Pgram[hf_freq_band]), x=None, dx=dy)
    LF_HF = float(LF) / HF

    return LF_HF

def RRI_2_LFHF(dataset):
    rri = dataset["rri"]
    ibi = np.array(rri)

    Fxx, Pxx = welch(ibi, fs=4.0, window='hanning', nperseg=256, noverlap=128)

    vlf = 0.04
    lf = 0.15
    hf = 0.4
    Fs = 250

    lf_freq_band = (Fxx >= vlf) & (Fxx <= lf)
    hf_freq_band = (Fxx >= lf) & (Fxx <= hf)

    dy = 1.0 / Fs
    LF = np.trapz(y=abs(Pxx[lf_freq_band]), x=None, dx=dy)
    HF = np.trapz(y=abs(Pxx[hf_freq_band]), x=None, dx=dy)
    LF_HF = float(LF) / HF

    return LF_HF

if __name__ == "__main__":
    main()