import soundfile as sf, csv, numpy as np
rows = list(csv.DictReader(open('data/manifests/train.csv')))
for r in rows:
    data, sr = sf.read(r['noisy_path'])
    peak = np.max(np.abs(data))
    rms = np.sqrt(np.mean(data**2))
    flag = ''
    if peak > 0.999:
        flag = '<-- POSSIBLE CLIPPING'
    if rms < 0.001:
        flag = '<-- POSSIBLE SILENCE'
    print(r['category'].ljust(15), 'snr=' + str(r['snr_db']).rjust(3), 'peak=%.3f' % peak, 'rms=%.4f' % rms, flag)
