"""Where the acoustic voicing decision and the EGG voicing decision disagree, which one is wrong?

Reads the frames.csv made by train_vad.py (with the egg_to_oq() diagnostics it caches).  The acoustic
predictions come from a logistic regression on amp and cpp, fit by cross-validation grouped by file.
Frames within --margin seconds of an EGG voicing transition are set aside, because there a disagreement
is mostly a question of timing, not of voicing.

Each frame is one of
    agree v    EGG voiced, predicted voiced
    agree uv   EGG unvoiced, predicted unvoiced
    EGG miss?  EGG unvoiced, predicted voiced   (the EGG analysis may have failed on real voicing)
    EGG extra? EGG voiced, predicted unvoiced   (the EGG analysis may have found cycles that are not voicing)

and the report asks, for the disputed frames, what egg_to_oq() found there (the closing and opening instants,
the height of the differentiated EGG, the swing of the EGG signal, and the reason it gave up), how the rate of
EGG misses depends on pitch and on speaker, whether the audio and EGG f0 agree when both are voiced, and
lists the longest disputed stretches to look at in the waveform.

Usage:
    python egg_audit.py frames.csv [--margin 0.03] [--min-ms 60]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vad_continuity import distance_to_transition, runs

warnings.filterwarnings("ignore")
PREDICTORS = ['amp', 'cpp']
DIAG = ['egg_n_gci', 'egg_n_goi', 'egg_degg_max', 'egg_min', 'egg_max', 'egg_range', 'egg_fail']
CLASSES = ['agree v', 'agree uv', 'EGG miss?', 'EGG extra?']


def classify(truth, pred):
    return np.select([truth & pred, ~truth & ~pred, ~truth & pred, truth & ~pred], CLASSES, 'x')


def pct(x):
    return f'{100 * x:.1f}%'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', help='the frames.csv file written by train_vad.py')
    ap.add_argument('--margin', type=float, default=0.03, help='seconds from an EGG transition to set aside')
    ap.add_argument('--min-ms', type=float, default=60, help='shortest disputed stretch to list')
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()

    df = pd.read_csv(a.frames)
    missing = [c for c in DIAG if c not in df.columns]
    if missing:
        raise SystemExit(f'{a.frames} lacks {missing}: remake it with the current train_vad.py')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=PREDICTORS)
    df = df.sort_values(['file', 'sec'], kind='stable').reset_index(drop=True)
    step = float(np.median(np.diff(df['sec'])[np.diff(df['sec']) > 0]))
    truth = (df['vuv'] == 'v').to_numpy()

    X = df[PREDICTORS].to_numpy()
    pred = np.zeros(len(df), bool)
    for tr, te in GroupKFold(a.folds).split(X, truth, df['file']):
        pred[te] = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced')).fit(X[tr], truth[tr]).predict(X[te])

    df['cls'] = classify(truth, pred)
    dist = distance_to_transition(df, truth)
    df['far'] = dist * step >= a.margin
    df['speaker'] = df['file'].str[:4]
    # EGG swing relative to what the same file has when the EGG finds voicing
    typical = df[truth].groupby('file')['egg_range'].median()
    df['rel_range'] = df['egg_range'] / df['file'].map(typical)
    far = df[df.far]

    # ---- A. the four classes
    print(f'{len(df)} frames, {df.file.nunique()} files.  "Far" = at least {a.margin * 1000:.0f} ms from an EGG transition.')
    t = pd.DataFrame({'all frames': df.cls.value_counts(normalize=True), 'far frames': far.cls.value_counts(normalize=True)}).reindex(CLASSES)
    print('\nA. Agreement between the acoustic and EGG voicing decisions (proportion of frames)')
    print(t.fillna(0).map(pct).to_string())

    # ---- B. what egg_to_oq found
    print('\nB. What egg_to_oq() found, by class (far frames; medians)')
    g = far.groupby('cls')
    tab = pd.DataFrame({'frames': g.size(), 'n_gci': g.egg_n_gci.median(), 'n_goi': g.egg_n_goi.median(),
                        'degg_max': g.egg_degg_max.median(), 'egg_min': g.egg_min.median(), 'egg_max': g.egg_max.median(),
                        'rel_range': g.rel_range.median(), 'audio amp': g.amp.median(), 'audio cpp': g.cpp.median(),
                        'audio f0': g.f0.median()}).reindex(CLASSES)
    print(tab.round(2).to_string())

    print('\n   Why the EGG analysis gave no OQ, for the frames it called unvoiced (far frames; share within class)')
    unv = far[~far.cls.isin(['agree v', 'EGG extra?'])]
    print(pd.crosstab(unv.egg_fail, unv.cls, normalize='columns').map(pct).to_string())

    miss = far[far.cls == 'EGG miss?']
    print('\n   EGG-miss frames by the failure reason: how far the signal was from meeting the criterion')
    h = miss.groupby('egg_fail')
    print(pd.DataFrame({'frames': h.size(), 'degg_max < 0.5': h.egg_degg_max.apply(lambda s: (s < 0.5).mean()),
                        'egg_max < 0.43': h.egg_max.apply(lambda s: (s < 0.43).mean()),
                        'egg_min > 0.43': h.egg_min.apply(lambda s: (s > 0.43).mean()),
                        'rel_range < 0.5': h.rel_range.apply(lambda s: (s < 0.5).mean())}).round(2).to_string())

    # ---- C. dependence on pitch and speaker
    print('\nC. Share of acoustically-voiced far frames that the EGG calls unvoiced, by audio f0 (Hz)')
    pv = far[pred[far.index]].copy()
    pv['f0 bin'] = pd.cut(pv.f0, [0, 80, 100, 130, 170, 220, 300, 1000])
    cb = pv.groupby('f0 bin', observed=True).apply(lambda d: pd.Series({'frames': len(d), 'EGG miss rate': (d.cls == 'EGG miss?').mean()}))
    print(cb.round(3).to_string())

    print('\n   By speaker (sorted by miss rate): the share of acoustically-voiced far frames the EGG calls unvoiced,')
    print('   with the commonest failure reason')
    rows = []
    for spk, d in pv.groupby('speaker'):
        m = d[d.cls == 'EGG miss?']
        rows.append({'speaker': spk, 'voiced frames': len(d), 'EGG miss rate': len(m) / len(d),
                     'commonest reason': m.egg_fail.mode().iat[0] if len(m) else '', 'degg_max': m.egg_degg_max.median(),
                     'rel_range': m.rel_range.median()})
    print(pd.DataFrame(rows).sort_values('EGG miss rate', ascending=False).round(3).to_string(index=False))

    # ---- D. EGG extras
    ex = far[far.cls == 'EGG extra?']
    print(f'\nD. EGG extra frames (EGG voiced, acoustically unvoiced): {len(ex)} far frames')
    if len(ex):
        print(ex[['amp', 'cpp', 'f0', 'egg_f0', 'egg_oq', 'rel_range']].describe().loc[['25%', '50%', '75%']].round(2).to_string())

    # ---- E. do the two f0 trackers agree?
    both = far[far.cls == 'agree v']
    r = both.egg_f0 / both.f0
    print(f'\nE. f0 agreement where both say voiced ({len(both)} far frames): EGG f0 / audio f0')
    print(f'   within 10%: {pct(((r > 0.9) & (r < 1.1)).mean())}   near half (0.45-0.55): {pct(((r > 0.45) & (r < 0.55)).mean())}   '
          f'near double (1.8-2.2): {pct(((r > 1.8) & (r < 2.2)).mean())}')
    print('   (audio f0 is from CPP(), whose peak picking can also err, so this is a check on both)')

    # ---- F. a different EGG voicing definition
    print('\nF. The EGG label used so far is "egg_to_oq() produced an OQ", which needs opening instants as well as closing')
    print('   instants.  Alternative: EGG voiced = two or more closing instants in the window (OQ may still be missing).')
    alt = (df['egg_n_gci'] >= 2).to_numpy()
    changed = alt & ~truth
    print(f'   {changed.sum()} frames ({pct(changed.mean())} of all) change from unvoiced to voiced; '
          f'EGG voiced share {pct(truth.mean())} -> {pct(alt.mean())}')
    print(f'   balanced accuracy of the acoustic predictions: {balanced_accuracy_score(truth, pred):.4f} against the OQ label, '
          f'{balanced_accuracy_score(alt, pred):.4f} against the closing-instant label')
    altcls = pd.Series(classify(alt, pred), index=df.index)
    far_alt = pd.crosstab(altcls[df.far], 'far frames', normalize='columns').reindex(CLASSES).fillna(0)
    print('   classes with the closing-instant label (far frames):')
    print(far_alt.map(pct).to_string())

    # ---- G. stretches to look at
    out = []
    for fname, idx in df.groupby('file').indices.items():
        c = (df.cls.to_numpy()[idx] == 'EGG miss?') & df.far.to_numpy()[idx]
        v, l = runs(c)
        starts = np.r_[0, np.flatnonzero(np.diff(c.astype(np.int8))) + 1]
        for s0, val, ln in zip(starts, v, l):
            if val and ln * step * 1000 >= a.min_ms:
                seg = idx[s0:s0 + ln]
                d = df.iloc[seg]
                out.append({'file': fname, 'sec': d.sec.iloc[0], 'ms': ln * step * 1000, 'reason': d.egg_fail.mode().iat[0],
                            'amp': d.amp.mean(), 'cpp': d.cpp.mean(), 'audio f0': d.f0.median(),
                            'degg_max': d.egg_degg_max.median(), 'rel_range': d.rel_range.median()})
    out = pd.DataFrame(out, columns=['file', 'sec', 'ms', 'reason', 'amp', 'cpp', 'audio f0', 'degg_max', 'rel_range'])
    print(f'\nG. {len(out)} EGG-miss stretches of {a.min_ms:.0f} ms or more.  Longest few for each reason, to look at in the waveform and EGG:')
    for reason, d in out.groupby('reason'):
        print(f'\n   {reason}: {len(d)} stretches')
        print(d.sort_values('ms', ascending=False).head(6).round(2).to_string(index=False))
