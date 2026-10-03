"""Train the coefficients of phon.VAD() against EGG voicing.

VAD() predicts voicing from two measurements: `amp`, the amplitude envelope level in a low frequency
band (AMP_BAND, 80-800 Hz) and `cpp`, the cepstral peak prominence.  For each wav file in the ASC corpus
(channel 0 = audio, channel 1 = EGG) this measures both at the frame times of egg_to_oq(), takes the
"true" voicing state from the `voiced` column of egg_to_oq() (two glottal closing instants one period apart;
EGG_OPTIONS below), and fits a ridge (L2) logistic regression.  The ridge penalty is chosen,
and the accuracy reported, by cross-validation grouped by file, so that frames from one recording are never
in both the training and the test set.

Usage:
    python train_vad.py /Volumes/WDBook/Corpora/asc/corpus [--frac 0.25] [--seed 1] [--cache frames.csv]

The cache also holds the EGG's f0 (egg_f0), open quotient (egg_oq) and the egg_to_oq() diagnostics
(egg_n_gci, egg_n_goi, egg_degg_max, egg_min, egg_max, egg_range, egg_fail), for examining the EGG decisions.
Paste the printed VAD_COEFS line into phonlab/acoustic/vad.py.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fileorganize import dir_to_df
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phonlab as phon
from phonlab.acoustic.vad import vad_features

warnings.filterwarnings("ignore")

F0_RANGE = [60, 400]
# How the EGG voicing decisions are made (see egg_to_oq()): the range normalization of the differentiated EGG
# uses a window of 0.3 seconds centered on each sample, and finds nothing where the local range is less than
# 5% of the file's largest.
EGG_OPTIONS = dict(norm_window=0.3, center=True, floor=0.05)
PREDICTORS = ['amp', 'cpp']


def collect(corpus, frac, seed):
    """Measure the predictors and the EGG voicing in a random fraction of the corpus' files."""
    rng = np.random.default_rng(seed)
    tgdf = dir_to_df(Path(corpus), fnpat=r"(?P<gender>[mf]).+.TextGrid$", addcols=['barename'])
    print(f'Found {len(tgdf)} TextGrid files.')
    frames = []
    for row in tgdf.itertuples():
        if rng.random() > frac:
            continue
        audio, egg, fs = phon.loadsig(str(Path(corpus) / row.relpath / f'{row.barename}.wav'), chansel=[0, 1])
        eggdf = phon.egg_to_oq(egg, fs, f0_range=F0_RANGE, diagnostics=True, **EGG_OPTIONS)
        df = vad_features(audio, fs, sec=eggdf['sec'].to_numpy(), f0_range=F0_RANGE)   # same times as the EGG
        df['vuv'] = np.where(eggdf['voiced'].to_numpy(), 'v', 'uv')
        df['egg_f0'] = eggdf['f0'].to_numpy()     # kept to examine the EGG decisions, not used in training
        df['egg_oq'] = eggdf['OQ'].to_numpy()
        for c in ['n_gci', 'n_goi', 'degg_max', 'egg_min', 'egg_max', 'egg_range', 'fail']:
            df[c if c.startswith('egg') else f'egg_{c}'] = eggdf[c].to_numpy()  # what egg_to_oq() found in each frame, and why it failed
        df['file'] = row.barename
        frames.append(df)
        print(f'{row.relpath}/{row.barename}: {len(df)} frames', flush=True)
    return pd.concat(frames, ignore_index=True)


def fit(df, folds):
    X, y, groups = df[PREDICTORS].to_numpy(), (df['vuv'] == 'v').to_numpy(), df['file'].to_numpy()
    splits = list(GroupKFold(n_splits=folds).split(X, y, groups))

    # ridge penalty, chosen by grouped cross-validation (predictors are standardized so the penalty is even-handed)
    cv = LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=splits, scoring='roc_auc',
                              class_weight='balanced', max_iter=1000)
    cv.fit(StandardScaler().fit_transform(X), y)
    C = float(cv.C_[0])

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=1000))

    auc, bacc = [], []
    for tr, te in splits:
        m = model().fit(X[tr], y[tr])
        auc.append(roc_auc_score(y[te], m.decision_function(X[te])))
        bacc.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    print(f'\nRidge penalty C = {C:.4g}')
    print(f'Held-out (by file) over {folds} folds:  AUC {np.mean(auc):.4f}   balanced accuracy {np.mean(bacc):.4f}')

    m = model().fit(X, y)
    pred = m.predict(X)
    print('\nAll frames (proportion of EGG voiced / unvoiced frames, by prediction):')
    print(pd.crosstab(pd.Series(np.where(pred, 'pred v', 'pred uv'), name=''),
                      pd.Series(np.where(y, 'v', 'uv'), name='EGG'), normalize='columns').round(3))

    sc, lm = m[0], m[1]       # convert the standardized coefficients back to raw units
    w = lm.coef_[0] / sc.scale_
    b = float(lm.intercept_[0] - np.sum(lm.coef_[0] * sc.mean_ / sc.scale_))
    coefs = {'intercept': round(b, 3), **{p: round(float(v), 4) for p, v in zip(PREDICTORS, w)}}
    print('\nVAD_COEFS = ' + str(coefs))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('corpus', help='root directory of the ASC corpus')
    ap.add_argument('--frac', type=float, default=0.25, help='fraction of files to use (default 0.25)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--cache', help='csv file: load the measurements from it if it exists, otherwise save them to it')
    a = ap.parse_args()

    if a.cache and Path(a.cache).exists():
        df = pd.read_csv(a.cache)
    else:
        df = collect(a.corpus, a.frac, a.seed)
        if a.cache:
            df.to_csv(a.cache, index=False)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=PREDICTORS)
    print(f'\nN = {len(df)} frames, {df["file"].nunique()} files')
    print(df.groupby('vuv')[PREDICTORS].mean().round(2))
    fit(df, a.folds)
