# Python equivalents for Praat objects and commands

```
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from audiolabel import read_label, df2tg
import skphon as skp
```

<details>
    <summary>The `New` menu</summary>

| `New` Menu | Command | Equivalent |
|:------------|:--------|:-----------|
| | Record mono Sound... | |
| | Record stereo Sound... | |
| Sound > | Create Sound as pure tone... | |
| Sound > | Create Sound from formula... | |
| Sound > | Create Sound as tone complex... | |
| Sound > | Create Sound as gammatone... | |
| Sound > | Create Sound as Shepard tone... | |
| Sound > | Create Sound from VowelEditor... | |
| | Create TextGrid... | |
| Tiers > | TODO | |
| | Create Corpus... | |
| Tables > | TODO | |
| Stats > | TODO | |
| Generics > | TODO | |
| Acoustic synthesis (Klatt) > | TODO | |
| Articulatory synthesis > | TODO | |
| Text-to-speech synthesis > | TODO | |
| Constraint grammars > | TODO | |
| Symmetric neural networks > | TODO | |
| Feedforward neural networks > | TODO | |

</details>

<details>
    <summary>The `Open` menu</summary>

| `Open` Menu | Command | Equivalent |
|:------------|:--------|:-----------|
| | Read from file... | For Sound, read into 1d numpy arrays:<br>`y, rate = skp.loadsig('mono.wav')`<br>`y1, y2, rate = loadsig('stereo.wav')`<br><br>For TextGrid, read tiers into dataframes:<br>`[df1, df2] = read_label('twotier.TextGrid', ftype='praat')`<br>TODO: create `skp.tg2df` |
| | Open long sound file... | |
| | Read separate channels from sound file... | `y1, y2, rate = skp.loadsig('stereo.wav')` |
| Read from special sound file > | Read Sound from raw Alaw file... | |
| Read from special sound file > | Read Sound from raw 16-bit Little Endian file... | |
| Read from special sound file > | Read Sound from raw 16-bit Big Endian file... | |
| | Read Table from tab-separated file... | [`df = pd.read_csv('file.csv', sep='\t')`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) |
| | Read Table from comma-separated file... | [`df = pd.read_csv('file.csv')`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) |
| | Read Table from semicolon-separated file... | [`df = pd.read_csv('file.csv', sep=';')`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) |
| | Read Table from whitespace-separated file... | [`df = pd.read_csv('file.csv', sep='\s')`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) |
| | Read TableOfReal from headerless spreadsheet file... | |
| | Read Matrix from raw text file... | |
| | Read Strings from raw text file... | |
| Read from special tier file > | Read TextTier from Xwaves... | |
| Read from special tier file > | Read IntervalTier from Xwaves... | |

</details>

<details>
    <summary>The `Save` menu</summary>

| `Save` Menu | Command | Equivalent |
|:------------|:--------|:-----------|
| | Save as text file... | |
| | Save as short text file... | |
| | Save as binary file... | |
| | Save as WAV file... | |
| | Save as AIFF file... | |
| | Save as AIFC file... | |
| | Save as NeXT/Sun file... | |
| | Save as NIST file... | |
| | Save as FLAC file... | |
| | Save as Kay sound file... | |
| | Save as 24-bit WAV file... | |
| | Save as 32-bit WAV file... | |
| | Save as highest quality MP3 file... | |
| | Save as 8-bit signed file... | |
| | Save as 8-bit unsigned file... | |
| | Save as 16-bit big-endian file... | |
| | Save as 16-bit little-endian file... | |
| | Save as 24-bit big-endian file... | |
| | Save as 24-bit little-endian file... | |
| | Save as 32-bit big-endian file... | |
| | Save as 32-bit little-endian file... | |
| | Save as 24-bit big-endian file... | |
| | Append to existing sound file... | |

</details>

<details>
    <summary>The `Sound` object menu</summary>

TODO: Note that many of the `Get` commands are illustrated with the equivalent of `Interpolation: none` in Praat.

| `Sound` Object Menu | Command | Equivalent |
|:------------|:--------|:-----------|
| | Play | |
| Draw > | Draw... | |
| Draw > | Draw where... | |
| Draw > | Paint where... | |
| Draw > | Paint enclosed... | |
| Query > Query time domain > | Get start time | `start = 0.0`<br>`start = 0.0 + toffset` |
| Query > Query time domain > | Get end time | `end = len(y) / rate`<br>`end = (len(y) / rate) + toffset` |
| Query > Query time domain > | Get total duration | `dur = len(y) / rate` |
| Query > | Get number of channels | The 1d array in `y` is always a single channel. To get the number of channels in an audio file identified by `filepath` do:<br>`nchan = sf.SoundFile(filepath).channels`<br>TODO: does this leave an open filehandle? |
| Query > Query time sampling > | Get number of samples | `n = len(y)`|
| Query > Query time sampling > | Get sampling period | `period = 1 / rate` |
| Query > Query time sampling > | Get sampling frequency | stored as `rate` from `loadsig` |
| Query > Query time sampling > | Get time from sample number | `time = (s / rate) + (0.5 / rate)` |
| Query > Query time sampling > | Get all sample times | `times = sampletimes(y, rate)`<br>`times = (np.arange(len(y)) / rate) + (0.5 / rate)` |
| Query > Query time sampling > | Get sample number from time | `s = np.round(time * rate).astype(int)` |
| Query > | Get value at time... | `val = y[ np.round(time * rate).astype(int) ]` |
| Query > | Get value at sample number... | `val = y[s]` |
| Query > | Get minimum... | `min = y.min()`<br>`min = y[skp.samplerange(0.0, 1.0, rate)].min()` |
| Query > | Get time of minimum... | `time = y.argmin() / rate`<br>`time = (y[skp.samplerange(0.0, 1.0, rate)].argmin() + s1) / rate` |
| Query > | Get maximum... | `max = y.max()`<br>`max = y[skp.samplerange(0.0, 1.0, rate)].max()` |
| Query > | Get time of maximum... | `time = y.argmax() / rate`<br>`time = (y[skp.samplerange(0.0, 1.0, rate)].argmax() + s1) / rate` |
| Query > | Get absolute extremum... | |
| Query > | Get nearest zero crossing... | |
| Query > | Get nearest level crossing... | |
| Query > | Get mean... | `mean = y.mean()`<br>`mean = y[skp.samplerange(0.0, 1.0, rate)].mean()` |
| Query > | Get root-mean-square... | `rms = np.sqrt(np.mean(np.square(y)))`<br>`rms = np.sqrt(np.mean(np.square(y[skp.samplerange(0.0, 1.0, rate)])))` |
| Query > | Get standard deviation... | `std = y.std()`<br>`std = y[skp.samplerange(0.0, 1.0, rate)].std()` |
| Query > | Get energy... | |
| Query > | Get power... | |
| Query > | Get energy in air | |
| Query > | Get power in air | |
| Query > | Get intensity (dB) | |
| Modify > Modify times > | Shift times by... | |
| Modify > Modify times > | Shift times to... | |
| Modify > Modify times > | Scale times by... | |
| Modify > Modify times > | Scale times to... | |
| | Reverse | `y_rev = np.flip(y)` |
| | Formula... | `result = y...` |
| | Formula (part)... | `result = y[skp.samplerange(0.0, 1.0, rate)]...` |
| | Add... | `y_add = y + addval` |
| | Subtract mean | `y_ctr = y - y.mean()` |
| | Multiply... | `y_scaled = y * factor` |
| | Multiply by window... | |
| | Scale peak... | |
| | Scale intensity... | |
| | Set value at sample number... | |
| | Set part to zero... | |
| | Override sampling frequency... | |
| Modify > In-place filters > | Filter with one formant (in-place)... | |
| Modify > In-place filters > | Pre-emphasize (in-place)... | |
| Modify > In-place filters > | De-emphasize (in-place)... | |
| Annotate > | To TextGrid... | |
| Annotate > | To TextGrid (speech activity)... | |
| Annotate > | To TextGrid (silences)... | |
| Analyse periodicity > | To Pitch (filtered ac)... | |
| Analyse periodicity > | To Pitch (raw cc)... | |
| Analyse periodicity > | To Pitch (raw ac)... | |
| Analyse periodicity > | To Pitch (filtered cc)... | |
| Analyse periodicity > | To Pitch (shs)... | |
| Analyse periodicity > | To Pitch (SPINET)... | |
| Analyse periodicity > | To PointProcess (periodic, cc)... | |
| Analyse periodicity > | To PointProcess (periodic, peaks)... | |
| Analyse periodicity > | To PointProcess (extrema)... | |
| Analyse periodicity > | To PointProcess (zeroes)... | |
| Analyse periodicity > | To Harmonicity (cc)... | |
| Analyse periodicity > | To Harmonicity (ac)... | |
| Analyse periodicity > | To Harmonicity (gne)... | |
| Analyse periodicity > | To PowerCepstrogram... | |
| Analyse periodicity > | Autocorrelate... | |
| Analyse spectrum > | To Spectrum... | |
| Analyse spectrum > | To Spectrum (resampled)... | |
| Analyse spectrum > | To Ltas... | |
| Analyse spectrum > | To Ltas (pitch-corrected)... | |
| Analyse spectrum > | To Spectrogram... | |
| Analyse spectrum > | To Cochleagram... | |
| Analyse spectrum > | To Spectrogram (pitch-dependent)... | |
| Analyse spectrum > | To BarkSpectrogram... | |
| Analyse spectrum > | To MelSpectrogram... | |
| Analyse spectrum > | To Formant (burg)... | |
| Analyse spectrum > To Formant (hack) > | To Formant (keep all)... | |
| Analyse spectrum > To Formant (hack) > | To Formant (sl)... | |
| Analyse spectrum > To Formant (hack) > | To Formant (robust)... | |
| Analyse spectrum > To Formant (hack) > | To FormantPath... | |
| Analyse spectrum > To LPC > | To LPC (autocorrelation)... | |
| Analyse spectrum > To LPC > | To LPC (covariance)... | |
| Analyse spectrum > To LPC > | To LPC (burg)... | |
| Analyse spectrum > To LPC > | To LPC (marple)... | |
| Analyse spectrum > | To MFCC... | |
| Analyse spectrum > | To FormantPath (burg)... | |
| | To Intensity... | |
| Manipulate > | To Manipulation... | |
| Manipulate > | To KlattGrid (simple)... | |
| Convert > | Convert to mono | |
| Convert > | Convert to stereo | |
| Convert > | Extract all channels | |
| Convert > | Extract one channels... | |
| Convert > | Extract channels... | |
| Convert > | Extract part... | |
| Convert > | Extract part for overlap... | |
| Convert > | Extract Electroglottogram... | |
| Convert > | To Sound (white channels)... | |
| Convert > | To Sound (bss)... | |
| Convert > | To CrossCorrelationTable... | |
| Convert > | Lengthen (overlap-add)... | |
| Convert > | Deepen band modulation... | |
| Convert > | Change gender... | |
| Convert > | Down to Matrix | |
| Filter > | Filter (pass Hann band)... | |
| Filter > | Filter (stop Hann band)... | |
| Filter > | Filter (formula)... | |
| Filter > | To Sound (derivative)... | |
| Filter > | Reduce noise... | |
| Filter > | Filter (one formant)... | |
| Filter > | Filter (pre-emphasis)... | |
| Filter > | Filter (de-emphasis)... | |
| Filter > | Filter (gammatone)... | |
| Combine > | Combine to stereo | |
| Combine > | Combine into SoundList | |
| Combine > | Combine into SoundSet | |
| Combine > | Concatenate | |
| Combine > | Concatenate recoverably | |
| Combine > | Concatenate with overlap... | |
| Combine > | Convolve... | |
| Combine > | Cross-correlate... | |
| Combine > | To CrossCorrelationTable (combined)... | |
| Combine > | To DTW... | |
| Combine > | To ParamCurve... | |

</details>
