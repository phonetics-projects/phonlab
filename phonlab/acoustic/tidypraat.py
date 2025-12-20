import numpy as np
import pandas as pd

def _check_and_get_tparams(obj, ts, tcol):
    '''
    Internal function to check time-related params and return time series
    for data points.

    Parameters
    ----------

    obj: `obj` param of a `*2df` function

    ts: `ts` param of a `*2df` function

    tcol: `tcol` param of a `*2df` function

    Returns
    -------

    1-d array
    Time series array.
    '''
    if ts is None:
        tpts = obj.ts()  # Get times directly from object
    elif isinstance(ts, pd.DataFrame):
        try:
            tpts = ts[tcol]
        except KeyError:
            raise ValueError(
                f"Input dataframe does not have a time column named '{tcol}'."
                "\nUse the `tcol` parameter to specify the time column."
            ) from None
    else:
        tpts = ts        # Iterable of float
    return tpts

def formant_to_df(obj, num, ts=None, unit='HERTZ', include_bw=False, tcol='sec'):
    """
    Return formant values from a Praat Formant object as a dataframe.

    See the examples for calls using the Parselmouth bindings to Praat.

    Parameters
    ==========

    obj : Praat formant obj
        
        The input Formant object, which is produced by a call to Praat with parselmouth.Sound('wavefile.wav').to_formant_burg() 
        
    num : int or list of int
        The formants from which values will be returned. If `num` is `int` type the values of the first `num` formants are returned. If `num` is a list of numbers then the formants in the list are returned.

    ts : DataFrame, Iterable of floats, or None (default)
        If None (default), then the Formant object's `ts` times are used to query for formant measurements. These are the centers of the analysis frames. The Formant object's values can also be queried at specific times by providing an Iterable of floats, such as a numpy array or Python list.

    unit : str, 'HERTZ' (default) or 'BARK'
        The unit in which formant values are returned. Lower case versions of the units are allowed. See Praat's `Formant_enums.h` file for defined values.

    include_bw : boolean, default False
        If True, bandwidth values for each formant returned are also included.

    tcol : str (default 'sec') or None
        The label of the time column in the output dataframe, 'sec' by default. If None, no time column is added to the dataframe.

    Returns
    =======

    DataFrame
        The output dataframe contains columns of times and formant measurements. The formant columns are labelled `FN`, where `N` is an integer. The time column is labelled by the value of `tcol` (default 'sec'), or omitted if `tcol` is None. If `include_bw` is True then bandwidth columns labelled `bwN` that correspond to `FN` columns are also returned. If `ts` is a dataframe, the output is a concatenation of the input dataframe and the formant/bandwidth measures.

    Examples
    ========

    .. code-block:: Python
    
         snd = parselmouth.Sound(mywav).extract_left_channel()
         
         maxf = 5000       # a vocal tract length parameter, maximum frequency of F4
         fmnt = snd.to_formant_burg(maximum_formant=maxf)  
        
         f12df = phon.formant_to_df(fmnt, 4, include_bw=True)  # convert praat to dataframe
        
         f2df = phon.formant_to_df(fmnt, [2], unit='bark')  # or just the second formant
              
         f1df.to_csv('formants.csv', sep='\t', header=True, index=False) # save to csv
         f1df.to_pickle('formants.zip')
    
    """
    
    fNs = np.arange(1, num+1) if isinstance(num, int) else num
    tpts = _check_and_get_tparams(obj, ts, tcol)
    data = {
        f'F{fn}': \
            np.array(
                [obj.get_value_at_time(fn, t, unit.upper()) for t in tpts]
            ) for fn in fNs
    }
    if include_bw is True:
        data.update({
            f'bw{fn}': \
                np.array(
                    [obj.get_bandwidth_at_time(fn, t) for t in tpts]
                ) for fn in fNs
        })
    if isinstance(ts, pd.DataFrame):
        df = pd.DataFrame(data)
        return pd.concat([ts, df.set_axis(ts.index)], axis='columns')
    else:
        if tcol is None:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame({**{tcol: tpts}, **data})

def pitch_to_df(obj, ts=None, unit='HERTZ', interpolation='LINEAR', tcol='sec'):
    """
    Return pitch values from a Praat Pitch object as a dataframe.

    Parameters
    ==========

    obj : Praat Pitch obj
        The Pitch object produced by parselmouth.Sound('wavefile.wav').to_pitch().

    ts : DataFrame, Iterable of floats, or None (default)
        If None (default), then the Pitch object's `ts` times are used to query for pitch measurements. These are the centers of the analysis frames. The Pitch object's values can also be queried at specific times by providing an Iterable of floats, such as a numpy array or Python list.
    
    unit : str 'HERTZ' (default), 'HERTZ_LOGARITHMIC', 'MEL', 'LOG_HERTZ', 'SEMITONES_1', 'SEMITONES_100', 'SEMITONES_200', 'SEMITONES_440', 'ERB'.
        The unit in which pitch values are returned. Lower case versions of the units are allowed. See Praat's `Pitch_enums.h` file for defined values.

    interpolation : str 'LINEAR' (default), 'NEAREST'
        The type of interpolation to use when returning values. Lower case versions of the units are allowed. See Praat's `Pitch.cpp` and Vector_enums.h` files for defined values.

    tcol : str (default 'sec') or None
        The label of the time column in the output dataframe, 'sec' by default. If None, no time column is added to the dataframe.

    Returns
    =======

    DataFrame
        The output dataframe contains a column of pitch measurements labelled `f0`. The time column is labelled by the value of `tcol` (default 'sec'), or omitted if `tcol` is None. If `ts` is a dataframe, the output is a concatenation of the input dataframe and the pitch measures.

    Examples
    ========

    .. code-block:: Python
    
         snd = parselmouth.Sound(mywav)
         pitch = snd.to_pitch()
        
         hzdf = phon.pitch_to_df(pitch)  # Get dataframe of pitch in Hz.
        
         meldf = phon.pitch_to_df(pitch, unit='mel')  # or, get dataframe of pitch in mel.

         ##  Now you can use all of the Pandas functions with your Praat data

         hzdf.to_csv('hzpitch.csv', sep='\t', header=True, index=False)
         hzdf.to_pickle('melpitch.zip')
         
    """
    
    tpts = _check_and_get_tparams(obj, ts, tcol)
    data = [
        obj.get_value_at_time(t, unit.upper(), interpolation.upper()) \
            for t in tpts
    ]
    if isinstance(ts, pd.DataFrame):
        ts['f0'] = data
        return ts
    else:
        if tcol is None:
            return pd.DataFrame({'f0': data})
        else:
            return pd.DataFrame({tcol: tpts, 'f0': data})

def intensity_to_df(obj, ts=None, interpolation='CUBIC', tcol='sec'):
    """
    Return intensity values from a Praat Intensity object as a dataframe.

    Parameters
    ==========

    obj : Praat Intensity obj
        An Intensity object produced by parselmouth.Sound('wavefile.wav').to_intensity().

    ts : DataFrame, Iterable of floats, or None (default)
        If None (default), then the Intensity object's `ts` times are used to query for intensity measurements. These are the centers of the analysis frames. The Intensity object's values can also be queried at specific times by providing `ts` as an Iterable of floats, such as a numpy array or Python list. If `ts` is a dataframe, the column labelled by the `tcol` parameter is used for the time values and the intensity measures are concatenated to the dataframe.

    interpolation : str 'CUBIC' (default), 'NEAREST', 'LINEAR', 'SINC70', 'SINC700'
        The type of interpolation to use when returning values. Lower case versions of the units are allowed. See Praat's `Intensity.cpp` and `Vector_enums.h` files for defined values.

    tcol : str (default 'sec') or None
        The label of the time column in the output dataframe, 'sec' by default. If None, no time column is added to the dataframe.

    Returns
    =======

    DataFrame
        The output dataframe contains a column of intensity measurements labelled `spl`. The time column is labelled by the value of `tcol` (default 'sec'), or omitted if `tcol` is None. If `ts` is a dataframe, the output is a concatenation of the input dataframe and the intensity measures.

    Examples
    ========

    .. code-block:: Python
    
         snd = parselmouth.Sound(mywav)
         intens = snd.to_intensity()
        
         spldf = phon.intensity_to_df(intens)  # Get dataframe of SPL.

         ## Now you can use all of the Pandas functions with your Praat data
    
         spldf.to_csv('intensity.csv', sep='\t', header=True, index=False)
         splddf.to_pickle('intensity.zip')
    """
    
    tpts = _check_and_get_tparams(obj, ts, tcol)
    data = [obj.get_value(t, interpolation.upper()) for t in tpts]
    if isinstance(ts, pd.DataFrame):
        ts['spl'] = data
        return ts
    else:
        if tcol is None:
            return pd.DataFrame({'spl': data})
        else:
            return pd.DataFrame({tcol: tpts, 'spl': data})

def mfcc_to_df(obj, num, ts=None, tcol='sec', include_energy=True):
    """
    Return mfcc values from a Praat MFCC object as a dataframe.

    Parameters
    ==========

    obj : Formant obj
        An MFCC object produced by parselmouth.Sound('wavefile.wav').to_mfcc().

    num : int
        The number of MFCC coefficients which will be calculated and returned in the columns labelled `c1` ... `cN`, where N is `num`. See the `include_energy` param for `c0`.

    ts : DataFrame, Iterable of floats, or None (default)
        If None (default), then the Formant object's `ts` times are used to query for formant measurements. These are the centers of the analysis frames. The Formant object's values can also be queried at specific times by providing an Iterable of floats, such as a numpy array or Python list.

    tcol : str (default 'sec') or None
        The label of the time column in the output dataframe, 'sec' by default. If None, no time column is added to the dataframe.

    include_energy : bool (default True)
        If True, include the zeroth cepstral coefficient in the column labelled`c0`. For an MFCC object this value relates to energy.

    Returns
    =======

    DataFrame
        The output dataframe contains columns of times and MFCC coefficients. The coefficient columns are labelled `c1` ... `cN`, where `N` is an integer. If `include_energy` is True, then the zeroth cepstral coefficient is also included in the `c0` column. The time column is labelled by the value of `tcol` (default 'sec'), or omitted if `tcol` is None. If `ts` is a dataframe, the output is a concatenation of the input dataframe and the MFCC values.

    Examples
    ========

    .. code-block:: Python
    
         ncoeff = 12
         snd = parselmouth.Sound(mywav)
         mfcc = snd.to_mfcc(number_of_coefficients=ncoeff)
         
         mdf = phon.mfcc_to_df(mfcc, num=ncoeff, include_energy=True)

         # Now you can use all of the Pandas functions with your Praat data

         mdf.to_csv('mfcc.csv', sep='\t', header=True, index=False)
         mdf.to_pickle('mfcc.zip')
    """
    
    tpts = _check_and_get_tparams(obj, ts, tcol)
    if include_energy is True:
        rng = range(num+1)
    else:
        rng = range(1, num+1)
    cols = [f'c{n}' for n in rng]
    data = obj.to_array()[rng].T
    df = pd.DataFrame(data, columns=cols)
    if isinstance(ts, pd.DataFrame):
        return pd.concat([ts, df.set_axis(ts.index)], axis='columns')
    else:
        if tcol is None:
            return df
        else:
            df.insert(0, tcol, tpts)
            return df
