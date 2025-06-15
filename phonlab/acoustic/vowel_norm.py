__all__=['get_deltaF', 'deltaF_norm']

import pandas as df
import numpy as np

def get_deltaF(df, return_value = "deltaF"):
    '''Calcuate the delta F vocal tract length factor from formant values in a dataframe **df**.  The estimate is more stable when the dataframe contains a representative set of vowels spoken by the talker (see Johnson, 2020).

Parameters
==========

df: DataFrame
    The input dataframe must contain columns for F1, F2, F3, and F4.  See phon.track_formants()

return_value: string, default = "deltaF"
    By default the `deltaF` normalization factor is returned. Normalized formant values are `Fn/deltaF`.  If you specify `return_value = "VTL"`, then the function will return the estimated vocal tract length as `VTL=35300/(2*∆F)`, where 35300 is the speed of sound (cm/sec) in warm air.

Returns
=======

deltaF or VTL: numeric
    the quantity returned depends on the parameter `return_value`.

References
==========

Johnson, K. (2020) The Delta F method of vocal tract length normalization for vowels. `Laboratory Phonology`, 11(1), 10. DOI: http://doi.org/10.5334/labphon.196

Example
=======

.. code-block:: Python

    fmtsdf = phon.track_formants(x,fs)
    VTL = phon.get_deltaf(fmtsdf,return_value='VTL')

    '''
    deltaf = np.nanmean([[df['F1']/0.5],[df['F2']/1.5],[df['F3']/2.5],[df['F4']/3.5]])

    if return_value=="VTL":
        return 35300/(2*deltaf)
    else:
        return deltaf

def deltaF_norm(df,groupby = None,deltaF=None):
    '''Perform vocal tract length normalization (deltaF normalization) for each speaker indicated by a 'groupby' variable in a dataframe of vowel formant measurements.  The estimate is more stable when the dataframe contains a representative set of vowels spoken by the talker (see Johnson, 2020).

Parameters
==========

df: DataFrame
    The input dataframe must contain columns for F1, F2, F3, and F4.  See phon.track_formants().  If multiple dataframes from different talkers have been combined into a large multitalker data frame, then there should be a column identifying the speaker for each row, and the name of this column should be passed as the `groupby` input variable.

groupby: string, default=None
    If `df` contains data from more than one talker, the talker identity should be indicated in a column and the name of that column passed in this input variable.

deltaF: numeric or None, default=None
    Supply a value of deltaF to be used for the normalization.  By default the deltaF normalization factor is computed by the function phon.get_deltaF(). 

Note
====

Nothing is returned by this function.  The input dataframe is modified in place with the addition of five new columns -- normalized values of the formants 'F1/∆F', 'F2/∆F', 'F3/∆F', 'F4/∆F', and the 'deltaF' factor used for normalization.

References
==========

Johnson, K. (2020) The Delta F method of vocal tract length normalization for vowels. `Laboratory Phonology`, 11(1), 10. DOI: http://doi.org/10.5334/labphon.196

Example
=======

.. code-block:: Python

    fmtsdf = phon.track_formants(x,fs)
    phon.deltaF_norm(fmtsdf)  # add normalized formant columns
    fmtsdf.head()  # now there are five new columns in the dataframe

    '''

    def _norm_one(df,deltaf=None):  
        # this function normalizes based on all observations in the df
        # use it in a groupby().apply() call to do once for each speaker
        if deltaf is None:
            deltaf = get_deltaF(df)  # by default calculate deltaf from the data
    
        df['F1/∆F'] = df['F1']/deltaf
        df['F2/∆F'] = df['F2']/deltaf
        df['F3/∆F'] = df['F3']/deltaf
        df['F4/∆F'] = df['F4']/deltaf

        df['deltaF'] = deltaf

    if groupby is None:
        df = _norm_one(df,deltaf=deltaF)
    else:
        df = df.groupby(groupby).apply(_norm_one,include_groups=False,kwargs={deltaf:deltaF}).reset_index()
        df.drop(columns=['level_1'], inplace=True)  # clean up the dataframe

    return
