__all__=['srt2tg']

import os
import pandas as pd
import numpy as np
import srt
from ..utils.tidy import df_to_tg

def srt2tg(filename, quiet=True, exclude_talkers = ["Keith Johnson"]):
    """Convert a subtitles text file into a Praat TextGrid. 

Utilities like otter.ai can generate a transcription of the speech in an audio file.  A common format for this sort of transcription is .srt -- a "SubRip subtitle" file.  This function takes  an `srt` file as input and parses it, with the srt Python library, and writes it out with the audiolabel.df2tg() function.

Parameters
==========

filename : string
    the name/path to a subtitle file.

exclude_talkers : list
    A list of talkers who can be ignored, no textgrid will be produced for talkers on this list.  In the current implementation, there should be only one talker who will be kept - i.e. only one TextGrid will be written by this function.

quiet : boolean, default = True
    allow diagnostic messages to print

Returns
=======

    nothing is returned, the effect of the function is to write a file, the name of which is the input filename with .srt replaced by .TextGrid
    """
    
    TextGrid_name = filename.replace(".srt",".TextGrid")

    if not quiet: print(f"converting {filename} to {TextGrid_name}")

    fd = open(filename,"r")
    dfs = {}  # dictionary to store a new dataframe (tier) for each talker
    previous_end = 0

    for sub in srt.parse(fd): # iterate over subtitle objects in file
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        parts = sub.content.split(":",1)  # otter.ai puts "talker: text" for a new talker

        if (len(parts)>1):   # switch from one talker to the next
            text = parts[1]
            talker = parts[0]
        else:
            text = sub.content
        
        if talker in exclude_talkers:
            continue   # go on to the next subtitle

        # make a df for this talker, set previous end
        if not talker in list(dfs.keys()):
            dfs[talker] = pd.DataFrame({talker:[],'t1':[],'t2':[]})
            previous_end = 0
        else:
            previous_end = dfs[talker].iloc[-1].t2   # end point of the last thing said by this talker
        
        if start > previous_end:
            dfs[talker].loc[len(dfs[talker].index)] = ["",previous_end,start]  # empty sections
        if start<end:
            dfs[talker].loc[len(dfs[talker].index)] = [text,start,end]  # parts said by the talker
        if start<previous_end:
            if not quiet: print("error at index {} in talker {}: {}".format(sub.index,talker,text))
    
    keys = list(dfs.keys())
    tg = df_to_tg(list(dfs.values()), keys, ftype='praat_short', outfile=TextGrid_name)
 
    if not quiet: print(keys)  # print the names of the talkers
    