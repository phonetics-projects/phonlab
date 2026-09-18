'''
A pure-Python reader for Praat TextGrid files.

This module provides the low-level parsing used by `phonlab.tg_to_df()`. It
replaces the earlier implementation, which shelled out to Praat by way of
`parselmouth`, and is derived from the `audiolabel` package
(https://github.com/rsprouse/audiolabel).

Both the `short` ("ooTextFile") and `long` TextGrid formats are supported, as
are interval tiers and point (text) tiers, multiline label content, escaped
quotation marks, empty tiers, and files encoded as UTF-8 (with or without a
byte-order mark), UTF-16LE, or UTF-16BE.

Reading is deliberately more literal and more forgiving than Praat in two
respects, both verified against Praat by way of `parselmouth`:

* The label counts declared in a textgrid's headers are not trusted. Every
  label present in the file is read, even when the count says otherwise.
  Praat stops at the declared count, or refuses the file outright when the
  structure runs past it.
* An interval tier that declares no intervals is read as having no labels.
  Praat instead repairs such a tier by supplying a single empty interval that
  spans the tier.

Textgrids written by Praat state their counts correctly and have no empty
interval tiers, so these differences arise only for hand-edited or
machine-generated files.
'''

__all__ = [
    'read_textgrid', 'read_textgrid_praat', 'tg_tiernames',
    'TextGridParseError'
]

import codecs
import re
import sys

class TextGridParseError(Exception):
    '''Raised when a file cannot be parsed as a Praat TextGrid.'''
    pass

# Regex that indicates the end of a label for lines that include an opening
# double quote.
_labendre = re.compile(
    r'''
        (?:^"|[^"])
        (?:"")*      # Allow an even number of preceding double quotes (Praat's
                     # way of including quotation marks in label content)
        "            # Line terminates with a double quote
        \s*          # Ignore whitespace
        $
        |            # OR
        ^\s*"\s*$    # Only a double quote and optional whitespace
    ''',
    re.VERBOSE
)

# Regex that indicates the end of a label for lines that do not include an
# opening double quote, i.e. the end of a multiline label text.
_mlabendre = re.compile(
    r'''
        (?:^|[^"])
        (?:"")*      # Allow an even number of preceding double quotes
        "            # Line terminates with a double quote
        \s*          # Ignore whitespace
        $
        |            # OR
        ^\s*"\s*$    # Only a double quote and optional whitespace
    ''',
    re.VERBOSE
)

# Regex that matches a label line that is exactly quotation marks.
_onlyquotere = re.compile('^(?:"")+$')

# Regexes used by the long format parser.
_t1_re = re.compile(r'(?:xmin|number) = ([^\s]+)')
_t2_re = re.compile(r'xmax = ([^\s]+)')
_text_re = re.compile(r'^\s*(?:text|mark) = (".*)')
_end_label_re = re.compile(r'^\s*(?:item|intervals|points)\s*\[\d+\]:?')
_item_re = re.compile(r'^\s*item\s*\[\d+\]:?')
_class_re = re.compile(r'class = "(.+)"')
_name_re = re.compile(r'name = "(.*)"')
_tstart_re = re.compile(r'xmin = (-?[\d.]+)')
_tend_re = re.compile(r'xmax = (-?[\d.]+)')
_size_re = re.compile(r'(?:intervals|points): size = (\d+)')

_POINT_CLASSES = ('TextTier', 'PointTier')

def _clean_praat_string(s):
    '''
    Strip whitespace at the edges of a label, remove the surrounding quotes,
    and unescape any doubled quotes.
    '''
    return re.sub('""', '"', re.sub('^"|"$', '', s.strip()))

class _LineReader:
    '''
    A minimal file-like reader over a list of lines.

    Lines are returned with their trailing newline so that label content that
    spans multiple lines can be reassembled exactly. An empty string is
    returned at end of input, as `file.readline()` does. Unlike a text-mode
    file object, `tell()`/`seek()` are cheap and exact, which the long format
    parser relies on to push a line back.
    '''

    def __init__(self, lines):
        self._lines = lines
        self._idx = 0

    def readline(self):
        try:
            line = self._lines[self._idx]
        except IndexError:
            return ''
        self._idx += 1
        return line

    def tell(self):
        return self._idx

    def seek(self, idx):
        self._idx = idx

def detect_encoding(tgfile):
    '''
    Guess the encoding of a TextGrid file from its byte-order mark.

    Limited to 'utf-8', 'utf_16_be', and 'utf_16_le'. If no BOM is found,
    'utf-8' is assumed.

Parameters
----------

tgfile : path-like
    Filepath of the input textgrid.

Returns
-------

codec : str
    The name of the detected codec.

has_bom : bool
    True if the detected codec was indicated by a byte-order mark, and False
    if the returned codec is the 'utf-8' default.
    '''
    with open(tgfile, 'rb') as fh:
        firstline = fh.readline()
    if firstline.startswith(codecs.BOM_UTF16_LE):
        return ('utf_16_le', True)
    elif firstline.startswith(codecs.BOM_UTF16_BE):
        return ('utf_16_be', True)
    elif firstline.startswith(codecs.BOM_UTF8):
        return ('utf-8', True)
    return ('utf-8', False)

def _read_lines(tgfile, codec=None):
    '''
    Decode a textgrid file and return (lines, codec).

    A byte-order mark is trusted over a user-specified `codec`, and a warning
    is issued to stderr if the two disagree. Line endings are normalized, and
    each returned line retains a trailing newline.
    '''
    detected_codec, has_bom = detect_encoding(tgfile)
    if has_bom is True:
        if codec is not None and codec != detected_codec:
            sys.stderr.write(
                f'WARNING: overriding user-specified encoding {codec}.\n'
                f'Found BOM for {detected_codec} encoding.\n'
            )
        codec = detected_codec
    elif codec is None:
        codec = detected_codec
    with open(tgfile, 'rb') as fh:
        content = fh.read()
    try:
        content = content.decode(codec)
    except UnicodeDecodeError as e:
        raise TextGridParseError(
            f'Could not decode "{tgfile}" using the "{codec}" codec. Use the '
            f'`codec` parameter to specify the correct encoding.'
        ) from None
    # Discard the BOM, which is decoded as a zero-width no-break space.
    if content.startswith('﻿'):
        content = content[1:]
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    lines = content.split('\n')
    # Trailing newline in the file does not make a final empty line.
    if lines and lines[-1] == '':
        lines = lines[:-1]
    return ([line + '\n' for line in lines], codec)

def _new_tier(tclass, name, start, end):
    '''Return an empty tier dict.'''
    return {
        'class': tclass,
        'name': name,
        'start': float(start),
        'end': float(end),
        'labels': []
    }

def _add_label(tier, t1, t2, text):
    '''Append a label to `tier`, casting times from str as needed.'''
    tier['labels'].append({
        't1': float(t1),
        't2': None if t2 is None else float(t2),
        'text': text
    })

def _read_praat_short(reader, collect=True):
    '''
    Parse a short format textgrid from `reader` and return a list of tiers.

    The interval/point counts in the file header are not trusted. Labels are
    read until the tier changes or input is exhausted. When `collect` is
    False the labels are still read, since that is the only way to find
    where each tier ends, but they are not added to the tiers.
    '''
    tiers = []
    # Discard the header lines. Their content is not used.
    for _ in range(7):
        reader.readline()

    tier = None
    while True:
        line = reader.readline()
        if line == '':
            break   # Reached EOF.
        line = line.strip()
        if line == '':
            continue    # Empty line.
        if line in ('"IntervalTier"', '"TextTier"'):
            # Start a new tier.
            if tier is not None:
                tiers.append(tier)
            tname = re.sub('^"|"$', '', reader.readline().strip())
            tstart = reader.readline()
            tend = reader.readline()
            reader.readline()   # Label count, which is not trusted.
            tclass = 'IntervalTier' if line == '"IntervalTier"' else 'TextTier'
            tier = _new_tier(tclass, tname, tstart.strip(), tend.strip())
        else:
            # Add a label to the existing tier.
            if tier is None:
                raise TextGridParseError(
                    f'Parse error. Found label data before any tier was '
                    f'declared, at "{line}".'
                )
            if tier['class'] == 'IntervalTier':
                t2 = reader.readline()
            else:
                t2 = None
            labtext = reader.readline()
            if _labendre.search(labtext.strip()) is None \
                and _onlyquotere.match(labtext.strip()) is None:
                # The label continues onto following lines.
                while True:
                    addline = reader.readline()
                    labtext += addline
                    if _mlabendre.search(addline) is not None:
                        break
                    elif addline == '':
                        raise TextGridParseError(
                            f'Parse error. Unterminated label "{labtext}" in '
                            f'tier "{tier["name"]}".'
                        )
            if collect:
                _add_label(tier, line, t2, _clean_praat_string(labtext))
    if tier is not None:
        tiers.append(tier)
    return tiers

def _read_praat_long_tier_metadata(reader):
    '''
    Read the metadata section at the top of a tier in a long format textgrid
    and return a (tier, numlabels) tuple. Return (None, 0) if the metadata
    could not be read, which indicates that there are no more tiers.
    '''
    fields = {}
    try:
        for key, regex in (
            ('cls', _class_re), ('tname', _name_re), ('tstart', _tstart_re),
            ('tend', _tend_re), ('numintvl', _size_re)
        ):
            line = reader.readline()
            assert line != ''
            fields[key] = regex.search(line).group(1)
    except (AssertionError, AttributeError):
        return (None, 0)
    if fields['cls'] == 'IntervalTier':
        tclass = 'IntervalTier'
    elif fields['cls'] in _POINT_CLASSES:
        tclass = 'TextTier'
    else:
        return (None, 0)
    tier = _new_tier(tclass, fields['tname'], fields['tstart'], fields['tend'])
    return (tier, int(fields['numintvl']))

def _read_praat_long(reader, collect=True):
    '''
    Parse a long format textgrid from `reader` and return a list of tiers.

    When `collect` is False the labels are still read, since that is the only
    way to find where each tier ends, but they are not added to the tiers.
    '''
    tiers = []
    reader.readline()   # 'File type' line.

    # Discard the remaining header lines. Their content is not used.
    while True:
        line = reader.readline()
        if _item_re.search(line):
            break
        if line == '':
            raise TextGridParseError(
                'Could not read file. No textgrid tiers were found.'
            )

    tier, numlabels = _read_praat_long_tier_metadata(reader)
    while tier is not None:
        if numlabels == 0:
            # An empty tier. Move on to the next one.
            tiers.append(tier)
            reader.readline()   # Skip the 'item [n]:' line.
            tier, numlabels = _read_praat_long_tier_metadata(reader)
            continue
        reader.readline()   # Skip the 'intervals|points [n]:' line.
        t1line = reader.readline()
        try:
            t1 = float(_t1_re.search(t1line).group(1))
            if tier['class'] == 'IntervalTier':
                t2 = float(_t2_re.search(reader.readline()).group(1))
            else:
                t2 = None
            # The captured group stops before the line terminator, which
            # is restored so that multiline label content is reassembled
            # with its newlines intact.
            text = _text_re.search(reader.readline()).group(1) + '\n'
        except AttributeError:
            raise TextGridParseError(
                f'Parse error in tier "{tier["name"]}" near "{t1line.strip()}".'
            ) from None
        while True:
            loc = reader.tell()
            line = reader.readline()
            if not (_end_label_re.search(line) or line == ''):
                text += line
                continue
            if collect:
                _add_label(tier, t1, t2, _clean_praat_string(text))
            if _item_re.search(line):       # Start a new tier.
                tiers.append(tier)
                tier, numlabels = _read_praat_long_tier_metadata(reader)
            elif line == '':                # Reached EOF.
                tiers.append(tier)
                tier = None
            else:      # Found a new label line (intervals|points).
                reader.seek(loc)
            break
    return tiers

def read_textgrid(tgfile, codec=None):
    '''
Read a Praat textgrid and return its tiers.

The short and long textgrid formats are detected automatically.

Parameters
----------

tgfile : path-like
    Filepath of the input textgrid.

codec : str or None (default None)
    The codec used to decode the textgrid, e.g. 'utf-8'. If the file has a
    byte-order mark, the codec it indicates is used instead and a warning is
    issued if it differs from this parameter. If `None` and the file has no
    byte-order mark, 'utf-8' is assumed.

Returns
-------

tiers : list of dict
    One `dict` per textgrid tier, in the order they appear in the textgrid.
    Each has the keys `class` ('IntervalTier' or 'TextTier'), `name`, `start`,
    `end`, and `labels`. The `labels` value is a list of `dict`, each with the
    keys `t1`, `t2`, and `text`. For a point tier, `t2` is `None`.

Raises
------

    TextGridParseError: Raised if the file is not a readable Praat textgrid.
    '''
    return _read_tiers(tgfile, codec, collect=True)

def _detect_format(lines, tgfile):
    '''
    Return 'long' or 'short' for a decoded textgrid, based on the fourth
    line, which holds `xmin`.
    '''
    try:
        xmin = lines[3]
    except IndexError:
        xmin = ''
    if re.match(r'\s*xmin\s*=\s*-?\d', xmin):
        return 'long'
    elif re.match(r'\s*-?\d', xmin):
        return 'short'
    raise TextGridParseError(
        f'"{tgfile}" does not appear to be in a Praat textgrid format.'
    )

def _read_tiers(tgfile, codec=None, collect=True):
    '''
    Detect a textgrid's format and parse it. If `collect` is False the tiers
    are returned with empty `labels` lists.
    '''
    lines, codec = _read_lines(tgfile, codec)
    reader = _LineReader(lines)
    if _detect_format(lines, tgfile) == 'long':
        return _read_praat_long(reader, collect)
    return _read_praat_short(reader, collect)

def _tiernames_fast_short(lines):
    '''
    Read the tier names of a short format textgrid without examining its
    labels, by skipping over them using the label counts the file declares.

    Each label occupies a fixed number of lines unless its content spans
    lines, so after each skip the next line must be a tier header or the end
    of the file. Return `None` when it is not, or when anything else is
    unexpected, so that the caller can fall back to parsing the structure.
    '''
    nlines = len(lines)
    # The first tier header is the first line of its kind; nothing before it
    # can be label content.
    idx = None
    for i, line in enumerate(lines):
        if line.strip() in ('"IntervalTier"', '"TextTier"'):
            idx = i
            break
    if idx is None:
        return None
    names = []
    while idx < nlines:
        if lines[idx].strip() == '':
            idx += 1    # Tolerate blank lines at the end of the file.
            continue
        tclass = lines[idx].strip()
        if tclass not in ('"IntervalTier"', '"TextTier"') or idx + 4 >= nlines:
            return None
        tname = lines[idx+1].strip()
        if len(tname) < 2 or not (tname.startswith('"') and tname.endswith('"')):
            return None
        try:
            count = int(lines[idx+4].strip())
        except ValueError:
            return None
        names.append(re.sub('^"|"$', '', tname))
        # t1/t2/text for an interval, t1/text for a point.
        idx += 5 + (count * (3 if tclass == '"IntervalTier"' else 2))
        if idx > nlines:
            return None
    return tuple(names)

def _tiernames_fast_long(lines):
    '''
    Read the tier names of a long format textgrid without examining its
    labels. As for `_tiernames_fast_short`, return `None` if the file does
    not match expectations, so that the caller can fall back.
    '''
    nlines = len(lines)
    idx = None
    for i, line in enumerate(lines):
        if _item_re.search(line):
            idx = i
            break
    if idx is None:
        return None
    names = []
    while idx < nlines:
        if lines[idx].strip() == '':
            idx += 1
            continue
        if not _item_re.search(lines[idx]) or idx + 5 >= nlines:
            return None
        clsm = _class_re.search(lines[idx+1])
        namem = _name_re.search(lines[idx+2])
        sizem = _size_re.search(lines[idx+5])
        if clsm is None or namem is None or sizem is None:
            return None
        if clsm.group(1) == 'IntervalTier':
            per = 4     # 'intervals [n]:', xmin, xmax, text
        elif clsm.group(1) in _POINT_CLASSES:
            per = 3     # 'points [n]:', number, mark
        else:
            return None
        names.append(namem.group(1))
        idx += 6 + (int(sizem.group(1)) * per)
        if idx > nlines:
            return None
    return tuple(names)

def tg_tiernames(tg, codec=None):
    '''
Return the names of the tiers in a Praat textgrid.

The textgrid's labels are not returned, and no dataframes are constructed, so
this is a good deal cheaper than reading the whole textgrid with
`phon.tg_to_df()` when only the tier names are wanted, for example to find
out which tiers a file has before selecting among them. The labels are
skipped over using the counts the file declares, and each skip is checked
against what follows it, so a textgrid whose counts are wrong is read by
walking its structure instead and still gives the right names.

Parameters
----------

tg : path-like
    Filepath of the input textgrid.

codec : str or None (default None)
    The codec used to decode the textgrid, e.g. 'utf-8'. If the file has a
    byte-order mark, the codec it indicates is used instead. If `None` and the
    file has no byte-order mark, 'utf-8' is assumed.

Returns
-------

names : tuple of str
    The tier names, in the order the tiers appear in the textgrid. Tiers with
    no name contribute an empty string, and a name used by more than one tier
    appears once for each of them, so the length of `names` is always the
    number of tiers.

Raises
------

    TextGridParseError: Raised if the file is not a readable Praat textgrid.

Example
-------

.. code-block:: Python

    textgrid_name = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'im_twelve.TextGrid'

    phon.tg_tiernames(textgrid_name)
    # ('word', 'phone', 'pointph')

    # Read only the tiers that are present.
    names = phon.tg_tiernames(textgrid_name)
    if 'phone' in names:
        [phdf] = phon.tg_to_df(textgrid_name, tiersel=['phone'])
    '''
    lines, codec = _read_lines(tg, codec)
    tgtype = _detect_format(lines, tg)
    if tgtype == 'long':
        names = _tiernames_fast_long(lines)
    else:
        names = _tiernames_fast_short(lines)
    if names is not None:
        return names
    # The label counts the file declares could not be relied on. Walk the
    # textgrid structure instead, still without collecting the labels.
    reader = _LineReader(lines)
    if tgtype == 'long':
        tiers = _read_praat_long(reader, collect=False)
    else:
        tiers = _read_praat_short(reader, collect=False)
    return tuple(t['name'] for t in tiers)

def _praat_extent(tgobj, pcall, tiers):
    '''
    Return the (start, end) times of a textgrid read by parselmouth. The
    attribute and query forms are both tried, since which is available
    depends on the type parselmouth gives the object, and the extent of the
    tiers is used as a last resort.
    '''
    try:
        return (float(tgobj.xmin), float(tgobj.xmax))
    except (AttributeError, TypeError):
        pass
    try:
        return (
            float(pcall(tgobj, 'Get start time')),
            float(pcall(tgobj, 'Get end time'))
        )
    except Exception:
        if tiers == []:
            return (0.0, 0.0)
        return (
            min(t['start'] for t in tiers),
            max(t['end'] for t in tiers)
        )

def read_textgrid_praat(tgfile):
    '''
Read a Praat textgrid with Praat itself, by way of `parselmouth`.

This is an alternative to `read_textgrid()` that returns the same structure,
so the two can be used interchangeably. It requires the `praat-parselmouth`
package, which is imported only when this function is called.

Praat differs from `read_textgrid()` in its handling of malformed textgrids:
it trusts the label counts declared in the file's headers, and it repairs an
interval tier that declares no intervals by supplying a single empty interval
spanning the tier. See this module's documentation for details.

Parameters
----------

tgfile : path-like
    Filepath of the input textgrid.

Returns
-------

tiers : list of dict
    As documented for `read_textgrid()`. Note that Praat determines the
    encoding of the file itself, so there is no `codec` parameter.

Raises
------

    ImportError: Raised if `parselmouth` is not installed.
    parselmouth.PraatError: Raised if Praat cannot read the file.
    '''
    try:
        from parselmouth.praat import call as pcall
    except ImportError:
        raise ImportError(
            'Reading a textgrid with Praat requires the `praat-parselmouth` '
            'package. Install it, or use the default pure-Python parser.'
        ) from None
    tgobj = pcall('Read from file...', str(tgfile))[0]
    ntiers = int(pcall(tgobj, 'Get number of tiers'))
    tiers = []
    for n in range(ntiers):
        isintvl = pcall(tgobj, 'Is interval tier...', n+1)
        labels = []
        if isintvl is True or isintvl == 1 or isintvl == '1':
            tclass = 'IntervalTier'
            for i in range(int(pcall(tgobj, 'Get number of intervals...', n+1))):
                labels.append({
                    't1': pcall(tgobj, 'Get start time of interval...', n+1, i+1),
                    't2': pcall(tgobj, 'Get end time of interval...', n+1, i+1),
                    'text': pcall(tgobj, 'Get label of interval...', n+1, i+1)
                })
        else:
            tclass = 'TextTier'
            for i in range(int(pcall(tgobj, 'Get number of points...', n+1))):
                labels.append({
                    't1': pcall(tgobj, 'Get time of point...', n+1, i+1),
                    't2': None,
                    'text': pcall(tgobj, 'Get label of point...', n+1, i+1)
                })
        tiers.append({
            'class': tclass,
            'name': pcall(tgobj, 'Get tier name...', n+1),
            'start': None,
            'end': None,
            'labels': labels
        })
    # Praat reports one extent for the whole textgrid rather than per tier.
    start, end = _praat_extent(tgobj, pcall, [])
    for tier in tiers:
        tier['start'], tier['end'] = start, end
    return tiers
