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
    'read_textgrid', 'read_textgrid_praat', 'read_textgrid_with',
    'tg_tiernames', 'TextGridParseError', 'TextGridParserFallbackWarning'
]

import codecs
import os
import re
import sys
import tempfile
import warnings

class TextGridParseError(Exception):
    '''Raised when a file cannot be parsed as a Praat TextGrid.'''
    pass

class TextGridParserFallbackWarning(UserWarning):
    '''
    Issued when the requested textgrid parser fails and the other parser is
    used in its place. Filter it with the `warnings` module to silence it,
    e.g. `warnings.simplefilter('ignore', TextGridParserFallbackWarning)`.
    '''
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

_TIER_CLASS_LINES = ('"IntervalTier"', '"TextTier"')

def _read_praat_short_header(reader, classline):
    '''
    Read the rest of a short format tier header, `classline` having been
    read, and return (tier, declared label count or None).
    '''
    tname = re.sub('^"|"$', '', reader.readline().strip())
    tstart = reader.readline()
    tend = reader.readline()
    try:
        count = int(reader.readline().strip())
    except ValueError:
        count = None
    tclass = 'IntervalTier' if classline == '"IntervalTier"' else 'TextTier'
    return (_new_tier(tclass, tname, tstart.strip(), tend.strip()), count)

def _read_praat_short_labels(reader, tier, collect=True):
    '''
    Read the labels of `tier` in a short format textgrid, from just after its
    header to the next tier header or the end of input. The reader is left at
    the next tier header, which is not consumed. The label count in the tier
    header is not used. When `collect` is False the labels are still read,
    since that is the only way to find where the tier ends, but they are not
    added to the tier.
    '''
    while True:
        loc = reader.tell()
        line = reader.readline()
        if line == '':
            return  # Reached EOF.
        line = line.strip()
        if line == '':
            continue    # Empty line.
        if line in _TIER_CLASS_LINES:
            reader.seek(loc)    # The next tier's header.
            return
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
    while True:
        line = reader.readline()
        if line == '':
            break   # Reached EOF.
        line = line.strip()
        if line == '':
            continue    # Empty line.
        if line not in _TIER_CLASS_LINES:
            raise TextGridParseError(
                f'Parse error. Found label data before any tier was '
                f'declared, at "{line}".'
            )
        tier, _ = _read_praat_short_header(reader, line)
        _read_praat_short_labels(reader, tier, collect)
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

def _read_praat_long_labels(reader, tier, numlabels, collect=True):
    '''
    Read the labels of `tier` in a long format textgrid, from just after its
    metadata to the next 'item [n]:' line or the end of input. The reader is
    left at the 'item [n]:' line, which is not consumed. When `collect` is
    False the labels are still read, since that is the only way to find
    where the tier ends, but they are not added to the tier.
    '''
    if numlabels == 0:
        return  # An empty tier.
    while True:
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
            if _item_re.search(line):
                reader.seek(loc)    # The next tier's 'item [n]:' line.
                return
            if line == '':
                return  # Reached EOF.
            reader.seek(loc)        # The next label's 'intervals|points [n]:'.
            break

def _read_praat_long_start(reader):
    '''
    Discard a long format textgrid's header, up to and including the first
    'item [n]:' line.
    '''
    reader.readline()   # 'File type' line.
    while True:
        line = reader.readline()
        if _item_re.search(line):
            return
        if line == '':
            raise TextGridParseError(
                'Could not read file. No textgrid tiers were found.'
            )

def _read_praat_long(reader, collect=True):
    '''
    Parse a long format textgrid from `reader` and return a list of tiers.

    When `collect` is False the labels are still read, since that is the only
    way to find where each tier ends, but they are not added to the tiers.
    '''
    tiers = []
    _read_praat_long_start(reader)
    tier, numlabels = _read_praat_long_tier_metadata(reader)
    while tier is not None:
        _read_praat_long_labels(reader, tier, numlabels, collect)
        tiers.append(tier)
        if reader.readline() == '':     # The next 'item [n]:' line.
            break
        tier, numlabels = _read_praat_long_tier_metadata(reader)
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

def _skip_lands_ok(lines, target, is_next, blank_ok):
    '''
    Check where a count-based skip over a tier's labels would land. Return the
    index at which the next tier starts (or `len(lines)` at the end of input)
    if the landing is consistent, or `None` if it is not. Blank lines before
    the next tier are passed over when `blank_ok` is True, and trailing blank
    lines at the end of input are always passed over.
    '''
    nlines = len(lines)
    if target > nlines:
        return None
    idx = target
    while idx < nlines and lines[idx].strip() == '':
        idx += 1
    if idx == nlines:
        return nlines
    if idx != target and not blank_ok:
        return None
    return idx if is_next(lines[idx]) else None

def _scan_tiernames_short(lines):
    '''
    Return (names, walked) for a short format textgrid. Each tier's labels
    are skipped over using the count its header declares, and the skip is
    accepted only if it lands on the next tier header or the end of input.
    Otherwise that tier alone is walked as a full read would walk it.
    `walked` holds the indexes of the tiers that were walked.
    '''
    reader = _LineReader(lines)
    for _ in range(7):
        reader.readline()
    names = []
    walked = []
    while True:
        line = reader.readline()
        if line == '':
            break
        line = line.strip()
        if line == '':
            continue
        if line not in _TIER_CLASS_LINES:
            raise TextGridParseError(
                f'Parse error. Found label data before any tier was '
                f'declared, at "{line}".'
            )
        tier, count = _read_praat_short_header(reader, line)
        names.append(tier['name'])
        start = reader.tell()
        nxt = None
        if count is not None and count >= 0:
            per = 3 if tier['class'] == 'IntervalTier' else 2
            nxt = _skip_lands_ok(
                lines, start + count * per,
                lambda l: l.strip() in _TIER_CLASS_LINES, blank_ok=True
            )
        if nxt is not None:
            reader.seek(nxt)
        else:
            walked.append(len(names) - 1)
            reader.seek(start)
            _read_praat_short_labels(reader, tier, collect=False)
    return (tuple(names), tuple(walked))

def _scan_tiernames_long(lines):
    '''
    Return (names, walked) for a long format textgrid, as for
    `_scan_tiernames_short`.
    '''
    reader = _LineReader(lines)
    _read_praat_long_start(reader)
    names = []
    walked = []
    tier, numlabels = _read_praat_long_tier_metadata(reader)
    while tier is not None:
        names.append(tier['name'])
        start = reader.tell()
        per = 4 if tier['class'] == 'IntervalTier' else 3
        nxt = _skip_lands_ok(
            lines, start + numlabels * per,
            lambda l: bool(_item_re.search(l)), blank_ok=False
        )
        if nxt is not None:
            reader.seek(nxt)
        else:
            walked.append(len(names) - 1)
            reader.seek(start)
            _read_praat_long_labels(reader, tier, numlabels, collect=False)
        if reader.readline() == '':     # The next 'item [n]:' line.
            break
        tier, numlabels = _read_praat_long_tier_metadata(reader)
    return (tuple(names), tuple(walked))

def tg_tiernames(tg, codec=None, parser='python'):
    '''
Return the names of the tiers in a Praat textgrid.

The textgrid's labels are not returned, and no dataframes are constructed, so
this is a good deal cheaper than reading the whole textgrid with
`phon.tg_to_df()` when only the tier names are wanted, for example to find
out which tiers a file has before selecting among them. The labels of each
tier are skipped over using the count the tier declares, and each skip is
checked against what follows it. A tier whose skip does not check out, for
example because its labels span several lines or its count is wrong, is read
label by label instead, so the names are always right, and the other tiers
keep the speed advantage.

Parameters
----------

tg : path-like
    Filepath of the input textgrid.

codec : str or None (default None)
    The codec used to decode the textgrid, e.g. 'utf-8'. If the file has a
    byte-order mark, the codec it indicates is used instead. If `None` and the
    file has no byte-order mark, 'utf-8' is assumed. This applies only to the
    'python' parser; Praat determines the encoding itself.

parser : str (default 'python')
    The parser to try first, 'python' or 'praat', with fallback to the other
    if it fails, or 'python.only' or 'praat.only' for no fallback. This
    behaves as the `parser` parameter of `phon.tg_to_df()` does. The 'praat'
    parser asks Praat for the tier names only, but Praat reads the whole file
    to answer, so it does not share the 'python' parser's speed advantage.

Returns
-------

names : tuple of str
    The tier names, in the order the tiers appear in the textgrid. Tiers with
    no name contribute an empty string, and a name used by more than one tier
    appears once for each of them, so the length of `names` is always the
    number of tiers.

Raises
------

    As for `phon.tg_to_df()`: `ValueError` for an unrecognized `parser`, the
    `OSError` of a file that cannot be opened, and `TextGridParseError` if
    both parsers fail. With the '.only' suffix, the named parser's own error.

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
    return _with_fallback(tg, parser, _tiername_readers(codec))

def _tiernames_python(tg, codec=None):
    '''Return the tier names of a textgrid using the pure-Python parser.'''
    lines, codec = _read_lines(tg, codec)
    if _detect_format(lines, tg) == 'long':
        names, _ = _scan_tiernames_long(lines)
    else:
        names, _ = _scan_tiernames_short(lines)
    return names

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

def _import_pcall():
    '''Import and return parselmouth's `praat.call`.'''
    try:
        from parselmouth.praat import call as pcall
    except ImportError:
        raise ImportError(
            'Reading a textgrid with Praat requires the `praat-parselmouth` '
            'package. Install it, or use the default pure-Python parser.'
        ) from None
    return pcall

def _tiernames_praat(tg):
    '''
    Return the tier names of a textgrid as Praat reads them, without asking
    Praat for any labels.
    '''
    pcall = _import_pcall()
    tgobj = pcall('Read from file...', str(tg))[0]
    ntiers = int(pcall(tgobj, 'Get number of tiers'))
    return tuple(pcall(tgobj, 'Get tier name...', n+1) for n in range(ntiers))

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

Praat reads the textgrid and writes what it read back out, to a temporary
file in its short text format, which is then parsed. The short format is used
whatever the format of the input, since it is several times smaller than the
long format and so quicker to parse and lighter on memory. Everything Praat does in
reading the file is reflected in the result, but the whole textgrid is
retrieved in a few calls to Praat rather than several calls per label,
which is far slower for textgrids of any size.

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
    pcall = _import_pcall()
    tgobj = pcall('Read from file...', str(tgfile))[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        praatfile = os.path.join(tmpdir, 'praat.TextGrid')
        pcall(tgobj, 'Save as short text file...', praatfile)
        return _read_praat_output(praatfile)

def _read_praat_output(praatfile):
    '''
    Parse a textgrid that Praat has written. Praat chooses its output
    encoding according to its text writing preference: UTF-16 with a
    byte-order mark, UTF-8, ASCII, or ISO Latin-1, the last two without a
    byte-order mark. A file without a byte-order mark that is not valid UTF-8
    is read as ISO Latin-1.
    '''
    codec = None
    if not detect_encoding(praatfile)[1]:
        with open(praatfile, 'rb') as fh:
            content = fh.read()
        try:
            content.decode('utf-8')
            codec = 'utf-8'
        except UnicodeDecodeError:
            codec = 'latin-1'
    return read_textgrid(praatfile, codec=codec)

def _read_textgrid_praat_calls(tgfile):
    '''
    Read a textgrid with Praat, retrieving each tier and label with its own
    call to Praat. This was the implementation of `read_textgrid_praat()`
    before it had Praat write the textgrid out instead. It is much slower,
    and is kept as an independent reference against which the tests check
    that faster approach.
    '''
    pcall = _import_pcall()
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

_PARSERS = ('python', 'praat')

def _parse_parser_name(parser):
    '''
    Split a `parser` value into the parser name and whether it is to be used
    alone, e.g. 'praat.only' -> ('praat', True). Raise `ValueError` for
    anything else.
    '''
    name, sep, suffix = parser.partition('.') if isinstance(parser, str) \
        else (None, '', '')
    if name not in _PARSERS or (sep and suffix != 'only'):
        choices = ', '.join(
            repr(v) for p in _PARSERS for v in (p, f'{p}.only')
        )
        raise ValueError(
            f'The `parser` parameter must be one of {choices}, not {parser!r}.'
        )
    return (name, sep == '.')

def _textgrid_readers():
    '''
    The full textgrid readers, by parser name. Built at call time, so that
    the module-level functions can be replaced, e.g. in tests.
    '''
    return {'python': read_textgrid, 'praat': read_textgrid_praat}

def _tiername_readers(codec=None):
    '''The tier name readers, by parser name. Built at call time.'''
    return {
        'python': lambda tg: _tiernames_python(tg, codec),
        'praat': _tiernames_praat,
    }

def read_textgrid_with(tgfile, parser='python'):
    '''
Read a Praat textgrid with the named parser, falling back to the other parser
if the named one fails.

Parameters
----------

tgfile : path-like
    Filepath of the input textgrid.

parser : str (default 'python')
    One of 'python', 'praat', 'python.only', or 'praat.only'. The named
    parser is tried first. If it cannot read the textgrid, the other parser
    is tried, and a `TextGridParserFallbackWarning` is issued if that one
    succeeds. With the '.only' suffix the named parser is used alone, and its
    error is raised if it fails. A failure of the 'praat' parser includes
    `praat-parselmouth` not being installed.

Returns
-------

tiers : list of dict
    As documented for `read_textgrid()`.

Raises
------

    ValueError: Raised for an unrecognized `parser` value, before the file is
    opened.

    OSError: Raised if the file cannot be opened, e.g. `FileNotFoundError`.
    This is not treated as a parser failure, and no fallback is tried.

    TextGridParseError: Raised if both parsers fail. It names both errors, and
    the named parser's error is its `__cause__`. If the fallback parser is
    unavailable because `parselmouth` is not installed, the named parser's
    own error is raised instead.

    With the '.only' suffix, the named parser's own error is raised, e.g.
    `TextGridParseError` for 'python.only', or `parselmouth.PraatError` or
    `ImportError` for 'praat.only'.
    '''
    return _with_fallback(tgfile, parser, _textgrid_readers())

def _with_fallback(tgfile, parser, readers, stacklevel=3):
    '''
    Read `tgfile` with `readers[name]` for the parser named by `parser`,
    falling back to the other reader as documented for
    `read_textgrid_with()`. `stacklevel` is passed to `warnings.warn`; the
    default attributes the warning to whoever called the function that
    called this one, so it points at the user's own line.
    '''
    name, only = _parse_parser_name(parser)
    try:
        return readers[name](tgfile)
    except OSError:
        raise
    except Exception as e:
        if only:
            raise
        first_err = e
    other = 'praat' if name == 'python' else 'python'
    try:
        result = readers[other](tgfile)
    except OSError:
        raise
    except ImportError:
        # The fallback is unavailable, so there is nothing to add to the named
        # parser's own error.
        raise first_err from None
    except Exception as second_err:
        raise TextGridParseError(
            f'Neither parser could read "{tgfile}". The {name!r} parser '
            f'raised {type(first_err).__name__}: {first_err} The {other!r} '
            f'parser raised {type(second_err).__name__}: {second_err}'
        ) from first_err
    warnings.warn(
        f'The {name!r} parser could not read "{tgfile}" '
        f'({type(first_err).__name__}: {first_err}). It was read with the '
        f'{other!r} parser instead. Use parser={name + ".only"!r} to prevent '
        f'this.',
        TextGridParserFallbackWarning,
        stacklevel=stacklevel
    )
    return result
