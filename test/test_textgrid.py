"""Tests for phonlab's Praat TextGrid reading and writing.

These are ported from the `audiolabel` package
(https://github.com/rsprouse/audiolabel), whose TextGrid parser
`phonlab.utils.textgrid` is based on. The reading tests that exercised
`audiolabel.LabelManager`/`read_label` are rewritten against
`phonlab.tg_to_df()`, and the `audiolabel.df2tg` tests are rewritten against
`phonlab.df_to_tg()`.

The TextGrid files in `test/data` are copied from the `audiolabel` test suite.
"""

from pathlib import Path

import warnings

import pandas as pd
import pytest

from phonlab.utils import textgrid as tgmodule
from phonlab.utils.textgrid import (
    TextGridParseError, TextGridParserFallbackWarning, detect_encoding,
    _read_textgrid, _read_textgrid_praat, _read_textgrid_with, tg_tiernames
)
from phonlab.utils.tidy import df_to_tg, tg_to_df

DATA = Path(__file__).parent / 'data'

def label_at(df, t, col):
    """Return the label in `col` whose interval contains time `t`."""
    hits = df[(df['t1'] <= t) & (df['t2'] > t)]
    assert len(hits) == 1
    return hits[col].iloc[0]

#### Reading: format detection and tier structure ####

def test_praat_long():
    """A long format textgrid is read, and its format detected."""
    tiers = _read_textgrid(DATA / 'this_is_a_label_file.long.TextGrid')
    assert len(tiers) == 3
    assert tg_tiernames(DATA / 'this_is_a_label_file.long.TextGrid') == \
        ('word', 'phone', 'stimulus')

def test_praat_short():
    """A short format textgrid is read, and its format detected."""
    tiers = _read_textgrid(DATA / 'this_is_a_label_file.short.TextGrid')
    assert len(tiers) == 3
    assert tg_tiernames(DATA / 'this_is_a_label_file.short.TextGrid') == \
        ('word', 'phone', 'stimulus')

@pytest.mark.parametrize('tgfile', [
    'empty_tier.long.TextGrid', 'empty_tier.short.TextGrid'
])
def test_praat_empty_tier(tgfile):
    """Empty tiers are read as empty dataframes, in both formats.

    Note that Praat repairs an interval tier that declares no intervals by
    supplying one empty interval spanning the tier, so reading
    'empty_interval' through Praat gives one row rather than none. The
    literal reading is intended here.
    """
    dfs = tg_to_df(DATA / tgfile)
    assert len(dfs) == 5
    assert tg_tiernames(DATA / tgfile) == (
        'V1', 'empty_point_1', 'empty_interval', 'V2', 'empty_point_end'
    )
    assert [len(df) for df in dfs] == [3, 0, 0, 4, 0]
    # Empty tiers still have the columns their tier type calls for.
    assert dfs[1].columns.tolist() == ['t1', 'empty_point_1']
    assert dfs[2].columns.tolist() == ['t1', 't2', 'empty_interval']

def test_praat_short_and_long_agree():
    """The same textgrid in short and long format gives the same dataframes."""
    shortdfs = tg_to_df(DATA / 'this_is_a_label_file.short.TextGrid')
    longdfs = tg_to_df(DATA / 'this_is_a_label_file.long.TextGrid')
    assert len(shortdfs) == len(longdfs)
    for sdf, ldf in zip(shortdfs, longdfs):
        pd.testing.assert_frame_equal(sdf, ldf)

def test_praat_short_multiline():
    """Label content that spans multiple lines is read intact."""
    [df] = tg_to_df(DATA / 'multiline.short.TextGrid')
    texts = ['', 'a', 'b\n', 'c\n', '"', '1', '""', '"\n', '""\n', '', '""\n"']
    assert df['multiline'].tolist() == texts
    # The interval count in the header is not trusted; this file understates it.
    assert len(df) == 11

def test_praat_long_label_count_not_trusted():
    """The declared label count is not trusted in long format either. Praat
    itself refuses this file, whose 'phone' tier declares 8 intervals but
    holds 9; all 9 are read."""
    tiers = _read_textgrid(DATA / 'ipa.TextGrid')
    assert [len(t['labels']) for t in tiers] == [6, 9, 3]
    phdf = tg_to_df(DATA / 'ipa.TextGrid', tiersel=['phone'])[0]
    assert len(phdf) == 9
    assert phdf['phone'].iloc[-1] == 'E'

def test_praat_quotes():
    """Doubled quotation marks in label content are unescaped."""
    [intdf, ptdf] = tg_to_df(DATA / 'quotes.TextGrid')
    assert intdf['interval'].tolist() == ['"a\'b\'c"d e"', '']
    assert ptdf['point'].tolist() == ['"a\'b\'c"d e"']

def test_praat_from_eaf():
    """Some textgrids exported from ELAN are valid (Praat can open them) even
    though they differ in some details from the long textgrids created by
    Praat. Make sure these can be read correctly."""
    [df] = tg_to_df(DATA / 'from_eaf.long.TextGrid')
    assert df.columns.tolist() == ['t1', 't2', 'transcript']
    assert df.shape == (6, 3)
    assert df['transcript'][0] == 'asdfasf'
    assert df['transcript'][4] == 'text'

def test_praat_empty_tier_name():
    """Tiers with empty names are read, and name the label column ''."""
    dfs = tg_to_df(DATA / 'empty_name.TextGrid')
    assert tg_tiernames(DATA / 'empty_name.TextGrid') == ('word', '', '')
    assert [df.columns[-1] for df in dfs] == ['word', '', '']

#### Reading: encodings ####

def test_detect_encoding():
    """Byte-order marks are detected, and utf-8 assumed when there is none."""
    assert detect_encoding(DATA / 'ipa.TextGrid') == ('utf-8', True)
    assert detect_encoding(DATA / 'utf8_no_BOM.TextGrid') == ('utf-8', False)
    assert detect_encoding(DATA / 'Turkmen_NA_20130919_G_3.TextGrid') == \
        ('utf_16_be', True)

def test_praat_utf_8():
    """Non-ASCII label content in a utf-8 textgrid is read correctly."""
    phdf = tg_to_df(DATA / 'ipa.TextGrid', tiersel=['phone'])[0]
    assert label_at(phdf, 0.6, 'phone') == u'ɯ'
    assert label_at(phdf, 0.7, 'phone') == u'ʤ'

def test_praat_utf_8_no_bom():
    """A utf-8 textgrid with no byte-order mark is read correctly."""
    [s1df, s2df] = tg_to_df(DATA / 'utf8_no_BOM.TextGrid')
    assert label_at(s1df, 8.0, 's1') == u'bet b\xedt'
    assert label_at(s2df, 6.0, 's2') == u'bat bat'

def test_praat_utf_16_be():
    """A utf-16be textgrid is read correctly."""
    [wddf, glossdf] = tg_to_df(DATA / 'Turkmen_NA_20130919_G_3.TextGrid')
    assert wddf.shape == (171, 3)
    assert glossdf.shape == (167, 3)
    assert wddf['word'][1] == u'jɛr'

def test_praat_utf_16_be_warn(capsys):
    """A byte-order mark overrides a user-specified codec, with a warning."""
    _read_textgrid(DATA / 'Turkmen_NA_20130919_G_3.TextGrid', codec='utf-8')
    err = capsys.readouterr().err
    assert 'overriding user-specified encoding utf-8' in err
    assert 'utf_16_be' in err

def test_praat_no_warn_without_bom(capsys):
    """No warning is issued when there is no byte-order mark to conflict with."""
    _read_textgrid(DATA / 'utf8_no_BOM.TextGrid', codec='utf-8')
    assert capsys.readouterr().err == ''

#### Reading: the tg_to_df interface ####

def test_tg_to_df():
    """All tiers are returned, in textgrid order, when `tiersel` is empty."""
    [phdf, wddf, ctxdf] = tg_to_df(DATA / 'this_is_a_label_file.TextGrid')
    assert wddf.columns.tolist() == ['t1', 't2', 'word']
    assert wddf.shape == (6, 3)
    assert wddf['word'][1] == 'IS'
    assert ctxdf.shape == (3, 3)
    assert ctxdf['context'][1] == 'sad'
    assert phdf.shape == (15, 3)
    assert phdf['phone'][2] == 'S'

def test_tg_to_df_dtypes():
    """Time columns are floats and label columns hold `str`."""
    [phdf, wddf, ctxdf] = tg_to_df(DATA / 'this_is_a_label_file.TextGrid')
    assert phdf['t1'].dtype == 'float64'
    assert phdf['t2'].dtype == 'float64'
    assert all(isinstance(lbl, str) for lbl in phdf['phone'])

def test_tg_to_df_tiersel():
    """`tiersel` selects and orders tiers, by name or by index."""
    [wddf, phdf] = tg_to_df(
        DATA / 'this_is_a_label_file.TextGrid', tiersel=['word', 'phone']
    )
    assert wddf.columns.tolist() == ['t1', 't2', 'word']
    assert phdf.columns.tolist() == ['t1', 't2', 'phone']
    assert wddf.shape == (6, 3)
    assert phdf.shape == (15, 3)
    # Same selection by index, and mixed name/index selection.
    [wddf2, phdf2] = tg_to_df(DATA / 'this_is_a_label_file.TextGrid', [1, 0])
    pd.testing.assert_frame_equal(wddf, wddf2)
    pd.testing.assert_frame_equal(phdf, phdf2)
    [wddf3, phdf3] = tg_to_df(
        DATA / 'this_is_a_label_file.TextGrid', tiersel=['word', 0]
    )
    pd.testing.assert_frame_equal(wddf, wddf3)
    pd.testing.assert_frame_equal(phdf, phdf3)

def test_tg_to_df_tiersel_not_modified():
    """`tg_to_df` does not modify the caller's `tiersel` list."""
    tiersel = ['word', 'phone']
    tg_to_df(DATA / 'this_is_a_label_file.TextGrid', tiersel=tiersel)
    assert tiersel == ['word', 'phone']

def test_tg_to_df_tiersel_bad_name():
    """An unknown tier name in `tiersel` raises `KeyError`."""
    with pytest.raises(KeyError):
        tg_to_df(DATA / 'this_is_a_label_file.TextGrid', tiersel=['nosuchtier'])

def test_tg_to_df_names():
    """`names` renames the label columns."""
    [phdf, wddf] = tg_to_df(
        DATA / 'this_is_a_label_file.TextGrid',
        tiersel=['phone', 'word'],
        names=['seg', 'wrd']
    )
    assert phdf.columns.tolist() == ['t1', 't2', 'seg']
    assert wddf.columns.tolist() == ['t1', 't2', 'wrd']
    assert phdf['seg'][2] == 'S'

def test_tg_to_df_names_str():
    """A `str` `names` value is used for every selected tier."""
    dfs = tg_to_df(
        DATA / 'this_is_a_label_file.TextGrid',
        tiersel=['phone', 'word'],
        names='label'
    )
    assert all(df.columns.tolist() == ['t1', 't2', 'label'] for df in dfs)

def test_tg_to_df_names_too_few():
    """Too few `names` raises a `ValueError`."""
    with pytest.raises(ValueError, match='Not enough names'):
        tg_to_df(
            DATA / 'this_is_a_label_file.TextGrid',
            tiersel=['phone', 'word'],
            names=['seg']
        )

def test_tg_to_df_point_tier():
    """Point tiers have a `t1` column and no `t2` column."""
    stdf = tg_to_df(
        DATA / 'this_is_a_label_file.short.TextGrid', tiersel=['stimulus']
    )[0]
    assert stdf.columns.tolist() == ['t1', 'stimulus']
    assert stdf['stimulus'].tolist() == ['1', '2', '3']

def test_tg_to_df_str_path():
    """A `str` filepath works as well as a `Path`."""
    frompath = tg_to_df(DATA / 'this_is_a_label_file.TextGrid')
    fromstr = tg_to_df(str(DATA / 'this_is_a_label_file.TextGrid'))
    for pdf, sdf in zip(frompath, fromstr):
        pd.testing.assert_frame_equal(pdf, sdf)

def test_tg_to_df_bad_file(tmp_path):
    """A file that is not a textgrid raises `TextGridParseError`."""
    notatg = tmp_path / 'notatg.TextGrid'
    notatg.write_text('this is not\na textgrid at all\n')
    with pytest.raises(TextGridParseError):
        tg_to_df(notatg)

#### Writing: df_to_tg round trips ####

@pytest.mark.parametrize('tgtype', ['short', 'long'])
def test_df_to_tg_round_trip(tgtype, tmp_path):
    """Dataframes written by `df_to_tg` are read back unchanged."""
    [wddf, phdf, stdf] = tg_to_df(
        DATA / 'this_is_a_label_file.short.TextGrid',
        tiersel=['word', 'phone', 'stimulus']
    )
    outfile = tmp_path / f'roundtrip.{tgtype}.TextGrid'
    df_to_tg(
        [wddf, phdf, stdf],
        tiercols=['word', 'phone', 'stimulus'],
        ts=[['t1', 't2'], ['t1', 't2'], ['t1', None]],
        tgtype=tgtype,
        outfile=outfile
    )
    [wddf2, phdf2, stdf2] = tg_to_df(
        outfile, tiersel=['word', 'phone', 'stimulus']
    )
    assert wddf2['word'][0] == ''
    assert wddf2['word'][1] == 'This'
    assert wddf2['word'][4] == 'label'
    assert phdf2['phone'][2] == 'IH'
    assert phdf2['phone'][5] == 'Z'
    assert stdf2['stimulus'][0] == '1'
    assert stdf2['stimulus'][2] == '3'
    pd.testing.assert_frame_equal(wddf, wddf2)
    pd.testing.assert_frame_equal(phdf, phdf2)
    pd.testing.assert_frame_equal(stdf, stdf2)

def test_df_to_tg_round_trip_quotes(tmp_path):
    """Quotation marks in label content survive a write/read round trip."""
    [intdf, ptdf] = tg_to_df(DATA / 'quotes.TextGrid')
    outfile = tmp_path / 'quotes.TextGrid'
    df_to_tg(
        [intdf, ptdf],
        tiercols=['interval', 'point'],
        ts=[['t1', 't2'], ['t1', None]],
        outfile=outfile
    )
    [intdf2, ptdf2] = tg_to_df(outfile)
    pd.testing.assert_frame_equal(intdf, intdf2)
    pd.testing.assert_frame_equal(ptdf, ptdf2)

def test_df_to_tg_round_trip_utf_8(tmp_path):
    """Non-ASCII label content survives a write/read round trip."""
    [wddf, phdf, ctxdf] = tg_to_df(DATA / 'ipa.TextGrid')
    outfile = tmp_path / 'ipa.TextGrid'
    df_to_tg([wddf, phdf, ctxdf], tiercols=['word', 'phone', 'context'],
        outfile=outfile)
    [wddf2, phdf2, ctxdf2] = tg_to_df(outfile)
    pd.testing.assert_frame_equal(phdf, phdf2)
    assert label_at(phdf2, 0.6, 'phone') == u'ɯ'

def test_df_to_tg_round_trip_from_csv(tmp_path):
    """Label columns read from csv, with NaN and numeric content, round trip."""
    [wddf, phdf, stdf] = tg_to_df(
        DATA / 'this_is_a_label_file.short.TextGrid',
        tiersel=['word', 'phone', 'stimulus']
    )
    csvdir = tmp_path / 'csv'
    csvdir.mkdir()
    for df, name in ((wddf, 'wd'), (phdf, 'ph'), (stdf, 'st')):
        df.to_csv(csvdir / f'{name}.csv', index=False)
    # Empty labels read back from csv as NaN, and 'stimulus' as int.
    wddf = pd.read_csv(csvdir / 'wd.csv')
    assert wddf['word'].isna().iloc[0]
    phdf = pd.read_csv(csvdir / 'ph.csv')
    stdf = pd.read_csv(csvdir / 'st.csv', dtype={'stimulus': int})
    outfile = tmp_path / 'fromcsv.TextGrid'
    df_to_tg(
        [wddf, phdf, stdf],
        tiercols=['word', 'phone', 'stimulus'],
        ts=[['t1', 't2'], ['t1', 't2'], ['t1', None]],
        outfile=outfile
    )
    [wddf2, phdf2, stdf2] = tg_to_df(
        outfile, tiersel=['word', 'phone', 'stimulus']
    )
    assert wddf2['word'][0] == ''
    assert wddf2['word'][1] == 'This'
    assert wddf2['word'][4] == 'label'
    assert phdf2['phone'][2] == 'IH'
    assert phdf2['phone'][5] == 'Z'
    assert stdf2['stimulus'][0] == '1'
    assert stdf2['stimulus'][2] == '3'

def test_df_to_tg_tiercol_rename(tmp_path):
    """A dict `tiercols` value names the tier differently from the column."""
    [phdf] = tg_to_df(DATA / 'this_is_a_label_file.TextGrid', tiersel=['phone'])
    outfile = tmp_path / 'renamed.TextGrid'
    df_to_tg(phdf, tiercols={'phone': 'segment'}, outfile=outfile)
    assert tg_tiernames(outfile) == ('segment',)

#### Writing: gap filling ####

def _degap_phones():
    """A non-contiguous phone dataframe for the gap-filling tests."""
    [phdf] = tg_to_df(DATA / 'this_is_a_label_file.TextGrid', tiersel=['phone'])
    return phdf[phdf['phone'].isin(['AH0', 'EY1', 'L'])]

def test_df_to_tg_fill_gaps():
    """Gaps between labels are filled with empty labels by default."""
    output = df_to_tg(
        _degap_phones(), 'phone', start=0.0, end=1.8, fmt='0.4f',
        tgtype='short'
    )
    assert output.strip().endswith('''
"IntervalTier"
"phone"
0.0000
1.8000
10
0.0000
0.6111
""
0.6111
0.6610
"AH0"
0.6610
0.8007
"L"
0.8007
0.9703
"EY1"
0.9703
1.0002
""
1.0002
1.0302
"AH0"
1.0302
1.1399
"L"
1.1399
1.4891
""
1.4891
1.6288
"L"
1.6288
1.8000
""
'''.strip())

def test_df_to_tg_fill_gaps_text():
    """`fill_gaps` sets the text of the inserted labels."""
    output = df_to_tg(
        _degap_phones(), 'phone', start=0.0, end=1.8, fmt='0.4f',
        fill_gaps='<sil>', tgtype='short'
    )
    assert '"<sil>"' in output
    assert '""' not in output.split('"phone"', 1)[1]
    [df] = tg_to_df_from_str(output)
    assert df['phone'].tolist() == [
        '<sil>', 'AH0', 'L', 'EY1', '<sil>', 'AH0', 'L', '<sil>', 'L', '<sil>'
    ]

def test_df_to_tg_no_fill_gaps(tmp_path):
    """`fill_gaps=None` leaves the gaps between labels unfilled."""
    outfile = tmp_path / 'nofill.TextGrid'
    df_to_tg(
        _degap_phones(), 'phone', start=0.0, end=1.8, fmt='0.4f',
        fill_gaps=None, tgtype='short', outfile=outfile
    )
    [df] = tg_to_df(outfile)
    assert df['phone'].tolist() == ['AH0', 'L', 'EY1', 'AH0', 'L', 'L']

def tg_to_df_from_str(tgstr, tmpdir=None):
    """Read a textgrid held in a `str` by way of a temporary file."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        'w', suffix='.TextGrid', delete=False, encoding='utf-8'
    ) as fh:
        fh.write(tgstr)
        name = fh.name
    try:
        return tg_to_df(name)
    finally:
        Path(name).unlink()

#### Writing: formatting and edge cases ####

def test_df_to_tg_fmt_applies_to_preamble():
    """`fmt` is applied to the textgrid xmin/xmax as well as to the labels."""
    output = df_to_tg(
        _degap_phones(), 'phone', start=0.0, end=1.8, fmt='0.4f',
        tgtype='short'
    )
    assert output.split('\n')[3:5] == ['0.0000', '1.8000']

def test_df_to_tg_fill_gaps_derived_start_end():
    """Gap filling uses the unrounded start/end, so rounding under `fmt` does
    not introduce zero-duration labels at the tier edges."""
    output = df_to_tg(
        _degap_phones(), 'phone', start=None, end=None, fmt='0.4f',
        tgtype='short'
    )
    [df] = tg_to_df_from_str(output)
    assert (df['t2'] > df['t1']).all()
    assert df['phone'].tolist() == ['AH0', 'L', 'EY1', '', 'AH0', 'L', '', 'L']
    assert output.strip().endswith('''
"IntervalTier"
"phone"
0.6111
1.6288
8
0.6111
0.6610
"AH0"
0.6610
0.8007
"L"
0.8007
0.9703
"EY1"
0.9703
1.0002
""
1.0002
1.0302
"AH0"
1.0302
1.1399
"L"
1.1399
1.4891
""
1.4891
1.6288
"L"
'''.strip())

@pytest.mark.parametrize('tgtype', ['short', 'long'])
def test_df_to_tg_multiline_labels(tgtype, tmp_path):
    """Label content containing newlines survives a write/read round trip."""
    [df] = tg_to_df(DATA / 'multiline.short.TextGrid')
    outfile = tmp_path / f'multiline.{tgtype}.TextGrid'
    df_to_tg(
        df, 'multiline', start=None, end=None, fill_gaps=None, tgtype=tgtype,
        outfile=outfile
    )
    [df2] = tg_to_df(outfile)
    pd.testing.assert_frame_equal(df, df2)

@pytest.mark.parametrize('tgtype', ['short', 'long'])
def test_df_to_tg_empty_tier(tgtype, tmp_path):
    """A tier with no labels is written and read back as an empty tier."""
    dfs = tg_to_df(DATA / 'empty_tier.short.TextGrid')
    outfile = tmp_path / f'empty.{tgtype}.TextGrid'
    df_to_tg(
        dfs,
        tiercols=[df.columns[-1] for df in dfs],
        ts=[['t1', 't2' if 't2' in df.columns else None] for df in dfs],
        start=None, end=None, fill_gaps=None, tgtype=tgtype, outfile=outfile
    )
    dfs2 = tg_to_df(outfile)
    assert [len(df) for df in dfs2] == [3, 0, 0, 4, 0]
    assert tg_tiernames(outfile) == (
        'V1', 'empty_point_1', 'empty_interval', 'V2', 'empty_point_end'
    )
    for df, df2 in zip(dfs, dfs2):
        pd.testing.assert_frame_equal(df, df2)

def test_df_to_tg_long_point_tier_keywords(tmp_path):
    """Long format output labels point tier entries 'points', as Praat does,
    and interval tier entries 'intervals'."""
    [wddf, phdf, stdf] = tg_to_df(
        DATA / 'this_is_a_label_file.short.TextGrid',
        tiersel=['word', 'phone', 'stimulus']
    )
    outfile = tmp_path / 'keywords.long.TextGrid'
    df_to_tg(
        [wddf, stdf],
        tiercols=['word', 'stimulus'],
        ts=[['t1', 't2'], ['t1', None]],
        tgtype='long',
        outfile=outfile
    )
    output = outfile.read_text(encoding='utf-8')
    wdblock, stblock = output.split('class = "TextTier"')
    assert 'intervals: size = 6' in wdblock
    assert 'intervals [1]:' in wdblock
    assert 'points: size = 3' in stblock
    assert 'points [1]:' in stblock
    assert 'intervals' not in stblock
    # The point tier still round trips.
    stdf2 = tg_to_df(outfile, tiersel=['stimulus'])[0]
    pd.testing.assert_frame_equal(stdf, stdf2)

def test_read_long_point_tier_intervals_keyword(tmp_path):
    """The reader stays lenient: a long format point tier whose entries are
    labelled 'intervals' is still read correctly."""
    tgfile = tmp_path / 'lenient.long.TextGrid'
    tgfile.write_text('''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "TextTier"
        name = "point"
        xmin = 0.0
        xmax = 1.0
        intervals: size = 2
        intervals [1]:
            number = 0.25
            mark = "a"
        intervals [2]:
            number = 0.75
            mark = "b"
''', encoding='utf-8')
    [df] = tg_to_df(tgfile)
    assert df.columns.tolist() == ['t1', 'point']
    assert df['t1'].tolist() == [0.25, 0.75]
    assert df['point'].tolist() == ['a', 'b']


#### The `parser` parameter ####

# Textgrids that Praat itself reads, and reads the same way as the pure-Python
# parser. The fixtures left out are the deliberately malformed ones: Praat
# refuses 'ipa' and 'empty_name' (their phone tier declares 8 intervals and
# holds 9) and reads only 10 of the 11 labels in 'multiline.short'.
PRAAT_READABLE = [
    'this_is_a_label_file.TextGrid',
    'this_is_a_label_file.long.TextGrid',
    'this_is_a_label_file.short.TextGrid',
    'quotes.TextGrid',
    'utf8_no_BOM.TextGrid',
    'Turkmen_NA_20130919_G_3.TextGrid',
    'from_eaf.long.TextGrid',
]

def test_tg_to_df_parser_default_is_python():
    """The default `parser` value gives the pure-Python parser's result."""
    default = tg_to_df(DATA / 'this_is_a_label_file.TextGrid')
    explicit = tg_to_df(DATA / 'this_is_a_label_file.TextGrid', parser='python')
    for ddf, edf in zip(default, explicit):
        pd.testing.assert_frame_equal(ddf, edf)

@pytest.mark.parametrize('parser', [
    'praat_short', 'Python', '', None, 'only', '.only', 'python.', 'praat.Only',
    'python.only.only', 'python-only', 'python_only'
])
def test_tg_to_df_parser_invalid(parser):
    """An unrecognized `parser` value raises a `ValueError`."""
    with pytest.raises(ValueError, match="must be one of"):
        tg_to_df(DATA / 'this_is_a_label_file.TextGrid', parser=parser)

def test_tg_to_df_parser_invalid_checked_before_reading(tmp_path):
    """The `parser` value is validated before the file is opened."""
    with pytest.raises(ValueError, match="must be one of"):
        tg_to_df(tmp_path / 'does_not_exist.TextGrid', parser='nosuchparser')

@pytest.mark.parametrize('tgfile', PRAAT_READABLE)
def test_tg_to_df_parsers_agree(tgfile):
    """Both parsers return the same dataframes for well-formed textgrids."""
    pytest.importorskip('parselmouth')
    pydfs = tg_to_df(DATA / tgfile, parser='python')
    prdfs = tg_to_df(DATA / tgfile, parser='praat')
    assert len(pydfs) == len(prdfs)
    for pydf, prdf in zip(pydfs, prdfs):
        pd.testing.assert_frame_equal(pydf, prdf)

def test_tg_to_df_praat_tiersel_and_names():
    """`tiersel` and `names` behave the same under `parser='praat'`."""
    pytest.importorskip('parselmouth')
    kwargs = dict(tiersel=['word', 'phone'], names=['wrd', 'seg'])
    pydfs = tg_to_df(DATA / 'this_is_a_label_file.TextGrid', **kwargs)
    prdfs = tg_to_df(DATA / 'this_is_a_label_file.TextGrid',
                     parser='praat', **kwargs)
    assert [df.columns.tolist() for df in prdfs] == \
        [['t1', 't2', 'wrd'], ['t1', 't2', 'seg']]
    for pydf, prdf in zip(pydfs, prdfs):
        pd.testing.assert_frame_equal(pydf, prdf)

def test_tg_to_df_praat_point_tier():
    """Point tiers read through Praat have a `t1` column and no `t2`."""
    pytest.importorskip('parselmouth')
    stdf = tg_to_df(
        DATA / 'this_is_a_label_file.short.TextGrid',
        tiersel=['stimulus'], parser='praat'
    )[0]
    assert stdf.columns.tolist() == ['t1', 'stimulus']
    assert stdf['stimulus'].tolist() == ['1', '2', '3']

def test_read_textgrid_praat_structure():
    """`_read_textgrid_praat` returns the same structure as `_read_textgrid`."""
    pytest.importorskip('parselmouth')
    tgfile = DATA / 'this_is_a_label_file.short.TextGrid'
    pytiers = _read_textgrid(tgfile)
    prtiers = _read_textgrid_praat(tgfile)
    assert len(pytiers) == len(prtiers)
    for pyt, prt in zip(pytiers, prtiers):
        assert pyt.keys() == prt.keys()
        assert pyt['name'] == prt['name']
        assert pyt['class'] == prt['class']
        assert pyt['labels'] == prt['labels']
        assert isinstance(prt['start'], float)
        assert isinstance(prt['end'], float)
        assert (prt['start'], prt['end']) == (0.0, 1.6550793650793652)

def test_tg_to_df_praat_empty_interval_tier():
    """Praat repairs an interval tier that declares no intervals by supplying
    one empty interval; the pure-Python parser reads it as empty. This test
    records the difference rather than asserting either is correct."""
    pytest.importorskip('parselmouth')
    pydfs = tg_to_df(DATA / 'empty_tier.short.TextGrid', parser='python')
    prdfs = tg_to_df(DATA / 'empty_tier.short.TextGrid', parser='praat')
    assert [len(df) for df in pydfs] == [3, 0, 0, 4, 0]
    assert [len(df) for df in prdfs] == [3, 0, 1, 4, 0]
    assert prdfs[2]['empty_interval'].tolist() == ['']

def test_tg_to_df_praat_rejects_bad_label_count():
    """Praat refuses a textgrid whose contents run past its declared counts,
    where the pure-Python parser reads it."""
    parselmouth = pytest.importorskip('parselmouth')
    assert len(tg_to_df(DATA / 'ipa.TextGrid', tiersel=['phone'])[0]) == 9
    with pytest.raises(parselmouth.PraatError):
        tg_to_df(DATA / 'ipa.TextGrid', parser='praat.only')


#### tg_tiernames ####

ALL_FIXTURES = [p.name for p in sorted(DATA.glob('*.TextGrid'))]

@pytest.mark.parametrize('tgfile', ALL_FIXTURES)
def test_tg_tiernames_matches_full_read(tgfile):
    """`tg_tiernames` agrees with a full read of the same textgrid."""
    assert tg_tiernames(DATA / tgfile) == \
        tuple(t['name'] for t in _read_textgrid(DATA / tgfile))

def test_tg_tiernames_returns_tuple():
    """Names are returned as a tuple, in textgrid order."""
    names = tg_tiernames(DATA / 'this_is_a_label_file.short.TextGrid')
    assert isinstance(names, tuple)
    assert names == ('word', 'phone', 'stimulus')

def test_tg_tiernames_point_and_interval_tiers():
    """Point tiers are named alongside interval tiers."""
    assert tg_tiernames(DATA / 'empty_tier.long.TextGrid') == (
        'V1', 'empty_point_1', 'empty_interval', 'V2', 'empty_point_end'
    )

def test_tg_tiernames_empty_and_duplicate_names():
    """Unnamed tiers give '', and one entry is returned per tier."""
    names = tg_tiernames(DATA / 'empty_name.TextGrid')
    assert names == ('word', '', '')
    assert len(names) == len(_read_textgrid(DATA / 'empty_name.TextGrid'))

def test_tg_tiernames_utf_16():
    """Tier names are decoded using the textgrid's byte-order mark."""
    assert tg_tiernames(DATA / 'Turkmen_NA_20130919_G_3.TextGrid') == \
        ('word', 'gloss')

def test_tg_tiernames_multiline_labels():
    """Label content spanning lines does not disturb the tier names."""
    assert tg_tiernames(DATA / 'multiline.short.TextGrid') == ('multiline',)

def test_tg_tiernames_label_that_looks_like_a_tier_header(tmp_path):
    """A label whose text is 'IntervalTier' is written as a line identical to
    a tier header. The tier structure is walked rather than scanned for such
    lines, so the decoy is not mistaken for a tier."""
    tgfile = tmp_path / 'decoy.TextGrid'
    tgfile.write_text('''File type = "ooTextFile"
Object class = "TextGrid"

0
1
<exists>
1
"IntervalTier"
"real"
0
1
2
0
0.5
"IntervalTier"
0.5
1
"b"
''', encoding='utf-8')
    assert tg_tiernames(tgfile) == ('real',)
    [df] = tg_to_df(tgfile)
    assert df.columns.tolist() == ['t1', 't2', 'real']
    assert df['real'].tolist() == ['IntervalTier', 'b']

def test_tg_tiernames_usable_as_tiersel():
    """The returned names select tiers in `tg_to_df`."""
    tgfile = DATA / 'this_is_a_label_file.TextGrid'
    names = tg_tiernames(tgfile)
    dfs = tg_to_df(tgfile, tiersel=list(names))
    assert [df.columns[-1] for df in dfs] == list(names)

def test_tg_tiernames_bad_file(tmp_path):
    """A file that is not a textgrid raises `TextGridParseError`."""
    notatg = tmp_path / 'notatg.TextGrid'
    notatg.write_text('this is not\na textgrid at all\n')
    with pytest.raises(TextGridParseError):
        tg_tiernames(notatg)


def _scan(tgfile):
    """Return (names, walked) from the per-tier tier name scan."""
    lines, _ = tgmodule._read_lines(tgfile, None)
    if tgmodule._detect_format(lines, tgfile) == 'long':
        return tgmodule._scan_tiernames_long(lines)
    return tgmodule._scan_tiernames_short(lines)

@pytest.mark.parametrize('tgfile', ALL_FIXTURES)
def test_tiernames_scan_matches_full_read(tgfile):
    """The per-tier scan gives the same names as a full read, whichever
    tiers it skips and whichever it walks."""
    names, walked = _scan(DATA / tgfile)
    assert names == tuple(t['name'] for t in _read_textgrid(DATA / tgfile))
    assert all(0 <= i < len(names) for i in walked)
    assert tg_tiernames(DATA / tgfile) == names

def test_tiernames_scan_walks_only_bad_tiers():
    """Only the tiers whose declared counts do not hold are walked."""
    # 'multiline.short' has one tier, which declares 10 labels and holds 11.
    assert _scan(DATA / 'multiline.short.TextGrid') == (('multiline',), (0,))
    # 'ipa' declares 8 intervals on its phone tier (index 1) and holds 9;
    # the word and context tiers are skipped.
    assert _scan(DATA / 'ipa.TextGrid') == (('word', 'phone', 'context'), (1,))

def test_tiernames_scan_skips_df_to_tg_output(tmp_path):
    """Textgrids written by `df_to_tg` declare correct counts, so no tier is
    walked."""
    [wddf, phdf, stdf] = tg_to_df(
        DATA / 'this_is_a_label_file.short.TextGrid',
        tiersel=['word', 'phone', 'stimulus']
    )
    for tgtype in ('short', 'long'):
        outfile = tmp_path / f'counts.{tgtype}.TextGrid'
        df_to_tg(
            [wddf, phdf, stdf],
            tiercols=['word', 'phone', 'stimulus'],
            ts=[['t1', 't2'], ['t1', 't2'], ['t1', None]],
            tgtype=tgtype, outfile=outfile
        )
        assert _scan(outfile) == (('word', 'phone', 'stimulus'), ())

def _tiers_with_multiline(position):
    """Three interval tiers and a point tier, with multiline label content in
    the tier at `position` only."""
    def tier(name, multiline):
        n = 6
        return pd.DataFrame({
            't1': [i / 10 for i in range(n)],
            't2': [(i + 1) / 10 for i in range(n)],
            name: [f'{name} {i}\nline two' if multiline and i % 2 else f'{name}{i}'
                   for i in range(n)],
        })
    names = ['a', 'b', 'c']
    dfs = [tier(nm, i == position) for i, nm in enumerate(names)]
    ptdf = pd.DataFrame({'t1': [0.05, 0.25], 'pt': ['p', 'q']})
    return dfs + [ptdf], names + ['pt']

@pytest.mark.parametrize('tgtype', ['short', 'long'])
@pytest.mark.parametrize('position', [0, 1, 2])
def test_tiernames_scan_walks_multiline_tier_only(tgtype, position, tmp_path):
    """A tier with multiline labels is walked wherever it falls, and the
    tiers before and after it are still skipped."""
    dfs, names = _tiers_with_multiline(position)
    outfile = tmp_path / f'ml{position}.{tgtype}.TextGrid'
    df_to_tg(
        dfs, tiercols=names,
        ts=[['t1', 't2']] * 3 + [['t1', None]],
        tgtype=tgtype, outfile=outfile
    )
    assert _scan(outfile) == (tuple(names), (position,))
    assert tg_tiernames(outfile) == tuple(names)
    # The walked tier is read exactly as a full read reads it.
    full = tg_to_df(outfile)
    assert full[position][names[position]].str.contains('\n').sum() == 3

def test_tg_tiernames_exported_at_package_level():
    """`tg_tiernames` is reachable as `phonlab.tg_tiernames`."""
    import phonlab
    assert phonlab.tg_tiernames is tg_tiernames
    assert 'tg_tiernames' in phonlab.__all__


#### Parser fallback ####

class _ReaderCalls:
    """Fake readers standing in for the two parsers, so the fallback logic is
    tested without depending on what either real parser can read."""
    def __init__(self, monkeypatch, python=None, praat=None):
        self.calls = []
        for name, attr, outcome in (
            ('python', '_read_textgrid', python),
            ('praat', '_read_textgrid_praat', praat),
        ):
            monkeypatch.setattr(tgmodule, attr, self._reader(name, outcome))

    def _reader(self, name, outcome):
        def reader(tgfile):
            self.calls.append(name)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return reader

def _tiers(name):
    """A one-tier result whose tier name identifies the reader."""
    return [{'class': 'IntervalTier', 'name': name, 'start': 0.0, 'end': 1.0,
             'labels': [(0.0, 1.0, name)]}]

def _no_warnings():
    """A context in which any warning is an error."""
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter('error')
    return ctx

@pytest.mark.parametrize('parser', ['python', 'praat', 'python.only', 'praat.only'])
def test_fallback_named_parser_succeeds(parser, monkeypatch):
    """When the named parser succeeds, the other is not tried and nothing
    is warned about."""
    calls = _ReaderCalls(monkeypatch, python=_tiers('python'), praat=_tiers('praat'))
    name = parser.split('.')[0]
    ctx = _no_warnings()
    try:
        tiers = _read_textgrid_with('x.TextGrid', parser)
    finally:
        ctx.__exit__(None, None, None)
    assert calls.calls == [name]
    assert tiers[0]['name'] == name

@pytest.mark.parametrize('name, other', [('python', 'praat'), ('praat', 'python')])
def test_fallback_to_other_parser(name, other, monkeypatch):
    """When the named parser fails, the other is used, with a warning."""
    outcomes = {name: TextGridParseError('named failed'), other: _tiers(other)}
    calls = _ReaderCalls(monkeypatch, **outcomes)
    with pytest.warns(TextGridParserFallbackWarning, match=f"'{name}' parser could not read"):
        tiers = _read_textgrid_with('x.TextGrid', name)
    assert calls.calls == [name, other]
    assert tiers[0]['name'] == other

@pytest.mark.parametrize('name', ['python', 'praat'])
def test_fallback_only_suffix_raises_named_error(name, monkeypatch):
    """With '.only', the named parser's own error is raised and the other
    parser is never tried."""
    err = TextGridParseError('named failed')
    other = 'praat' if name == 'python' else 'python'
    calls = _ReaderCalls(monkeypatch, **{name: err, other: _tiers(other)})
    with pytest.raises(TextGridParseError) as excinfo:
        _read_textgrid_with('x.TextGrid', f'{name}.only')
    assert excinfo.value is err
    assert calls.calls == [name]

def test_fallback_both_fail(monkeypatch):
    """When both parsers fail, a `TextGridParseError` names both errors and
    has the named parser's error as its cause."""
    first = ValueError('python trouble')
    calls = _ReaderCalls(monkeypatch, python=first, praat=RuntimeError('praat trouble'))
    with pytest.raises(TextGridParseError, match='Neither parser') as excinfo:
        _read_textgrid_with('x.TextGrid', 'python')
    msg = str(excinfo.value)
    assert 'ValueError: python trouble' in msg
    assert 'RuntimeError: praat trouble' in msg
    assert excinfo.value.__cause__ is first
    assert calls.calls == ['python', 'praat']

def test_fallback_unavailable_raises_named_error(monkeypatch):
    """If the fallback parser cannot be imported, the named parser's own error
    is raised, as if there had been no fallback."""
    err = TextGridParseError('python failed')
    _ReaderCalls(monkeypatch, python=err, praat=ImportError('no parselmouth'))
    with pytest.raises(TextGridParseError) as excinfo:
        _read_textgrid_with('x.TextGrid', 'python')
    assert excinfo.value is err

def test_fallback_when_praat_not_installed(monkeypatch):
    """'praat' falls back to 'python' when parselmouth is missing, while
    'praat.only' raises the `ImportError`."""
    _ReaderCalls(monkeypatch, python=_tiers('python'), praat=ImportError('no parselmouth'))
    with pytest.warns(TextGridParserFallbackWarning):
        assert _read_textgrid_with('x.TextGrid', 'praat')[0]['name'] == 'python'
    with pytest.raises(ImportError):
        _read_textgrid_with('x.TextGrid', 'praat.only')

@pytest.mark.parametrize('name', ['python', 'praat'])
def test_fallback_not_used_for_oserror(name, monkeypatch):
    """A file that cannot be opened is not a parser failure: its `OSError` is
    raised and the other parser is not tried."""
    other = 'praat' if name == 'python' else 'python'
    calls = _ReaderCalls(monkeypatch, **{name: FileNotFoundError('gone'), other: _tiers(other)})
    with pytest.raises(FileNotFoundError):
        _read_textgrid_with('x.TextGrid', name)
    assert calls.calls == [name]

def test_fallback_oserror_from_fallback_parser(monkeypatch):
    """An `OSError` from the fallback parser is raised as it is. (Praat reports
    a missing file as a `PraatError`, so this is how a missing file surfaces
    under parser='praat'.)"""
    _ReaderCalls(monkeypatch, praat=RuntimeError('praat trouble'),
                 python=FileNotFoundError('gone'))
    with pytest.raises(FileNotFoundError):
        _read_textgrid_with('x.TextGrid', 'praat')

def test_fallback_tg_to_df_passes_parser_through(monkeypatch):
    """`tg_to_df` builds its dataframes from whichever parser succeeded."""
    _ReaderCalls(monkeypatch, python=TextGridParseError('no'), praat=_tiers('praat'))
    with pytest.warns(TextGridParserFallbackWarning):
        [df] = tg_to_df('x.TextGrid')
    assert df.columns.tolist() == ['t1', 't2', 'praat']
    assert df['praat'].tolist() == ['praat']

def test_tg_to_df_missing_file_default_parser(tmp_path):
    """A missing file raises `FileNotFoundError` under the default parser."""
    with pytest.raises(FileNotFoundError):
        tg_to_df(tmp_path / 'does_not_exist.TextGrid')

def test_tg_to_df_python_only_on_bad_file(tmp_path):
    """'python.only' raises the pure-Python parser's own error."""
    notatg = tmp_path / 'notatg.TextGrid'
    notatg.write_text('this is not\na textgrid at all\n')
    with pytest.raises(TextGridParseError, match='does not appear to be'):
        tg_to_df(notatg, parser='python.only')

def test_tg_to_df_python_only_matches_python():
    """'python.only' gives the same result as 'python' on a readable file."""
    tgfile = DATA / 'this_is_a_label_file.TextGrid'
    for a, b in zip(tg_to_df(tgfile, parser='python'),
                    tg_to_df(tgfile, parser='python.only')):
        pd.testing.assert_frame_equal(a, b)

def test_tg_to_df_praat_falls_back_on_praat_refusal():
    """Praat refuses 'ipa.TextGrid', so parser='praat' reads it with the
    pure-Python parser instead, and warns."""
    pytest.importorskip('parselmouth')
    with pytest.warns(TextGridParserFallbackWarning, match="'praat' parser could not read"):
        [phdf] = tg_to_df(DATA / 'ipa.TextGrid', tiersel=['phone'], parser='praat')
    assert len(phdf) == 9

def test_tg_to_df_praat_only_no_warning_on_success():
    """A successful 'praat.only' read issues no warning."""
    pytest.importorskip('parselmouth')
    ctx = _no_warnings()
    try:
        dfs = tg_to_df(DATA / 'this_is_a_label_file.TextGrid', parser='praat.only')
    finally:
        ctx.__exit__(None, None, None)
    assert len(dfs) == 3


#### tg_tiernames parser ####

class _NameReaderCalls:
    """Fake tier name readers standing in for the two parsers."""
    def __init__(self, monkeypatch, python=None, praat=None):
        self.calls = []
        self.codecs = []
        def pyreader(tg, codec=None):
            self.calls.append('python')
            self.codecs.append(codec)
            if isinstance(python, BaseException):
                raise python
            return python
        def prreader(tg):
            self.calls.append('praat')
            if isinstance(praat, BaseException):
                raise praat
            return praat
        monkeypatch.setattr(tgmodule, '_tiernames_python', pyreader)
        monkeypatch.setattr(tgmodule, '_tiernames_praat', prreader)

@pytest.mark.parametrize('parser', [
    'praat_short', 'Python', '', None, '.only', 'python.', 'praat.Only'
])
def test_tg_tiernames_parser_invalid(parser, tmp_path):
    """An unrecognized `parser` raises `ValueError` before the file is read."""
    with pytest.raises(ValueError, match="must be one of"):
        tg_tiernames(tmp_path / 'does_not_exist.TextGrid', parser=parser)

@pytest.mark.parametrize('parser', ['python', 'python.only'])
def test_tg_tiernames_python_parsers(parser):
    """'python' and 'python.only' give the pure-Python result."""
    assert tg_tiernames(DATA / 'this_is_a_label_file.TextGrid', parser=parser) \
        == ('phone', 'word', 'context')

@pytest.mark.parametrize('name, other', [('python', 'praat'), ('praat', 'python')])
def test_tg_tiernames_fallback(name, other, monkeypatch):
    """When the named parser fails, the other is used, with a warning."""
    calls = _NameReaderCalls(
        monkeypatch, **{name: TextGridParseError('named failed'), other: (other,)}
    )
    with pytest.warns(TextGridParserFallbackWarning, match=f"'{name}' parser could not read"):
        assert tg_tiernames('x.TextGrid', parser=name) == (other,)
    assert calls.calls == [name, other]

@pytest.mark.parametrize('name', ['python', 'praat'])
def test_tg_tiernames_only_suffix(name, monkeypatch):
    """With '.only', the named parser's error is raised and the other parser
    is never tried."""
    err = TextGridParseError('named failed')
    other = 'praat' if name == 'python' else 'python'
    calls = _NameReaderCalls(monkeypatch, **{name: err, other: (other,)})
    with pytest.raises(TextGridParseError) as excinfo:
        tg_tiernames('x.TextGrid', parser=f'{name}.only')
    assert excinfo.value is err
    assert calls.calls == [name]

def test_tg_tiernames_both_fail(monkeypatch):
    """When both parsers fail, a `TextGridParseError` names both errors."""
    first = TextGridParseError('python trouble')
    _NameReaderCalls(monkeypatch, python=first, praat=RuntimeError('praat trouble'))
    with pytest.raises(TextGridParseError, match='Neither parser') as excinfo:
        tg_tiernames('x.TextGrid')
    assert excinfo.value.__cause__ is first

def test_tg_tiernames_codec_reaches_python_parser(monkeypatch):
    """`codec` is passed to the pure-Python parser, and is not needed by the
    Praat parser."""
    calls = _NameReaderCalls(monkeypatch, python=('a',), praat=('b',))
    assert tg_tiernames('x.TextGrid', codec='latin-1') == ('a',)
    assert calls.codecs == ['latin-1']
    assert tg_tiernames('x.TextGrid', codec='latin-1', parser='praat.only') == ('b',)

def test_tg_tiernames_codec_real_file(capsys):
    """The codec is honored by the real parser: a conflicting BOM warns."""
    assert tg_tiernames(DATA / 'Turkmen_NA_20130919_G_3.TextGrid', codec='utf-8') \
        == ('word', 'gloss')
    assert 'overriding user-specified encoding utf-8' in capsys.readouterr().err

def test_tg_tiernames_missing_file(tmp_path):
    """A missing file raises `FileNotFoundError`, with no fallback."""
    with pytest.raises(FileNotFoundError):
        tg_tiernames(tmp_path / 'does_not_exist.TextGrid')

def test_fallback_warning_points_at_caller(monkeypatch):
    """The fallback warning is attributed to the line that called
    `tg_to_df` or `tg_tiernames`, not to a line inside phonlab."""
    _ReaderCalls(monkeypatch, python=TextGridParseError('no'), praat=_tiers('praat'))
    _NameReaderCalls(monkeypatch, python=TextGridParseError('no'), praat=('praat',))
    for call in (
        lambda: tg_to_df('x.TextGrid'),
        lambda: tg_tiernames('x.TextGrid'),
        lambda: _read_textgrid_with('x.TextGrid'),
    ):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            call()
        fallback = [r for r in record
                    if issubclass(r.category, TextGridParserFallbackWarning)]
        assert len(fallback) == 1
        assert fallback[0].filename == __file__

@pytest.mark.parametrize('tgfile', PRAAT_READABLE)
def test_tg_tiernames_parsers_agree(tgfile):
    """Both parsers give the same tier names for well-formed textgrids."""
    pytest.importorskip('parselmouth')
    assert tg_tiernames(DATA / tgfile, parser='praat.only') == \
        tg_tiernames(DATA / tgfile, parser='python.only')

def test_tg_tiernames_praat_falls_back_on_praat_refusal():
    """Praat refuses 'ipa.TextGrid', so parser='praat' reads its names with the
    pure-Python parser instead, and warns; 'praat.only' raises."""
    parselmouth = pytest.importorskip('parselmouth')
    with pytest.warns(TextGridParserFallbackWarning):
        assert tg_tiernames(DATA / 'ipa.TextGrid', parser='praat') == \
            ('word', 'phone', 'context')
    with pytest.raises(parselmouth.PraatError):
        tg_tiernames(DATA / 'ipa.TextGrid', parser='praat.only')


#### Fast 'praat' path ####

def _praat_like_textgrid(tmp_path, text, encoding, bom=b''):
    """A one-tier long format textgrid with label `text`, written with the
    given encoding, as Praat might write it."""
    df = pd.DataFrame({'t1': [0.0, 0.5], 't2': [0.5, 1.0], 'lab': [text, 'b']})
    tgstr = df_to_tg(df, 'lab', tgtype='long')
    outfile = tmp_path / f'praat-{encoding}.TextGrid'
    outfile.write_bytes(bom + tgstr.encode(encoding))
    return outfile

@pytest.mark.parametrize('encoding, bom', [
    ('utf-8', b''),
    ('latin-1', b''),
    ('utf-16-le', b'\xff\xfe'),
    ('utf-16-be', b'\xfe\xff'),
])
def test_read_praat_output_encodings(encoding, bom, tmp_path):
    """Praat's output is decoded whichever encoding Praat chose: a byte-order
    mark is trusted, and otherwise UTF-8 is tried before ISO Latin-1."""
    outfile = _praat_like_textgrid(tmp_path, 'b\xedt na\xefve', encoding, bom)
    [tier] = tgmodule._read_praat_output(outfile)
    assert [text for _, _, text in tier['labels']] == ['b\xedt na\xefve', 'b']

@pytest.mark.parametrize('encoding, bom', [
    ('utf-8', b''),
    ('latin-1', b''),
    ('utf-16-le', b'\xff\xfe'),
    ('utf-16-be', b'\xfe\xff'),
])
def test_read_praat_output_encodings_short(encoding, bom, tmp_path):
    """Praat's output is short format, which is decoded as the long format is."""
    df = pd.DataFrame({'t1': [0.0, 0.5], 't2': [0.5, 1.0],
                       'lab': ['b\xedt na\xefve', 'b']})
    outfile = tmp_path / f'praat-short-{encoding}.TextGrid'
    outfile.write_bytes(bom + df_to_tg(df, 'lab', tgtype='short').encode(encoding))
    [tier] = tgmodule._read_praat_output(outfile)
    assert [text for _, _, text in tier['labels']] == ['b\xedt na\xefve', 'b']

def test_read_textgrid_praat_saves_short_format(monkeypatch):
    """Praat is asked for its short text format, whatever the input format."""
    pytest.importorskip('parselmouth')
    real = tgmodule._import_pcall()
    commands = []
    def recording(*args):
        commands.extend(a for a in args if isinstance(a, str) and a.endswith('...'))
        return real(*args)
    monkeypatch.setattr(tgmodule, '_import_pcall', lambda: recording)
    formats = []
    real_output = tgmodule._read_praat_output
    def check_format(praatfile):
        lines, _ = tgmodule._read_lines(praatfile, None)
        formats.append(tgmodule._detect_format(lines, praatfile))
        return real_output(praatfile)
    monkeypatch.setattr(tgmodule, '_read_praat_output', check_format)
    for tgfile in ('this_is_a_label_file.long.TextGrid',
                   'this_is_a_label_file.short.TextGrid'):
        _read_textgrid_praat(DATA / tgfile)
    assert 'Save as short text file...' in commands
    assert formats == ['short', 'short']

def test_read_praat_output_ascii(tmp_path):
    """ASCII output, which is also valid UTF-8, is read as such."""
    outfile = _praat_like_textgrid(tmp_path, 'plain', 'ascii')
    [tier] = tgmodule._read_praat_output(outfile)
    assert tier['labels'][0][2] == 'plain'

# Textgrids that Praat reads. The empty tier fixtures are included here
# because Praat's repair of the empty interval tier must survive the write.
PRAAT_READS = PRAAT_READABLE + [
    'empty_tier.short.TextGrid', 'empty_tier.long.TextGrid'
]

@pytest.mark.parametrize('tgfile', PRAAT_READS)
def test_read_textgrid_praat_matches_per_label_calls(tgfile):
    """Having Praat write the textgrid out gives exactly what retrieving each
    label with its own call to Praat gives."""
    pytest.importorskip('parselmouth')
    fast = _read_textgrid_praat(DATA / tgfile)
    slow = tgmodule._read_textgrid_praat_calls(DATA / tgfile)
    assert len(fast) == len(slow)
    for f, s in zip(fast, slow):
        assert (f['name'], f['class']) == (s['name'], s['class'])
        assert f['labels'] == s['labels']

def test_read_textgrid_praat_call_count(monkeypatch):
    """The whole textgrid is retrieved in a few calls to Praat, however many
    labels it has."""
    pytest.importorskip('parselmouth')
    real = tgmodule._import_pcall()
    calls = []
    def counting(*args):
        calls.append(args[1] if len(args) > 1 and not isinstance(args[0], str)
                     else args[0])
        return real(*args)
    monkeypatch.setattr(tgmodule, '_import_pcall', lambda: counting)
    tiers = _read_textgrid_praat(DATA / 'Turkmen_NA_20130919_G_3.TextGrid')
    assert sum(len(t['labels']) for t in tiers) == 338
    assert len(calls) <= 3, calls


#### Line splitting ####

@pytest.mark.parametrize('raw, expected', [
    (b'a\nb\n', ['a\n', 'b\n']),
    (b'a\nb', ['a\n', 'b\n']),                  # no final newline
    (b'a\r\nb\r\n', ['a\n', 'b\n']),             # CRLF
    (b'a\rb\r', ['a\n', 'b\n']),                 # CR only
    (b'a\r\nb\rc\n', ['a\n', 'b\n', 'c\n']),      # mixed
    (b'a\n\n', ['a\n', '\n']),                  # blank last line kept once
    (b'\xef\xbb\xbfa\nb\n', ['a\n', 'b\n']),     # UTF-8 BOM dropped
    (b'a\x0bb\x0cc\n', ['a\x0bb\x0cc\n']),        # only CR and LF split lines
    (b'', []),
])
def test_read_lines(raw, expected, tmp_path):
    """Files are split into lines that each end with a single newline."""
    tgfile = tmp_path / 'lines.txt'
    tgfile.write_bytes(raw)
    lines, codec = tgmodule._read_lines(tgfile, None)
    assert lines == expected
    assert codec == 'utf-8'


#### Label structure ####

@pytest.mark.parametrize('tgfile', [
    'this_is_a_label_file.short.TextGrid', 'this_is_a_label_file.long.TextGrid'
])
def test_labels_are_tuples(tgfile):
    """Labels are (t1, t2, text) tuples, with t2 None for point tiers."""
    tiers = _read_textgrid(DATA / tgfile)
    for tier in tiers:
        for label in tier['labels']:
            assert type(label) is tuple and len(label) == 3
            t1, t2, text = label
            assert type(t1) is float and type(text) is str
            if tier['class'] == 'IntervalTier':
                assert type(t2) is float and t2 > t1
            else:
                assert t2 is None
    word, phone, stim = tiers
    assert word['labels'][1] == (0.04531977534891905, 0.4207853308004855, 'This')
    assert stim['labels'][0] == (0.020579796888932116, None, '1')
