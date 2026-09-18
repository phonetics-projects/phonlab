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

import pandas as pd
import pytest

from phonlab.utils.textgrid import (
    TextGridParseError, detect_encoding, read_textgrid, read_textgrid_praat
)
from phonlab.utils.tidy import df_to_tg, tg_to_df

DATA = Path(__file__).parent / 'data'

def label_at(df, t, col):
    """Return the label in `col` whose interval contains time `t`."""
    hits = df[(df['t1'] <= t) & (df['t2'] > t)]
    assert len(hits) == 1
    return hits[col].iloc[0]

def tiernames(tg):
    """Return the tier names of a textgrid as a tuple."""
    return tuple(t['name'] for t in read_textgrid(tg))

#### Reading: format detection and tier structure ####

def test_praat_long():
    """A long format textgrid is read, and its format detected."""
    tiers = read_textgrid(DATA / 'this_is_a_label_file.long.TextGrid')
    assert len(tiers) == 3
    assert tiernames(DATA / 'this_is_a_label_file.long.TextGrid') == \
        ('word', 'phone', 'stimulus')

def test_praat_short():
    """A short format textgrid is read, and its format detected."""
    tiers = read_textgrid(DATA / 'this_is_a_label_file.short.TextGrid')
    assert len(tiers) == 3
    assert tiernames(DATA / 'this_is_a_label_file.short.TextGrid') == \
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
    assert tiernames(DATA / tgfile) == (
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
    tiers = read_textgrid(DATA / 'ipa.TextGrid')
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
    assert tiernames(DATA / 'empty_name.TextGrid') == ('word', '', '')
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
    read_textgrid(DATA / 'Turkmen_NA_20130919_G_3.TextGrid', codec='utf-8')
    err = capsys.readouterr().err
    assert 'overriding user-specified encoding utf-8' in err
    assert 'utf_16_be' in err

def test_praat_no_warn_without_bom(capsys):
    """No warning is issued when there is no byte-order mark to conflict with."""
    read_textgrid(DATA / 'utf8_no_BOM.TextGrid', codec='utf-8')
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
    assert tiernames(outfile) == ('segment',)

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
    assert tiernames(outfile) == (
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

@pytest.mark.parametrize('parser', ['praat_short', 'Python', '', None])
def test_tg_to_df_parser_invalid(parser):
    """An unrecognized `parser` value raises a `ValueError`."""
    with pytest.raises(ValueError, match="must be 'python' or 'praat'"):
        tg_to_df(DATA / 'this_is_a_label_file.TextGrid', parser=parser)

def test_tg_to_df_parser_invalid_checked_before_reading(tmp_path):
    """The `parser` value is validated before the file is opened."""
    with pytest.raises(ValueError, match="must be 'python' or 'praat'"):
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
    """`read_textgrid_praat` returns the same structure as `read_textgrid`."""
    pytest.importorskip('parselmouth')
    tgfile = DATA / 'this_is_a_label_file.short.TextGrid'
    pytiers = read_textgrid(tgfile)
    prtiers = read_textgrid_praat(tgfile)
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
    pytest.importorskip('parselmouth')
    assert len(tg_to_df(DATA / 'ipa.TextGrid', tiersel=['phone'])[0]) == 9
    with pytest.raises(Exception):
        tg_to_df(DATA / 'ipa.TextGrid', parser='praat')
