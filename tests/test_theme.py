import matplotlib as mpl

from bondviz import theme


def test_apply_mpl_style_sets_dark_background():
    mpl.rcParams["figure.facecolor"] = "white"  # ensure a known starting point
    theme.apply_mpl_style()
    assert mpl.rcParams["figure.facecolor"] == theme.BG
    assert mpl.rcParams["axes.facecolor"] == theme.PANEL
    assert mpl.rcParams["axes.grid"] is True


def test_palette_constants_exist():
    for name in ("BG", "PANEL", "ACCENT", "TEXT", "MUTED"):
        assert getattr(theme, name)
