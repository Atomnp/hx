import pytest

from hx import __version__
from hx.cli import build_parser


def test_version_flag(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])
    assert __version__ in capsys.readouterr().out


def test_verbose_flag():
    assert build_parser().parse_args(["-v"]).verbose
    assert not build_parser().parse_args([]).verbose
