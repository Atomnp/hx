from hx import __version__
from hx.cli import main


def test_main_runs(capsys):
    assert main([]) == 0
    assert __version__ in capsys.readouterr().out
