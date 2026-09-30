from text_utils import shout, slugify

def test_shout():
    assert shout("hi") == "HI!"

def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"

def test_slugify_punctuation():
    assert slugify("  Rock & Roll!  ") == "rock-roll"
