from service import greeting
import users

def test_greeting():
    assert greeting(1) == "hello ada"

def test_new_name_exists():
    assert users.get_user(2) == "linus"
