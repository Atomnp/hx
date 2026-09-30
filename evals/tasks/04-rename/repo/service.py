from users import get_usr

def greeting(uid):
    name = get_usr(uid)
    return f"hello {name}" if name else "who?"
