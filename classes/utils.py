def return_none():
    return None


def none_filter(old, new):
    """Return new values, replacing None values with corresponding old values"""
    nest = False
    if not (type(old)==type(new) and isinstance(old, (tuple, list))):  # if not both tuple or list, nest in list
        old, new = [old], [new]
        nest = True
    for i, (o, n) in enumerate(zip(old, new)):
        if n is not None:
            old[i] = n
    if nest:
        old = old[0]
    return old