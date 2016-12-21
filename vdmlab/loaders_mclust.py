import numpy as np
import vdmlab as vdm


def load_mclust_header(filename):
    """Loads a mclust .t tetrode file.

    Parameters
    ----------
    filename: str

    Returns
    -------
    header: list
        Contains byte strings

    """
    f = open(filename, 'rb')

    file_contents = f.read()

    # From mclust documentation, the file contains a header.
    # This header begins with %%BEGINHEADER and ends with %%ENDHEADER.
    # Here we separate that header from the timestamps (data).
    lines = file_contents.split(b'\n')

    header_begin = b'%%BEGINHEADER'
    header_end = b'%%ENDHEADER'
    header_start_idx = None
    header_stop_idx = None

    for i, line in enumerate(lines):
        if line.strip().startswith(header_begin):
            header_start_idx = i
        if line.strip().startswith(header_end):
            header_stop_idx = i

    if header_start_idx is None or header_stop_idx is None:
        raise IOError("Header not found in .t file for " + filename)

    header = lines[header_start_idx:header_stop_idx + 1]

    f.close()

    return header



