import numpy as np

import vdmlab as vdm


def test_idx_in_pos():
    position = vdm.Position([[0, 1, 2], [9, 7, 5]], [10, 11, 12])
    pos = position[1]

    assert np.allclose(pos.x, 1)
    assert np.allclose(pos.y, 7)
    assert np.allclose(pos.time, 11)
