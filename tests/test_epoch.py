import collections

import nept
import numpy as np
import pytest


def test_epoch_duration():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    assert epoch.n_epochs == 3
    assert np.allclose(epoch.durations, np.array([1.0, 0.6, 0.4]))


def test_epoch_stops():
    starts = np.array([[0.0], [0.9], [1.6]])
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    assert epoch.n_epochs == 3
    assert np.allclose(epoch.stops, np.array([1.0, 1.5, 2.0]))


def test_epoch_index():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)
    sliced_epoch = epoch[:2]

    assert sliced_epoch.n_epochs == 2
    assert np.allclose(sliced_epoch.starts, np.array([0.0, 0.9]))
    assert np.allclose(sliced_epoch.stops, np.array([1.0, 1.5]))


def test_epoch_sort():
    starts = [1.6, 0.0, 0.9]
    stops = [2.0, 1.0, 1.5]
    epoch = nept.Epoch(starts, stops)

    assert epoch.n_epochs == 3
    assert np.allclose(epoch.starts, np.array([0.0, 0.9, 1.6]))
    assert np.allclose(epoch.stops, np.array([1.0, 1.5, 2.0]))


def test_epoch_sortlist():
    starts = [0.9, 0.0, 1.6]
    stops = [1.5, 1.0, 2.0]
    epoch = nept.Epoch(starts, stops)

    assert epoch.n_epochs == 3
    assert np.allclose(epoch.starts, np.array([0.0, 0.9, 1.6]))
    assert np.allclose(epoch.stops, np.array([1.0, 1.5, 2.0]))


def test_epoch_reshape():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    assert np.allclose(epoch.time.shape, (3, 2))


def test_epoch_centers():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.4, 2.0]
    epoch = nept.Epoch(starts, stops)

    assert epoch.n_epochs == 3
    assert np.allclose(epoch.centers, np.array([0.5, 1.15, 1.8]))


def test_epoch_too_many_parameters():
    starts = np.array([[0.0, 1.0], [0.9, 1.5], [1.6, 2.0]])

    stops = np.array([1.0, 0.6, 0.4])

    with pytest.raises(ValueError) as excinfo:
        epoch = nept.Epoch(starts, stops)

    assert str(excinfo.value) == "time cannot have more than 1 dimension."


def test_epoch_intersect_case1():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [1.55]
    stops2 = [1.8]
    epoch_2 = nept.Epoch(starts2, stops2)

    intersects = epoch_1.intersect(epoch_2)

    assert intersects.n_epochs == 1
    assert np.allclose(intersects.starts, np.array([1.6]))
    assert np.allclose(intersects.stops, np.array([1.8]))


def test_epoch_overlaps_case1_bounds():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [1.55]
    stops2 = [1.8]
    epoch_2 = nept.Epoch(starts2, stops2)

    overlaps = epoch_1.overlaps(epoch_2)

    assert overlaps.n_epochs == 1
    assert np.allclose(overlaps.starts, np.array([1.55]))
    assert np.allclose(overlaps.stops, np.array([1.8]))


def test_epoch_intersect_case2():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [1.2]
    stops2 = [1.6]
    epoch_2 = nept.Epoch(starts2, stops2)

    intersects = epoch_1.intersect(epoch_2)

    assert intersects.n_epochs == 1
    assert np.allclose(intersects.starts, np.array([1.2]))
    assert np.allclose(intersects.stops, np.array([1.5]))


def test_epoch_intersect_empty():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = []
    stops2 = []
    epoch_2 = nept.Epoch(starts2, stops2)

    intersects = epoch_1.intersect(epoch_2)

    assert intersects.isempty


def test_excludes_case1():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [1.55]
    stops2 = [1.8]
    epoch_2 = nept.Epoch(starts2, stops2)

    excludes = epoch_1.excludes(epoch_2)

    assert excludes.n_epochs == 3
    assert np.allclose(excludes.starts, np.array([0.0, 1.1, 1.8]))
    assert np.allclose(excludes.stops, np.array([1.0, 1.5, 2.0]))


def test_excludes_case2():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [1.2]
    stops2 = [1.6]
    epoch_2 = nept.Epoch(starts2, stops2)

    excludes = epoch_1.excludes(epoch_2)

    assert excludes.n_epochs == 3
    assert np.allclose(excludes.starts, np.array([0.0, 1.1, 1.6]))
    assert np.allclose(excludes.stops, np.array([1.0, 1.2, 2.0]))


def test_excludes_case3():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [1.2]
    stops2 = [1.3]
    epoch_2 = nept.Epoch(starts2, stops2)

    excludes = epoch_1.excludes(epoch_2)

    assert excludes.n_epochs == 4
    assert np.allclose(excludes.starts, np.array([0.0, 1.1, 1.3, 1.6]))
    assert np.allclose(excludes.stops, np.array([1.0, 1.2, 1.5, 2.0]))


def test_excludes_case4():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [1.1]
    stops2 = [1.3]
    epoch_2 = nept.Epoch(starts2, stops2)

    excludes = epoch_1.excludes(epoch_2)

    assert excludes.n_epochs == 3
    assert np.allclose(excludes.starts, np.array([0.0, 1.3, 1.6]))
    assert np.allclose(excludes.stops, np.array([1.0, 1.5, 2.0]))


def test_excludes_case5():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [0.9]
    stops2 = [1.3]
    epoch_2 = nept.Epoch(starts2, stops2)

    excludes = epoch_1.excludes(epoch_2)

    assert excludes.n_epochs == 3
    assert np.allclose(excludes.starts, np.array([0.0, 1.3, 1.6]))
    assert np.allclose(excludes.stops, np.array([0.9, 1.5, 2.0]))


def test_excludes_case6():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [2.1]
    stops2 = [2.5]
    epoch_2 = nept.Epoch(starts2, stops2)

    excludes = epoch_1.excludes(epoch_2)

    assert excludes.n_epochs == 3
    assert np.allclose(excludes.starts, np.array([0.0, 1.1, 1.6]))
    assert np.allclose(excludes.stops, np.array([1.0, 1.5, 2.0]))


def test_excludes_empty():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [0.0]
    stops2 = [2.5]
    epoch_2 = nept.Epoch(starts2, stops2)

    excludes = epoch_1.excludes(epoch_2)

    assert excludes.time.size == 0


def test_epoch_overlaps_case2_bounds():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = [1.2]
    stops2 = [1.6]
    epoch_2 = nept.Epoch(starts2, stops2)

    overlaps = epoch_1.overlaps(epoch_2)

    assert overlaps.n_epochs == 1
    assert np.allclose(overlaps.starts, np.array([1.2]))
    assert np.allclose(overlaps.stops, np.array([1.6]))


def test_epoch_overlaps_empty():
    starts1 = [0.0, 1.1, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    starts2 = []
    stops2 = []
    epoch_2 = nept.Epoch(starts2, stops2)

    overlaps = epoch_1.overlaps(epoch_2)

    assert overlaps.isempty


def test_epoch_intersect_case3():
    epoch_a = nept.Epoch([1.0], [2.0])
    epoch_b = nept.Epoch([0.0], [3.0])

    intersects = epoch_a.intersect(epoch_b)

    assert intersects.n_epochs == 1
    assert np.allclose(intersects.starts, np.array([1.0]))
    assert np.allclose(intersects.stops, np.array([2.0]))


def test_epoch_overlaps_case3_bounds():
    epoch_a = nept.Epoch([0.0], [2.0])
    epoch_b = nept.Epoch([0.0], [3.0])

    overlaps = epoch_a.overlaps(epoch_b)

    assert overlaps.n_epochs == 1
    assert np.allclose(overlaps.starts, np.array([0.0]))
    assert np.allclose(overlaps.stops, np.array([3.0]))


def test_epoch_intersect_case4():
    epoch_a = nept.Epoch([1.0], [2.0])
    epoch_b = nept.Epoch([1.1], [1.9])

    intersects = epoch_a.intersect(epoch_b)

    assert intersects.n_epochs == 1
    assert np.allclose(intersects.starts, np.array([1.1]))
    assert np.allclose(intersects.stops, np.array([1.9]))


def test_epoch_overlaps_case4_bounds():
    epoch_a = nept.Epoch([1.0], [2.0])
    epoch_b = nept.Epoch([1.1], [1.9])

    overlaps = epoch_a.overlaps(epoch_b)

    assert overlaps.n_epochs == 1
    assert np.allclose(overlaps.starts, np.array([1.1]))
    assert np.allclose(overlaps.stops, np.array([1.9]))


def test_epoch_intersect_case5():
    epoch_a = nept.Epoch([1.5], [2.5])
    epoch_b = nept.Epoch([1.5], [2.5])

    intersects = epoch_a.intersect(epoch_b)

    assert intersects.n_epochs == 1
    assert np.allclose(intersects.starts, np.array([1.5]))
    assert np.allclose(intersects.stops, np.array([2.5]))


def test_epoch_overlaps_case5_bounds():
    epoch_a = nept.Epoch([1.5], [2.5])
    epoch_b = nept.Epoch([1.5], [2.5])

    overlaps = epoch_a.overlaps(epoch_b)

    assert overlaps.n_epochs == 1
    assert np.allclose(overlaps.starts, np.array([1.5]))
    assert np.allclose(overlaps.stops, np.array([2.5]))


def test_epoch_intersect_multiple():
    starts_a = [1.0, 4.0, 6.0, 8.0]
    stops_a = [2.0, 5.0, 7.0, 9.0]
    epoch_a = nept.Epoch(starts_a, stops_a)

    starts_b = [0.5, 4.3, 5.1, 8.2]
    stops_b = [1.7, 5.0, 7.2, 8.4]
    epoch_b = nept.Epoch(starts_b, stops_b)

    intersects = epoch_a.intersect(epoch_b)

    assert intersects.n_epochs == 4
    assert np.allclose(intersects.starts, np.array([1.0, 4.3, 6.0, 8.2]))
    assert np.allclose(intersects.stops, np.array([1.7, 5.0, 7.0, 8.4]))


def test_epoch_overlaps_multiple_bounds():
    starts_a = [1.0, 4.0, 6.0, 8.0]
    stops_a = [2.0, 5.0, 7.0, 9.0]
    epoch_a = nept.Epoch(starts_a, stops_a)

    starts_b = [0.5, 4.3, 5.1, 8.2]
    stops_b = [1.7, 5.0, 7.2, 8.4]
    epoch_b = nept.Epoch(starts_b, stops_b)

    overlaps = epoch_a.overlaps(epoch_b)

    assert overlaps.n_epochs == 4
    assert np.allclose(overlaps.starts, np.array([0.5, 4.3, 5.1, 8.2]))
    assert np.allclose(overlaps.stops, np.array([1.7, 5.0, 7.2, 8.4]))


def test_epoch_intersect_multiple2():
    starts_a = [1.0, 4.0, 6.0, 8.0]
    stops_a = [2.0, 5.0, 7.0, 9.0]
    epoch_a = nept.Epoch(starts_a, stops_a)

    starts_b = [1.0, 4.0, 5.9, 7.8]
    stops_b = [2.0, 6.2, 6.2, 9.3]
    epoch_b = nept.Epoch(starts_b, stops_b)

    intersects = epoch_a.intersect(epoch_b)

    assert intersects.n_epochs == 4
    assert np.allclose(intersects.starts, np.array([1.0, 4.0, 6.0, 8.0]))
    assert np.allclose(intersects.stops, np.array([2.0, 5.0, 6.2, 9.0]))


def test_epoch_overlaps_multiple2_bounds():
    starts_a = [0.0, 4.0, 6.0, 8.0]
    stops_a = [2.0, 5.0, 7.0, 9.0]
    epoch_a = nept.Epoch(starts_a, stops_a)

    starts_b = [1.0, 4.0, 5.9, 7.8]
    stops_b = [2.0, 6.2, 6.2, 9.3]
    epoch_b = nept.Epoch(starts_b, stops_b)

    overlaps = epoch_a.overlaps(epoch_b)

    print(overlaps.starts)
    print(overlaps.stops)

    assert overlaps.n_epochs == 3
    assert np.allclose(overlaps.starts, np.array([1.0, 4.0, 7.8]))
    assert np.allclose(overlaps.stops, np.array([2.0, 6.2, 9.3]))


def test_epoch_no_intersect():
    starts1 = [0.0, 0.9, 1.6]
    stops1 = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts1, stops1)

    epoch_2 = nept.Epoch([1.5], [1.6])

    intersects = epoch_1.intersect(epoch_2)

    assert intersects.n_epochs == 0
    assert np.allclose(intersects.starts, np.array([]))
    assert np.allclose(intersects.stops, np.array([]))


def test_epoch_merge_overlap():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    merged = epoch.merge()

    assert merged.n_epochs == 2
    assert np.allclose(merged.starts, np.array([0.0, 1.6]))
    assert np.allclose(merged.stops, np.array([1.5, 2.0]))


def test_epoch_merge_with_gap():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    merged = epoch.merge(gap=0.1)

    assert merged.n_epochs == 1
    assert np.allclose(merged.starts, np.array([0.0]))
    assert np.allclose(merged.stops, np.array([2.0]))


def test_epoch_merge_negative_gap():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    with pytest.raises(ValueError) as excinfo:
        merged = epoch.merge(gap=-0.1)

    assert str(excinfo.value) == "gap cannot be negative"


def test_epoch_merge_far_stop():
    starts = [0.0, 1.0, 2.0, 11.0]
    stops = [10.0, 3.0, 5.0, 12.0]
    epoch = nept.Epoch(starts, stops)

    merged = epoch.merge()

    assert merged.n_epochs == 2
    assert np.allclose(merged.starts, np.array([0.0, 11.0]))
    assert np.allclose(merged.stops, np.array([10.0, 12.0]))


def test_epoch_merge_no_overlap():

    starts = [1.1, 3.5, 11.1]
    stops = [2.3, 5.2, 12.0]

    epoch = nept.Epoch(starts, stops)
    merged = epoch.merge()

    assert merged.n_epochs == 3
    assert np.allclose(merged.starts, np.array([1.1, 3.5, 11.1]))
    assert np.allclose(merged.stops, np.array([2.3, 5.2, 12.0]))


def test_epoch_merge_no_overlap_gap():
    starts = [1.0, 3.5, 11.0]
    stops = [2.5, 5.0, 12.0]
    epoch = nept.Epoch(starts, stops)

    merged = epoch.merge(gap=0.2)

    assert merged.n_epochs == 3
    assert np.allclose(merged.starts, np.array([1.0, 3.5, 11.0]))
    assert np.allclose(merged.stops, np.array([2.5, 5.0, 12.0]))


def test_epoch_merge_unordered_stops():
    starts = [-0.2, 0.0, 1.0, 2.2]
    stops = [0.8, 0.8, 3.6, 3.0]
    epoch = nept.Epoch(starts, stops)

    merged = epoch.merge()

    assert merged.n_epochs == 2
    assert np.allclose(merged.starts, np.array([-0.2, 1.0]))
    assert np.allclose(merged.stops, np.array([0.8, 3.6]))


def test_epoch_merge_mult_unordered_stops():
    starts = [1.0, 3.0, 4.0, 6.0, 9.0]
    stops = [3.0, 4.0, 8.0, 7.0, 10.0]
    epoch = nept.Epoch(starts, stops)

    merged = epoch.merge()

    assert merged.n_epochs == 2
    assert np.allclose(merged.starts, np.array([1.0, 9.0]))
    assert np.allclose(merged.stops, np.array([8.0, 10.0]))


def test_epoch_expand_both():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    resized = epoch.expand(0.5)

    assert resized.n_epochs == 3
    assert np.allclose(resized.starts, np.array([-0.5, 0.4, 1.1]))
    assert np.allclose(resized.stops, np.array([1.5, 2.0, 2.5]))


def test_epoch_expand_start():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    resized = epoch.expand(0.5, direction="start")

    assert resized.n_epochs == 3
    assert np.allclose(resized.starts, np.array([-0.5, 0.4, 1.1]))
    assert np.allclose(resized.stops, np.array([1.0, 1.5, 2.0]))


def test_epoch_expand_stop():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    resized = epoch.expand(0.5, direction="stop")

    assert resized.n_epochs == 3
    assert np.allclose(resized.starts, np.array([0.0, 0.9, 1.6]))
    assert np.allclose(resized.stops, np.array([1.5, 2.0, 2.5]))


def test_epoch_expand_incorrect_direction_input():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    with pytest.raises(ValueError) as excinfo:
        resized = epoch.expand(0.5, direction="all")

    assert str(excinfo.value) == "direction must be 'both', 'start', or 'stop'"


def test_epoch_shrink():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    shrinked = epoch.shrink(0.1)

    assert shrinked.n_epochs == 3
    assert np.allclose(shrinked.starts, np.array([0.1, 1.0, 1.7]))
    assert np.allclose(shrinked.stops, np.array([0.9, 1.4, 1.9]))


def test_epoch_shrink_toobig_both():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    with pytest.raises(ValueError) as excinfo:
        shrinked = epoch.shrink(2.0)

    assert str(excinfo.value) == "shrink amount too large"


def test_epoch_shrink_toobig_single():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    with pytest.raises(ValueError) as excinfo:
        shrinked = epoch.shrink(1.0, direction="start")

    assert str(excinfo.value) == "shrink amount too large"


def test_epoch_join():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch_1 = nept.Epoch(starts, stops)

    epoch_2 = nept.Epoch([1.8], [2.5])

    union = epoch_1.join(epoch_2)

    assert union.n_epochs == 4
    assert np.allclose(union.starts, np.array([0.0, 0.9, 1.6, 1.8]))
    assert np.allclose(union.stops, np.array([1.0, 1.5, 2.0, 2.5]))


def test_epoch_start_stop():
    epoch = nept.Epoch([721.9412, 900.0], [1000.0, 1027.1])

    assert np.allclose(epoch.start, 721.9412)
    assert np.allclose(epoch.stop, 1027.1)


def test_epoch_contains_true():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    assert epoch.contains(0.5)


def test_epoch_contains_not():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    assert not epoch.contains(1.55)


def test_epoch_contains_edge():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    assert not epoch.contains(2.0, edge=False)


def test_epoch_isempty():
    epoch = nept.Epoch([], [])

    assert epoch.isempty


def test_epoch_notempty():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    assert not epoch.isempty


def test_epoch_ndim():
    starts = np.ones((2, 3, 4))
    stops = np.ones((2, 3, 4)) * 2

    with pytest.raises(ValueError) as excinfo:
        epoch = nept.Epoch(starts, stops)

    assert str(excinfo.value) == "time cannot have more than 1 dimension."


def test_epoch_stop_before_start():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 0.2, 2.0]

    with pytest.raises(ValueError) as excinfo:
        epoch = nept.Epoch(starts, stops)

    assert str(excinfo.value) == "start must be less than stop"


def test_epoch_mismatch_start_stop():
    with pytest.raises(ValueError) as excinfo:
        nept.Epoch([0.0, 1.0], [0.5])

    assert str(excinfo.value) == "must have the same number of start and stop times"


def test_epoch_iter():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    epoch_iter = iter(epoch)

    assert isinstance(epoch_iter, collections.Iterable)


def test_epoch_using_iter():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    epoch_iter = iter(epoch)
    this_instance = next(epoch_iter)

    assert this_instance.starts == np.array([0.0])
    assert this_instance.stops == np.array([1.0])


def test_epoch_time_slice_simple():
    starts = [0.0, 0.9, 1.6]
    stops = [1.0, 1.5, 2.0]
    epoch = nept.Epoch(starts, stops)

    sliced_epoch = epoch.time_slice(1, 2)

    assert sliced_epoch.n_epochs == 2
    assert np.allclose(sliced_epoch.starts, np.array([1.0, 1.6]))
    assert np.allclose(sliced_epoch.stops, np.array([1.5, 2.0]))


def test_epoch_time_slice_overlap():
    starts = [0.0, 0.9, 1.6]
    stops = [1.1, 1.5, 3.0]
    epoch = nept.Epoch(starts, stops)

    sliced_epoch = epoch.time_slice(1, 2)

    assert sliced_epoch.n_epochs == 3
    assert np.allclose(sliced_epoch.starts, np.array([1.0, 1.0, 1.6]))
    assert np.allclose(sliced_epoch.stops, np.array([1.1, 1.5, 2.0]))


def test_epoch_time_slice_empty():
    starts = [0.9, 1.6]
    stops = [1.5, 3.0]
    epoch = nept.Epoch(starts, stops)

    sliced_epoch = epoch.time_slice(0.1, 0.5)

    assert sliced_epoch.n_epochs == 0
    assert np.allclose(sliced_epoch.starts, np.array([]))
    assert np.allclose(sliced_epoch.stops, np.array([]))


def test_epoch_time_slice_cutoff():
    starts = [0.9, 1.5]
    stops = [1.6, 3.0]
    epoch = nept.Epoch(starts, stops)

    sliced_epoch = epoch.time_slice(0.1, 1.0)

    assert sliced_epoch.n_epochs == 1
    assert np.allclose(sliced_epoch.starts, np.array([0.9]))
    assert np.allclose(sliced_epoch.stops, np.array([1.0]))


def test_epoch_time_slice_one_epoch():
    epoch = nept.Epoch(starts=[1.0], stops=[2.0])

    sliced_epoch = epoch.time_slice(1.1, 1.9)
    print(sliced_epoch.starts, sliced_epoch.stops)

    assert sliced_epoch.n_epochs == 1
    assert np.allclose(sliced_epoch.starts, np.array([1.1]))
    assert np.allclose(sliced_epoch.stops, np.array([1.9]))
