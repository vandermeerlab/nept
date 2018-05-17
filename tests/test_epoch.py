import numpy as np
import pytest
import nept


def test_epoch_duration():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    assert np.allclose(epoch.durations, np.array([1., 0.6, 0.4]))


def test_epoch_stops():
    start_times = np.array([[0.0],
                            [0.9],
                            [1.6]])
    durations = np.array([1., 0.6, 0.4])
    epoch = nept.Epoch(start_times, durations)

    assert np.allclose(epoch.stops, np.array([1., 1.5, 2.]))


def test_epoch_index():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)
    sliced_epoch = epoch[:2]

    assert np.allclose(sliced_epoch.starts, np.array([0., 0.9]))
    assert np.allclose(sliced_epoch.stops, np.array([1., 1.5]))


def test_epoch_sort():
    times = np.array([[1.6, 2.0],
                      [0.0, 1.0],
                      [0.9, 1.5]])
    epoch = nept.Epoch(times)

    assert np.allclose(epoch.starts, np.array([0., 0.9, 1.6]))
    assert np.allclose(epoch.stops, np.array([1., 1.5, 2.]))


def test_epoch_sortlist():
    start_times = [0.9, 0.0, 1.6]
    durations = [0.6, 1.0, 0.4]
    epoch = nept.Epoch(start_times, durations)

    assert np.allclose(epoch.starts, np.array([0.0, 0.9, 1.6]))
    assert np.allclose(epoch.stops, np.array([1., 1.5, 2.]))


def test_epoch_reshape():
    times = np.array([[0.0, 0.9, 1.6], [1.0, 1.5, 2.0]])
    epoch = nept.Epoch(times)

    assert np.allclose(epoch.time.shape, (3, 2))


def test_epoch_centers():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.4],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)
    assert np.allclose(epoch.centers, np.array([0.5, 1.15, 1.8]))


def test_epoch_too_many_parameters():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])

    durations = np.array([1., 0.6, 0.4])

    with pytest.raises(ValueError) as excinfo:
        epoch = nept.Epoch(times, durations)

    assert str(excinfo.value) == 'duration not allowed when using start and stop times'


def test_epoch_intersect_case1():
    times_1 = np.array([[0.0, 1.0],
                        [1.1, 1.5],
                        [1.6, 2.0]])
    epoch_1 = nept.Epoch(times_1)

    times_2 = np.array([[1.55, 1.8]])
    epoch_2 = nept.Epoch(times_2)

    intersects = epoch_1.intersect(epoch_2)

    assert np.allclose(intersects.starts, np.array([1.6]))
    assert np.allclose(intersects.stops, np.array([1.8]))


def test_epoch_overlaps_case1_bounds():
    times_1 = np.array([[0.0, 1.0],
                        [1.1, 1.5],
                        [1.6, 2.0]])
    epoch_1 = nept.Epoch(times_1)

    times_2 = np.array([[1.55, 1.8]])
    epoch_2 = nept.Epoch(times_2)

    overlaps = epoch_1.overlaps(epoch_2)

    assert np.allclose(overlaps.starts, np.array([1.55]))
    assert np.allclose(overlaps.stops, np.array([1.8]))


def test_epoch_intersect_case2():
    times_1 = np.array([[0.0, 1.0],
                        [1.1, 1.5],
                        [1.6, 2.0]])
    epoch_1 = nept.Epoch(times_1)

    times_2 = np.array([[1.2, 1.6]])
    epoch_2 = nept.Epoch(times_2)

    intersects = epoch_1.intersect(epoch_2)

    assert np.allclose(intersects.starts, np.array([1.2]))
    assert np.allclose(intersects.stops, np.array([1.5]))


def test_epoch_intersect_empty():
    times_1 = np.array([[0.0, 1.0],
                        [1.1, 1.5],
                        [1.6, 2.0]])
    epoch_1 = nept.Epoch(times_1)

    times_2 = np.array([[], []])
    epoch_2 = nept.Epoch(times_2)

    intersects = epoch_1.intersect(epoch_2)

    assert intersects.time.size == 0


def test_epoch_overlaps_case2_bounds():
    times_1 = np.array([[0.0, 1.0],
                        [1.1, 1.5],
                        [1.6, 2.0]])
    epoch_1 = nept.Epoch(times_1)

    times_2 = np.array([[1.2, 1.6]])
    epoch_2 = nept.Epoch(times_2)

    overlaps = epoch_1.overlaps(epoch_2)

    assert np.allclose(overlaps.starts, np.array([1.2]))
    assert np.allclose(overlaps.stops, np.array([1.6]))


def test_epoch_overlaps_empty():
    times_1 = np.array([[0.0, 1.0],
                        [1.1, 1.5],
                        [1.6, 2.0]])
    epoch_1 = nept.Epoch(times_1)

    times_2 = np.array([[], []])
    epoch_2 = nept.Epoch(times_2)

    overlaps = epoch_1.overlaps(epoch_2)

    assert overlaps.time.size == 0


def test_epoch_intersect_case3():
    times_a = np.array([[1.0, 2.0]])
    epoch_a = nept.Epoch(times_a)

    times_b = np.array([[0.0, 3.0]])
    epoch_b = nept.Epoch(times_b)

    intersects = epoch_a.intersect(epoch_b)

    assert np.allclose(intersects.starts, np.array([1.0]))
    assert np.allclose(intersects.stops, np.array([2.0]))


def test_epoch_overlaps_case3_bounds():
    times_a = np.array([[1.0, 2.0]])
    epoch_a = nept.Epoch(times_a)

    times_b = np.array([[0.0, 3.0]])
    epoch_b = nept.Epoch(times_b)

    overlaps = epoch_a.overlaps(epoch_b)

    assert np.allclose(overlaps.starts, np.array([0.0]))
    assert np.allclose(overlaps.stops, np.array([3.0]))


def test_epoch_intersect_case4():
    times_a = np.array([[1.0, 2.0]])
    epoch_a = nept.Epoch(times_a)

    times_b = np.array([[1.1, 1.9]])
    epoch_b = nept.Epoch(times_b)

    intersects = epoch_a.intersect(epoch_b)

    assert np.allclose(intersects.starts, np.array([1.1]))
    assert np.allclose(intersects.stops, np.array([1.9]))


def test_epoch_overlaps_case4_bounds():
    times_a = np.array([[1.0, 2.0]])
    epoch_a = nept.Epoch(times_a)

    times_b = np.array([[1.1, 1.9]])
    epoch_b = nept.Epoch(times_b)

    overlaps = epoch_a.overlaps(epoch_b)

    assert np.allclose(overlaps.starts, np.array([1.1]))
    assert np.allclose(overlaps.stops, np.array([1.9]))


def test_epoch_intersect_case5():
    times_a = np.array([[1.5, 2.5]])
    epoch_a = nept.Epoch(times_a)

    times_b = np.array([[1.5, 2.5]])
    epoch_b = nept.Epoch(times_b)

    intersects = epoch_a.intersect(epoch_b)

    assert np.allclose(intersects.starts, np.array([1.5]))
    assert np.allclose(intersects.stops, np.array([2.5]))


def test_epoch_overlaps_case5_bounds():
    times_a = np.array([[1.5, 2.5]])
    epoch_a = nept.Epoch(times_a)

    times_b = np.array([[1.5, 2.5]])
    epoch_b = nept.Epoch(times_b)

    overlaps = epoch_a.overlaps(epoch_b)

    assert np.allclose(overlaps.starts, np.array([1.5]))
    assert np.allclose(overlaps.stops, np.array([2.5]))


def test_epoch_intersect_multiple():
    times_a = np.array([[1.0, 2.0],
                        [4.0, 5.0],
                        [6.0, 7.0],
                        [8.0, 9.0]])
    epoch_a = nept.Epoch(times_a)

    times_b = np.array([[0.5, 1.7],
                        [4.3, 5.0],
                        [5.1, 7.2],
                        [8.2, 8.4]])
    epoch_b = nept.Epoch(times_b)

    intersects = epoch_a.intersect(epoch_b)

    assert np.allclose(intersects.starts, np.array([1.0, 4.3, 6.0, 8.2]))
    assert np.allclose(intersects.stops, np.array([1.7, 5.0, 7.0, 8.4]))


def test_epoch_overlaps_multiple_bounds():
    times_a = np.array([[1.0, 2.0],
                        [4.0, 5.0],
                        [6.0, 7.0],
                        [8.0, 9.0]])
    epoch_a = nept.Epoch(times_a)

    times_b = np.array([[0.5, 1.7],
                        [4.3, 5.0],
                        [5.1, 7.2],
                        [8.2, 8.4]])
    epoch_b = nept.Epoch(times_b)

    overlaps = epoch_a.overlaps(epoch_b)

    assert np.allclose(overlaps.starts, np.array([0.5, 4.3, 5.1, 8.2]))
    assert np.allclose(overlaps.stops, np.array([1.7, 5.0, 7.2, 8.4]))


def test_epoch_intersect_multiple2():
    times_a = np.array([[1.0, 2.0],
                        [4.0, 5.0],
                        [6.0, 7.0],
                        [8.0, 9.0]])
    epoch_a = nept.Epoch(times_a)

    times_b = np.array([[1.0, 2.0],
                        [4.0, 6.2],
                        [5.9, 6.2],
                        [7.8, 9.3]])
    epoch_b = nept.Epoch(times_b)

    intersects = epoch_a.intersect(epoch_b)

    assert np.allclose(intersects.starts, np.array([1.0, 4.0, 6.0, 8.0]))
    assert np.allclose(intersects.stops, np.array([2.0, 5.0, 6.2, 9.0]))


def test_epoch_overlaps_multiple2_bounds():
    times_a = np.array([[0.0, 2.0],
                        [4.0, 5.0],
                        [6.0, 7.0],
                        [8.0, 9.0]])
    epoch_a = nept.Epoch(times_a)

    times_b = np.array([[1.0, 2.0],
                        [4.0, 6.2],
                        [5.9, 6.2],
                        [7.8, 9.3]])
    epoch_b = nept.Epoch(times_b)

    overlaps = epoch_a.overlaps(epoch_b)

    print(overlaps.starts)
    print(overlaps.stops)

    assert np.allclose(overlaps.starts, np.array([1.0, 4.0, 7.8]))
    assert np.allclose(overlaps.stops, np.array([2.0, 6.2, 9.3]))


def test_epoch_no_intersect():
    times_1 = np.array([[0.0, 1.0],
                        [0.9, 1.5],
                        [1.6, 2.0]])
    epoch_1 = nept.Epoch(times_1)

    times_2 = np.array([[1.5, 1.6]])
    epoch_2 = nept.Epoch(times_2)

    intersects = epoch_1.intersect(epoch_2)

    assert np.allclose(intersects.starts, np.array([]))
    assert np.allclose(intersects.stops, np.array([]))


def test_epoch_merge_overlap():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])

    epoch = nept.Epoch(times)

    merged = epoch.merge()
    assert np.allclose(merged.starts, np.array([0., 1.6]))
    assert np.allclose(merged.stops, np.array([1.5, 2.0]))


def test_epoch_merge_with_gap():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])

    epoch = nept.Epoch(times)

    merged = epoch.merge(gap=0.1)
    assert np.allclose(merged.starts, np.array([0.]))
    assert np.allclose(merged.stops, np.array([2.0]))


def test_epoch_merge_negative_gap():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])

    epoch = nept.Epoch(times)

    with pytest.raises(ValueError) as excinfo:
        merged = epoch.merge(gap=-0.1)

    assert str(excinfo.value) == "gap cannot be negative"


def test_epoch_merge_far_stop():
    times = np.array([[0.0, 10.0],
                      [1.0, 3.0],
                      [2.0, 5.0],
                      [11.0, 12.0]])

    epoch = nept.Epoch(times)
    merged = epoch.merge()
    assert np.allclose(merged.starts, np.array([0.0, 11.0]))
    assert np.allclose(merged.stops, np.array([10.0, 12.0]))


def test_epoch_merge_no_overlap():
    times = np.array([[1.1, 2.3],
                      [3.5, 5.2],
                      [11.1, 12.0]])

    epoch = nept.Epoch(times)
    merged = epoch.merge()

    assert np.allclose(merged.starts, np.array([1.1, 3.5, 11.1]))
    assert np.allclose(merged.stops, np.array([2.3, 5.2, 12.0]))


def test_epoch_merge_no_overlap_gap():
    times = np.array([[1.0, 2.5],
                      [3.5, 5.0],
                      [11.0, 12.0]])

    epoch = nept.Epoch(times)
    merged = epoch.merge(gap=0.2)

    assert np.allclose(merged.starts, np.array([1.0, 3.5, 11.0]))
    assert np.allclose(merged.stops, np.array([2.5, 5.0, 12.0]))


def test_epoch_merge_unordered_stops():
    times = np.array([[-0.2, 0.8],
                      [0., 0.8],
                      [1., 3.6],
                      [2.2, 3.],
                      [4., 4.4]])

    epoch = nept.Epoch(times)

    merged = epoch.merge()
    assert np.allclose(merged.starts, np.array([-0.2, 1., 4.]))
    assert np.allclose(merged.stops, np.array([0.8, 3.6, 4.4]))


def test_epoch_expand_both():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    resized = epoch.expand(0.5)

    assert np.allclose(resized.starts, np.array([-0.5, 0.4, 1.1]))
    assert np.allclose(resized.stops, np.array([1.5, 2.0, 2.5]))


def test_epoch_expand_start():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    resized = epoch.expand(0.5, direction='start')

    assert np.allclose(resized.starts, np.array([-0.5, 0.4, 1.1]))
    assert np.allclose(resized.stops, np.array([1.0, 1.5, 2.0]))


def test_epoch_expand_stop():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    resized = epoch.expand(0.5, direction='stop')

    assert np.allclose(resized.starts, np.array([0.0, 0.9, 1.6]))
    assert np.allclose(resized.stops, np.array([1.5, 2.0, 2.5]))


def test_epoch_expand_incorrect_direction_input():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    with pytest.raises(ValueError) as excinfo:
        resized = epoch.expand(0.5, direction='all')

    assert str(excinfo.value) == "direction must be 'both', 'start', or 'stop'"


def test_epoch_shrink():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    shrinked = epoch.shrink(0.1)

    assert np.allclose(shrinked.starts, np.array([0.1, 1.0, 1.7]))
    assert np.allclose(shrinked.stops, np.array([0.9, 1.4, 1.9]))


def test_epoch_shrink_toobig_both():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    with pytest.raises(ValueError) as excinfo:
        shrinked = epoch.shrink(2.)

    assert str(excinfo.value) == "shrink amount too large"


def test_epoch_shrink_toobig_single():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    with pytest.raises(ValueError) as excinfo:
        shrinked = epoch.shrink(1., direction='start')

    assert str(excinfo.value) == "shrink amount too large"


def test_epoch_join():
    times_1 = np.array([[0.0, 1.0],
                        [0.9, 1.5],
                        [1.6, 2.0]])
    epoch_1 = nept.Epoch(times_1)

    times_2 = np.array([[1.8, 2.5]])
    epoch_2 = nept.Epoch(times_2)

    union = epoch_1.join(epoch_2)

    assert np.allclose(union.starts, np.array([0.0, 0.9, 1.6, 1.8]))
    assert np.allclose(union.stops, np.array([1.0, 1.5, 2.0, 2.5]))


def test_epoch_start_stop():
    epoch = nept.Epoch(np.array([[721.9412, 900.0],
                                [1000.0, 1027.1]]))

    assert np.allclose(epoch.start, 721.9412)
    assert np.allclose(epoch.stop, 1027.1)


def test_epoch_contain_true():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    assert epoch.contains(0.5)


def test_epoch_contain_not():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    assert not epoch.contains(1.55)


def test_epoch_isempty():
    epoch = nept.Epoch([[], []])

    assert epoch.isempty


def test_epoch_notempty():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = nept.Epoch(times)

    assert not epoch.isempty


def test_epoch_incorrect_time_duration():
    times = np.array([[0.0],
                      [0.9],
                      [1.6]])

    with pytest.raises(ValueError) as excinfo:
        epoch = nept.Epoch(times, duration=0.3)

    assert str(excinfo.value) == "must have same number of time and duration samples"


def test_epoch_ndim():
    times = np.ones((2, 3, 4))

    with pytest.raises(ValueError) as excinfo:
        epoch = nept.Epoch(times)

    assert str(excinfo.value) == "time cannot have more than 2 dimensions"


def test_epoch_stop_before_start():
    times = np.array([[0.0, 1.0],
                      [0.9, 0.2],
                      [1.6, 2.0]])

    with pytest.raises(ValueError) as excinfo:
        epoch = nept.Epoch(times)

    assert str(excinfo.value) == "start must be less than stop"


def test_epoch_mismatch_start_stop():
    with pytest.raises(ValueError) as excinfo:
        nept.Epoch([[0.0, 1.0], [0.5]])

    assert str(excinfo.value) == "must have the same number of start and stop times"
