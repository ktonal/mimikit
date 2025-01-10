from mimikit.extract.label_filter import label_filter
import numpy as np
from assertpy import assert_that
import pytest


def test_should_extend_repetition_on_the_edges():
    given_labels = np.r_[0, 0, 1, 2, 3, 4, 4]
    given_min_rep = 2
    expected_result = np.r_[0, 0, 0, 0, 4, 4, 4]

    result = label_filter(given_labels, given_min_rep, relabel_output=False)

    assert_that(np.all(result == expected_result)).is_true()


def test_should_extend_edges_and_replace_single_labels_with_undecidable_label():
    given_labels = np.r_[0, 0, 1, 2, 3, 4, 5, 5]
    given_min_rep = 2
    # if min_rep was 3, the -1 in the middle would be absorbed by 0 and 5
    expected_result = np.r_[0, 0, 0, -1, -1, 5, 5, 5]

    result = label_filter(given_labels, given_min_rep, relabel_output=False)

    assert_that(np.all(result == expected_result)).is_true()


def test_should_extend_edges_and_not_replace_single_labels_without_undecidable_label():
    given_labels = np.r_[0, 0, 1, 2, 3, 4, 5, 5]
    given_min_rep = 2
    # if min_rep was 3, the -1 in the middle would be absorbed by 0 and 5
    expected_result = np.r_[0, 0, 0, 0, 5, 5, 5, 5]

    result = label_filter(given_labels, given_min_rep, label_undecidable=False, relabel_output=False)

    assert_that(np.all(result == expected_result)).is_true()


def test_should_replace_undecidable_with_minus_one():
    given_labels = np.r_[0, 1, 2, 1, 2, 0]
    given_min_rep = 2
    expected_result = np.r_[-1, -1, -1, -1, -1, -1]

    result = label_filter(given_labels, given_min_rep, relabel_output=False)

    assert_that(np.all(result == expected_result)).is_true()


def test_should_replace_with_surrounding_elem_without_undecidable_label():
    given_labels = np.r_[0, 1, 2, 1, 2, 0]
    given_min_rep = 2
    expected_result = np.r_[1, 1, 1, 2, 2, 2]

    result = label_filter(given_labels, given_min_rep, label_undecidable=False, relabel_output=False)

    assert_that(np.all(result == expected_result)).is_true()


def test_should_return_input_if_undecidable():
    given_labels = np.r_[0, 1, 2, 3, 1, 2, 3, 0]
    given_min_rep = 2
    expected_result = given_labels

    result = label_filter(given_labels, given_min_rep, label_undecidable=False, relabel_output=False)

    assert_that(np.all(result == expected_result)).is_true()


def test_should_fallback_to_global_counts():
    given_labels = np.r_[0, 1, 2, 1, 2]
    given_min_rep = 2
    expected_result = np.r_[1, 1, -1, -1, -1]

    result = label_filter(given_labels, given_min_rep, relabel_output=False)

    assert_that(np.all(result == expected_result)).is_true()


def test_should_handle_edges_correctly():
    given_labels = np.r_[0, 0, 1, 1, 1]
    given_min_rep = 3
    expected_result = np.r_[1, 1, 1, 1, 1]

    result = label_filter(given_labels, given_min_rep, relabel_output=False)

    assert_that(np.all(result == expected_result)).is_true()