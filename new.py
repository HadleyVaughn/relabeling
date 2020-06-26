import numpy as np
from itertools import combinations_with_replacement, product


def is_uniform(cnt):
    """
    Determines if a distribution is uniform.
    :param cnt: numpy array of outcome counts
    :return: a boolean mask of valid faces
    """
    is_strictly_uniform = cnt[:,0].reshape((cnt.shape[0],1)) == cnt[:,1:]
    is_zero = cnt[:,1:] == 0
    is_uniform = np.all(np.logical_or(is_strictly_uniform, is_zero), axis=1)
    return is_uniform


def is_filled(unq):
    """
    Determines if a distribution is filled.
    :param unq: numpy array of unique outcomes
    :return: a boolean mask of valid faces
    """
    min = unq[:,0]
    max = np.max(unq, axis=1)
    total = np.count_nonzero(unq, axis=1)
    return max - min + 1 == total


def is_standard(m, n, unq, cnt):
    """
    Determines if a distribution is filled.
    :param m: number of dice
    :param n: number of faces
    :param unq: numpy array of unique outcomes
    :param cnt: numpy array of outcome counts
    :return: a boolean mask of valid faces
    """
    min = unq[:,0]
    max = np.max(unq, axis=1)
    return np.logical_and( np.logical_and(min == m, max == m * n), np.logical_and(is_filled(unq), is_uniform(cnt)))


def count_unique_rows(arr):
    """
    Returns the unique elements and counts for each row (axis=1)
    :param arr: an array of things to count
    :return: a tuple containing two 2d-arrays, both zero padded.
       The first contains the unique elements for each die.
       The second contains the counts for each unique element in the first array for each die.
    """
    weight = 1j*np.linspace(0, arr.shape[1], arr.shape[0], endpoint=False)
    b = arr + weight[:, np.newaxis]
    u, ind, c = np.unique(b, return_index=True, return_counts=True)
    unq = np.zeros_like(arr).reshape((arr.shape[0], -1))
    cnt = np.zeros_like(arr).reshape((arr.shape[0], -1))
    np.put(unq, ind, arr.flat[ind])
    np.put(cnt, ind, c)
    return unq, cnt


def relabel(m, n, r):
    """
    Finds a standard-sums relabeling
    :param m: the number of dice
    :param n: the number of faces
    :param r: a *unique* array of possible faces
    :return:
    """
    min = m
    max = m * n
    print("Generating all possible dice")
    possible_dice = np.asarray(list(combinations_with_replacement(r, n)))
    print("Generating dice distributions")
    possible_dice_unq, possible_dice_cnt = count_unique_rows(possible_dice)
    print("Finding valid dice")
    valid_dice_mask = is_uniform(possible_dice_cnt)
    dice = possible_dice[valid_dice_mask]
    dice_cnt = possible_dice_cnt[valid_dice_mask]
    num_dice = len(dice)
    print("Found", num_dice, "dice")
    print("Finding valid dice sets")
    dice_indecies = np.arange(len(dice)).reshape((-1, 1))
    dice_sets = dice_indecies
    dice_sets_outcome = dice
    dice_sets_unq = None
    dice_sets_cnt = dice_cnt
    for i in range(1, m):
        num_sets = len(dice_sets)
        r_sets = np.repeat(dice_sets, num_dice, axis=0)
        r_outcomes = np.repeat(dice_sets_outcome, num_dice, axis=0)
        r_dice = np.tile(dice, (num_sets, 1))
        print('Considering adding a die to', num_sets, 'sets of', i + 1, 'dice')
        print('> Calculating dice outcomes')
        dice_sets_outcome = np.add.outer(r_outcomes, r_dice).diagonal(0,2,0).T.reshape((num_sets * num_dice, -1))
        new_dice = np.tile(dice_indecies, (num_sets, 1))
        dice_sets = np.concatenate((r_sets, new_dice), axis=1)
        print('> Counting unique dice outcomes')
        dice_sets_unq, dice_sets_cnt = count_unique_rows(dice_sets_outcome)
        print('> Updating search')
        nonuniform = np.where(np.logical_not(is_uniform(dice_sets_cnt)))
        dice_sets = np.delete(dice_sets, nonuniform, axis=0)
        dice_sets_outcome = np.delete(dice_sets_outcome, nonuniform,  axis=0)
        dice_sets_unq = np.delete(dice_sets_unq, nonuniform,  axis=0)
        dice_sets_cnt = np.delete(dice_sets_cnt, nonuniform,  axis=0)
    print('Done')
    std_mask = is_standard(m, n, dice_sets_unq, dice_sets_cnt)
    return dice[dice_sets[std_mask]]