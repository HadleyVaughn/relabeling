import numpy as np
from itertools import combinations_with_replacement, product
from tqdm import tqdm

def is_uniform_array(cnt):
    """
    Determines which of an array of distributions are uniform.
    :param cnt: numpy array of outcome counts
    :return: a boolean mask of valid faces
    """
    is_strictly_uniform = cnt[:,0].reshape((cnt.shape[0],1)) == cnt[:,1:]
    is_zero = cnt[:,1:] == 0
    is_uniform = np.all(np.logical_or(is_strictly_uniform, is_zero), axis=1)
    return is_uniform


def is_uniform(cnt):
    return np.all(cnt == cnt[0])


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
    possible_dice = np.asarray(list(combinations_with_replacement(r, n)), dtype=np.uint8)
    print("Generating dice distributions")
    possible_dice_unq, possible_dice_cnt = count_unique_rows(possible_dice)
    print("Finding valid dice")
    valid_dice_mask = is_uniform_array(possible_dice_cnt)
    dice = possible_dice[valid_dice_mask]
    num_dice = len(dice)
    print("Found", num_dice, "dice")
    print("Finding valid dice sets")
    dice_indecies = np.arange(len(dice), dtype=np.uint8).reshape((-1, 1))
    dice_sets = dice_indecies
    dice_set_outcomes = dice
    for i in range(1, m):
        num_sets = len(dice_sets)
        print('Considering adding a die to', num_sets, 'sets of', i, 'dice')
        dice_sets = np.concatenate((np.repeat(dice_sets, num_dice, axis=0), np.tile(dice_indecies, (num_sets, 1))), axis=1)
        dice_set_outcomes = np.repeat(dice_set_outcomes, num_dice, axis=0)
        new_outcomes = np.empty((dice_set_outcomes.shape[0], n**(i+1)), dtype=np.uint8)
        nonuniform = []
        for u, set in enumerate(dice_sets):
            new_outcomes[u] = np.add.outer(dice_set_outcomes[u][dice_set_outcomes[u] != 0], dice[set[-1]]).reshape((-1))
            unq, cnt = np.unique(new_outcomes[u], return_counts=True)
            if not is_uniform(cnt):
                nonuniform.append(u)
            elif i == m - 1 and min == m and max == m * n and max - min + 1 == len(unq):
                return dice[set]
            if (u % 10000 == 0):
                print('Checked', u, 'new sets')
        dice_sets = np.delete(dice_sets, nonuniform, axis=0)
        dice_set_outcomes = new_outcomes
