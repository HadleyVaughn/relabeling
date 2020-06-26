from itertools import combinations_with_replacement
from colorama import Fore, Back, Style
from tqdm import tqdm
import numpy as np
import math

def can_relabel(m, n):
    """Determines if a relabeling if m n-sided dice is possible"""
    return n ** m % ((m * (n - 1)) + 1) == 0


def compute_dist(dice):
    """Computes the distribution over a set of arbitrary dice"""
    v = dice[0]
    for die in dice[1:]:
        v = np.add.outer(v, die)
    return dict(zip(*np.unique(v, return_counts=True)))

def compute_dist_and_test(dice):
    """Computes the distribution over a set of arbitrary dice"""
    u = dice[0]
    for die in dice[1:]:
        u, count = np.unique(np.add.outer(u, die), return_counts=True)
        if not all([c == count[0] for c in count[1:]]):
            return False, None
    return True, u

def is_uniform(dist):
    """Determines if a distribution is uniform"""
    c = list(dist.values())[0]
    for k in dist.keys():
        if dist[k] != c:
            return False
    return True


def is_filled(dist):
    """Determines if a distribution is filled"""
    keys = dist.keys()
    for i in range(min(keys), max(keys)):
        if i not in keys:
            return False
    return True


def is_stupid(faces):
    return 1 == len(set(faces))

def relabel(m, n, r=False, stupid=False, target_dist=is_uniform, dice_dist=is_uniform, q=0):
    """
    Finds relabeling for a set of m n-sided dice.
    'r' is the list of possible face values (defaults to [1 ... n])
    when 'stupid' is false we wont generate any stupid dice with all the same face
    """
    if not r:
        r = range(1, n + 1)

    solutions = []
    is_valid_face = lambda faces: dice_dist(compute_dist([faces])) and (stupid or not is_stupid(faces))
    not_all_stupid = lambda dice: not all([is_stupid(faces) for faces in dice])
    print("Generating valid die")
    all_faces = map(np.asarray, filter(is_valid_face, combinations_with_replacement(r, n)))
    #all_faces = map(np.asarray, combinations_with_replacement(r, n))
    print("Generating combinations of dice")
    all_dice = filter(not_all_stupid, combinations_with_replacement(all_faces, m))
    for dice in all_dice:
        dist = compute_dist(dice)
        if target_dist(dist):
            solutions.append(dice)
            if len(solutions) == q and q > 0:
                return solutions
    return solutions

def relabel_std(m, n, r=None, q=1):
    """
    Finds *standard* relabeling for a set of m n-sided dice.
    """
    if r is None:
        r = range(1, n*m)
    solutions = []
    is_valid_face = lambda faces: is_uniform(compute_dist([faces])) and (1 not in compute_dist([faces]) or n % compute_dist([faces])[1] == 0)
    is_valid_set = lambda dice: not all([is_stupid(faces) for faces in dice]) \
                                and any([min(r) in faces for faces in dice]) \
                                and any([max(r) in faces for faces in dice])
    all_faces = map(np.asarray, filter(is_valid_face, combinations_with_replacement(r, n)))
    all_dice = filter(is_valid_set, combinations_with_replacement(all_faces, m))
    for dice in tqdm(all_dice):
        uniform, items = compute_dist_and_test(dice)
        if uniform and all([i in items for i in range(items[0], items[-1])]) and items[0] == m and items[-1] == m*n:
            solutions.append(([list(faces) for faces in dice], list(items)))
            if len(solutions) == q and q > 0:
                return solutions
    return solutions

def auto_relabel_std(m, n):
    if not can_relabel(m, n):
        return [], []
    for i in range(m, m * n):
        print(i)
        solutions = relabel_std(m, n, r=range(1, i + 1))
        if len(solutions) != 0:
            return solutions[0]


# function for GCD
def GCD(c, b):
    if b == 0:
        return c
    return GCD(b, c % b)


# Fucnction return smallest + ve
# integer that holds condition
# A ^ k(mod N ) = 1
def multiplicative_order(q, n):
    if GCD(q, n) != 1:
        return -1

    # result store power of A that rised
    # to the power N-1
    result = 1

    k = 1
    while k < n:

        # modular arithmetic
        result = (result * q) % n

        # return smallest + ve integer
        if result == 1:
            return k

            # increment power
        k = k + 1

    return -1

def dice_relabel_num(n, q, p):
    """
    Returns [m1, m2] where each are possible numbers of dice for a relabellings
    :param n: The number of faces
    :param p: A prime
    :param q: A prime such that p * q = n and p != q
    """
    x = 0
    y = 1
    return [
        (((n**x) * (q**(multiplicative_order(q,(n-1))*y)))-1)%(n-1) == 0,
        (((n**x) * (p**(multiplicative_order(p,(n-1))*y)))-1)%(n-1) == 0
    ]

def stupid_die_min(n, p, q):
    """
    Returns minimum number of stupid dice for distinct primes p and q
    :param n: The number of faces
    :param p: A prime
    :param q: A prime such that p * q = n and p != q
    :return:
    """
    x = 0
    y = 1
    r = dice_relabel_num(n, q, p)
    return [
        r[0] - (2 * x) - (multiplicative_order(q, n-1) * y),
        r[1] - (2 * x) - (multiplicative_order(p, n-1) * y)
    ]

def stupid_die_max(n, p, q):
    """
    Returns maximum number of stupid dice for distinct primes p and q
    :param n: The number of faces
    :param p: A prime
    :param q: A prime such that p * q = n and p != q
    :return:
    """
    x = 0
    y = 1
    r = dice_relabel_num(n, q, p)
    return [
        r[0] - x - (multiplicative_order(q, n-1) * y),
        r[1] - x - (multiplicative_order(p, n-1) * y)
    ]

def gen_primes(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n//3 + (n%6==2), dtype=np.bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]

def semiprimesto(n):
    """ Generates semiprimes from the primes from 2 to n"""
    primes = gen_primes(n)
    prime_combinations = combinations_with_replacement(primes, 2)
    for p, q in prime_combinations:
        if p != q:
            yield (p*q, p, q)

def prime_ex(w, p=2, c=0):
    for q in gen_primes(w):
        if q > c:
            m = multiplicative_order(p, p*q - 1)
            if (p*q-1)*m == p**m - 1:
                yield (q, m)


# Python program to check if the input number is prime or not

# prime numbers are greater than 1
def check_if_prime(num):
    if num > 1:
    # check for factors
        for i in range(2, int(math.sqrt(num))):
            if (num % i) == 0:
                print(num, "is not a prime number")
                print(i, "times", num // i, "is", num)
                break
        else:
            print(num, "is a prime number")

    # if input number is less than
    # or equal to 1, it is not prime
    else:
        print(num, "is not a prime number")


def prime_ex2(w):
    for p in gen_primes(w):
        for m in range(2,w):
            if (p**m + m - 1) % (m*p) == 0:
                q = int((p**m + m - 1) / (m*p))
                j = 0
                for i in range(2, int(math.sqrt(q))):
                    if (q % i) == 0 and q>=1:
                        j = 1
                if j == 0:
                    print(p,q,m)
