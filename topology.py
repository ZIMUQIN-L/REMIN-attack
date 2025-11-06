import numpy as np
import tqdm
import sympy


def remove_duplicates(responses):
    new_responses = []
    d = {}
    for i in tqdm.tqdm(responses):
        if frozenset(i) not in d:
            new_responses.append(i)
            d[frozenset(i)] = 0
    return new_responses


def find_prime_responses(responses, max_primes=0):
    primes = []
    for i in responses:
        if sympy.isprime(len(i)):
            if max_primes == 0 or (len(i) <= max_primes):
                primes.append(i)

    return primes

def augment_responses(responses):
    """
    Augment responses without domain-based replacement. This method increases
    the number of size=2 responses while preserving the original data.

    :param responses: List of sets, each representing a response.
    :return: Augmented response list.
    """
    responses = remove_duplicates(responses)
    seen = set()
    frozen_responses = []

    for i in responses:
        frozen_responses.append(frozenset(i))
        seen.add(frozenset(i))

    # Generate additional responses by intersecting existing ones
    todo = []
    for i in tqdm.tqdm(frozen_responses):
        for j in frozen_responses:
            inter = i.intersection(j)
            if len(inter) > 0 and frozenset(inter) not in seen:
                todo.append(frozenset(inter))
                seen.add(frozenset(inter))

    frozen_responses.extend(todo)

    new_responses = remove_duplicates(frozen_responses)

    tuples = find_prime_responses(new_responses, 2)

    return tuples
