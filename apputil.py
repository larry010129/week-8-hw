from collections import defaultdict

import numpy as np


class MarkovText(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.term_dict = None  # you'll need to build this

    def get_term_dict(self):
        # Split on whitespace and leave punctuation attached, so "kind,"
        # stays one token, matching the sample sentence in the prompt.
        tokens = self.corpus.split()

        # A missing key starts as [], so the first follower can be appended.
        term_dict = defaultdict(list)

        # Map each word to every word that comes immediately after it.
        # Repeats stay in the list. np.random.choice then picks a common
        # follower more often, which is the transition probability.
        for current, nxt in zip(tokens, tokens[1:]):
            term_dict[current].append(nxt)

        # The last token has nothing after it. If it appears only there,
        # leave it in the dict with an empty list so generate() can see
        # the dead end instead of crashing on a missing key.
        # setdefault does not wipe a list built from an earlier occurrence.
        if tokens:
            term_dict.setdefault(tokens[-1], [])

        self.term_dict = dict(term_dict)
        return self.term_dict

    def generate(self, seed_term=None, term_count=15):
        # The notebook builds the dict first. This covers a direct call.
        if self.term_dict is None:
            self.get_term_dict()

        # No seed: start on a random corpus token.
        # A seed that never appears in the corpus raises ValueError.
        if seed_term is None:
            current = np.random.choice(self.corpus.split())
        elif seed_term not in self.term_dict:
            raise ValueError(f"{seed_term!r} is not in the corpus")
        else:
            current = seed_term

        generated = [current]

        # Words we can jump to when the chain hits a token with no follower.
        continuable = [
            term
            for term, followers in self.term_dict.items()
            if followers
        ]

        # The seed counts as the first term, so draw term_count - 1 more.
        for _ in range(term_count - 1):
            followers = self.term_dict[current]
            if followers:
                current = np.random.choice(followers)
            else:
                # Dead end, usually the final token seen only once.
                # Continue from a word that has a follower so the result
                # still has term_count words. This jump is not a transition.
                current = np.random.choice(continuable)
            generated.append(current)

        return " ".join(generated)
