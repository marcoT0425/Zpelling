import ssl
import sys
import unicodedata

# Automatically bypass macOS SSL restrictions for NLTK components
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
import inflect
from nltk.corpus import wordnet as wn
from nltk.metrics.distance import edit_distance

# Download NLTK database assets quietly
nltk.download('wordnet', quiet=True)

# Initialize the programmatic inflection framework
p_engine = inflect.engine()


def strip_diacritics(text):
    """Recursively drops structural accent marks (e.g., coöperate -> cooperate)."""
    normalized = unicodedata.normalize('NFD', text)
    return "".join(c for c in normalized if unicodedata.category(c) != 'Mn')


def determine_relationship_tags(w1, w2):
    """
    Evaluates word pairs cleanly using advanced spelling pattern mutations.
    Tracks foreign plurals, hyphens, and fine-grained orthographic variations.
    """
    tags = set()
    w1_lower = w1.lower()
    w2_lower = w2.lower()

    # 1. Capitalisation [CC]
    if w1 != w2 and w1_lower == w2_lower:
        tags.add('CC')
        return tags

    # Clean baselines
    w1_no_acc = strip_diacritics(w1_lower)
    w2_no_acc = strip_diacritics(w2_lower)

    # 2. Accents [A]
    if w1_lower != w2_lower and w1_no_acc == w2_no_acc:
        tags.add('A')
        return tags

    # Flatten hyphens/spaces
    w1_flat = w1_no_acc.replace('-', '').replace(' ', '')
    w2_flat = w2_no_acc.replace('-', '').replace(' ', '')

    # 3. Hyphens [H] & Compounding Spacing [C]
    if w1_flat == w2_flat:
        if '-' in w1 or '-' in w2: tags.add('H')
        if ' ' in w1 or ' ' in w2: tags.add('C')
        return tags

    # -------------------------------------------------------------------------
    # NON-ALGORITHMIC DICTIONARY PLURAL CHECK [P] (Foreign & Irregular)
    # -------------------------------------------------------------------------
    for pos_type in ['n', 'v']:
        try:
            exc_map = wn._exception_map[pos_type]
            if w1_lower in exc_map and w2_lower in exc_map[w1_lower]:
                tags.add('P')
            if w2_lower in exc_map and w1_lower in exc_map[w2_lower]:
                tags.add('P')
        except Exception:
            pass

    # Fallback to classical foreign structural suffix transitions
    classical_foreign_endings = [
        ('a', 'ae'), ('ae', 'a'), ('us', 'i'), ('i', 'us'),
        ('um', 'a'), ('a', 'um'), ('is', 'es'), ('es', 'is'),
        ('on', 'a'), ('a', 'on'), ('ex', 'ices'), ('ices', 'ex'),
        ('ix', 'matrices'), ('ix', 'matrixes'), ('o', 'i'), ('i', 'o')
    ]
    for s1, s2 in classical_foreign_endings:
        if w1_lower.endswith(s1) and w2_lower.endswith(s2):
            if w1_lower[:-len(s1)] == w2_lower[:-len(s2)]:
                tags.add('P')

    # Standard engine inflection tracking
    try:
        if not tags:
            is_p1_plural_of_w2 = (p_engine.plural(w2_lower) == w1_lower or p_engine.plural_noun(w2_lower) == w1_lower)
            is_p2_plural_of_w1 = (p_engine.plural(w1_lower) == w2_lower or p_engine.plural_noun(w1_lower) == w2_lower)
            if is_p1_plural_of_w2 or is_p2_plural_of_w1:
                if w1_lower.endswith('ium') and w2_lower.endswith('ium') and w1_lower != w2_lower:
                    return None
                tags.add('P')
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # ADVANCED SPELLING PATTERN DETECTORS
    # -------------------------------------------------------------------------

    # Pattern A: Suffix -able / -ible shifts
    if (w1_no_acc.endswith('able') and w2_no_acc.endswith('ible')) or \
            (w1_no_acc.endswith('ible') and w2_no_acc.endswith('able')):
        base1 = w1_no_acc[:-4]
        base2 = w2_no_acc[:-4]
        if base1 == base2 or base1 == base2 + 'e' or base2 == base1 + 'e':
            tags.update(['S', 'R'])

    # Pattern B: Silent 'e' drop vs. retain before functional suffixes
    suffixes_to_check = ('ment', 'ing', 'able', 'age', 'ance', 'acy')
    for suf in suffixes_to_check:
        if w1_no_acc.endswith(suf) and w2_no_acc.endswith(suf):
            rem1 = w1_no_acc[:-len(suf)]
            rem2 = w2_no_acc[:-len(suf)]
            if (rem1 + 'e' == rem2) or (rem2 + 'e' == rem1):
                tags.update(['S', 'R'])

    # Pattern C: Greek Digraph mutations
    if w1_no_acc.replace('ae', 'e') == w2_no_acc or w2_no_acc.replace('ae', 'e') == w1_no_acc or \
            w1_no_acc.replace('oe', 'e') == w2_no_acc or w2_no_acc.replace('oe', 'e') == w1_no_acc:
        tags.update(['S', 'R'])

    # Pattern D: C vs. S substitutions
    if len(w1_no_acc) == len(w2_no_acc):
        diffs = [i for i, (c1, c2) in enumerate(zip(w1_no_acc, w2_no_acc)) if c1 != c2]
        if len(diffs) == 1:
            idx = diffs[0]
            if (w1_no_acc[idx] == 'c' and w2_no_acc[idx] == 's') or (w1_no_acc[idx] == 's' and w2_no_acc[idx] == 'c'):
                tags.add('R')

    # Pattern E: F vs. PH substitutions
    if w1_no_acc.replace('ph', 'f') == w2_no_acc or w2_no_acc.replace('ph', 'f') == w1_no_acc:
        tags.add('R')

    # Pattern F: Systematic Suffix transforms
    suffix_pairs = [('or', 'our'), ('ize', 'ise'), ('er', 're'), ('og', 'ogue')]
    for s1, s2 in suffix_pairs:
        if (w1_no_acc.endswith(s1) and w2_no_acc.endswith(s2)) or (w1_no_acc.endswith(s2) and w2_no_acc.endswith(s1)):
            if w1_no_acc[:-len(s1)] == w2_no_acc[:-len(s2)]:
                tags.update(['S', 'R'])

    # Pattern G: Consonant doubling variations
    if len(w1_no_acc) != len(w2_no_acc):
        if w1_no_acc.replace('ll', 'l') == w2_no_acc or w2_no_acc.replace('ll', 'l') == w1_no_acc or \
                w1_no_acc.replace('ss', 's') == w2_no_acc or w2_no_acc.replace('ss', 's') == w1_no_acc or \
                w1_no_acc.replace('tt', 't') == w2_no_acc or w2_no_acc.replace('tt', 't') == w1_no_acc:
            tags.update(['S', 'R'])

    # -------------------------------------------------------------------------
    # FILTER AND DISTANCE GUARDRAILS
    # -------------------------------------------------------------------------
    if not tags:
        # FILTER GUARD: Block distinct functional cross-derivatives
        false_suffixes = ('ly', 'ness', 'tion', 'al', 'ity', 'ish', 'ous', 'ed', 'ing', 'er', 'est')
        for fs in false_suffixes:
            if (w1_lower.endswith(fs) and not w2_lower.endswith(fs) and w1_lower.startswith(w2_lower)) or \
                    (w2_lower.endswith(fs) and not w1_lower.endswith(fs) and w2_lower.startswith(w1_lower)):
                return None

        # FILTER GUARD: Block non-spelling prefix shifts
        false_prefixes = ('pre-', 'anti-', 'un-', 'dis-', 'non-', 'counter-', 'sub-', 'co-', 're-')
        for fp in false_prefixes:
            if (w1_lower.startswith(fp) and not w2_lower.startswith(fp)) or \
                    (w2_lower.startswith(fp) and not w1_lower.startswith(fp)):
                if w1_lower.replace('-', '') == w2_lower.replace('-', ''):
                    continue
                return None

    # Precise edit distance metric guard
    dist = edit_distance(w1_no_acc, w2_no_acc)
    if dist > 2 or dist == 0:
        if 'P' not in tags and 'S' not in tags and 'R' not in tags:
            return None

    if not tags and dist <= 2:
        tags.add('R')

    return tags if tags else None


def scan_wordnet_carefully():
    print("Executing precise WordNet dictionary pattern scanning configuration...")
    final_output_groups = set()

    for synset in wn.all_synsets():
        lemmas = [lemma.name().replace('_', ' ') for lemma in synset.lemmas()]
        unique_lemmas = sorted(list(set(lemmas)))

        if len(unique_lemmas) < 2:
            continue

        connections = {word: set() for word in unique_lemmas}
        registry = {}

        for i in range(len(unique_lemmas)):
            for j in range(i + 1, len(unique_lemmas)):
                w1 = unique_lemmas[i]
                w2 = unique_lemmas[j]

                if abs(len(w1) - len(w2)) > 4:
                    continue

                computed_tags = determine_relationship_tags(w1, w2)
                if computed_tags:
                    connections[w1].add(w2)
                    connections[w2].add(w1)
                    registry[tuple(sorted([w1, w2]))] = computed_tags

        # Component resolution
        visited = set()
        for word in unique_lemmas:
            if word in visited:
                continue

            cluster = [word]
            queue = list(connections[word])

            while queue:
                current = queue.pop(0)
                if current not in cluster:
                    if all(current in connections[member] for member in cluster):
                        cluster.append(current)
                        queue.extend(connections[current])

            for w in cluster:
                visited.add(w)

            if len(cluster) >= 2:
                sorted_cluster = sorted(cluster, key=lambda x: (x.lower(), x))
                all_cluster_tags = set()

                for i in range(len(sorted_cluster)):
                    for j in range(i + 1, len(sorted_cluster)):
                        pkey = tuple(sorted([sorted_cluster[i], sorted_cluster[j]]))
                        if pkey in registry:
                            all_cluster_tags.update(registry[pkey])

                if all_cluster_tags:
                    tag_str = f"[{','.join(sorted(list(all_cluster_tags)))}]"
                    variants_str = ", ".join(sorted_cluster)
                    final_output_groups.add(f"{variants_str} {tag_str}")

    sorted_master_output = sorted(list(final_output_groups), key=lambda x: x.lower())
    print(f"\n--- Verified Spelling, Suffix, & Plural Groups ({len(sorted_master_output)} Found) ---\n")
    for group_line in sorted_master_output:
        print(group_line)


if __name__ == "__main__":
    scan_wordnet_carefully()
