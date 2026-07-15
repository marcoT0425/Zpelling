import ssl
import sys

# Automatically bypass macOS SSL restrictions for NLTK components
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
import inflect
import unicodedata
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
    Evaluates word pairs cleanly using true structural morphology.
    Blocks derivative errors like 'bacterium' while tracking spelling and plural variants [P].
    """
    tags = set()
    w1_lower = w1.lower()
    w2_lower = w2.lower()

    # 1. Capitalisation [CC]
    if w1 != w2 and w1_lower == w2_lower:
        tags.add('CC')
        return tags

    # FILTER GUARD: Block classic morphological functional suffixes (e.g., wide vs widely)
    false_suffixes = ('ly', 'ness', 'ment', 'tion', 'able', 'al', 'ity', 'ish', 'ous', 'ed', 'ing', 'er', 'est')
    for fs in false_suffixes:
        if (w1_lower.endswith(fs) and not w2_lower.endswith(fs) and w1_lower.startswith(w2_lower)) or \
                (w2_lower.endswith(fs) and not w1_lower.endswith(fs) and w2_lower.startswith(w1_lower)):
            return None

    # FILTER GUARD: Block prefix transitions (e.g., eminence vs pre-eminence, valuation vs revaluation)
    false_prefixes = ('pre-', 'anti-', 'un-', 'dis-', 'non-', 'counter-', 'sub-', 'co-', 're-')
    for fp in false_prefixes:
        if (w1_lower.startswith(fp) and not w2_lower.startswith(fp)) or \
                (w2_lower.startswith(fp) and not w1_lower.startswith(fp)):
            if w1_lower.replace('-', '') == w2_lower.replace('-', ''):
                continue
            return None

    # 2. Pluralisation [P] Algorithm Implementation
    # Check if one word is a proven plural variant of the exact same root unit
    try:
        is_p1_plural_of_w2 = (p_engine.plural(w2_lower) == w1_lower or p_engine.plural_noun(w2_lower) == w1_lower)
        is_p2_plural_of_w1 = (p_engine.plural(w1_lower) == w2_lower or p_engine.plural_noun(w1_lower) == w2_lower)

        # Capture classical foreign root doublets (e.g., formulas vs formulae, indexes vs indices)
        classical_suffixes = ('ae', 'as', 'i', 'es', 'ises', 'izes', 'a', 'ums')
        has_classical_mutation = any(w1_lower.endswith(s) for s in classical_suffixes) and any(
            w2_lower.endswith(s) for s in classical_suffixes)

        if is_p1_plural_of_w2 or is_p2_plural_of_w1 or (
                has_classical_mutation and edit_distance(w1_lower, w2_lower) <= 3):
            # STRICT GUARD: Explicitly block singular words matching their raw root incorrectly (e.g., bacterium/bacteria mismatching)
            if w1_lower.endswith('ium') and w2_lower.endswith('ium') and w1_lower != w2_lower:
                return None
            tags.add('P')
    except Exception:
        pass

    # Flatten diacritics
    w1_no_acc = strip_diacritics(w1_lower)
    w2_no_acc = strip_diacritics(w2_lower)

    # 3. Accents [A]
    if w1_lower != w2_lower and w1_no_acc == w2_no_acc:
        tags.add('A')
        return tags

    # Remove hyphens and spaces to check structural doublets
    w1_flat = w1_no_acc.replace('-', '').replace(' ', '')
    w2_flat = w2_no_acc.replace('-', '').replace(' ', '')

    # 4. Hyphens [H] & Compounding Spacing [C]
    if w1_flat == w2_flat:
        if '-' in w1 or '-' in w2:
            tags.add('H')
        if ' ' in w1 or ' ' in w2:
            tags.add('C')
        return tags

    # Calculate precise Levenshtein editing distance
    dist = edit_distance(w1_no_acc, w2_no_acc)
    if dist > 2 or dist == 0:
        # If it's a confirmed plural mutation, allow a larger variance cap
        if 'P' not in tags:
            return None

    # 5. Suffix Transformations [S] & Regional Tags [R] (-or/-our, -ize/-ise)
    common_prefix_len = 0
    for c1, c2 in zip(w1_no_acc, w2_no_acc):
        if c1 == c2:
            common_prefix_len += 1
        else:
            break

    if common_prefix_len > max(len(w1_no_acc), len(w2_no_acc)) - 4:
        tags.add('S')
        tags.add('R')

    # 6. Regional / Miscellaneous [R] (e.g., yogurt/yoghurt)
    if not tags and dist <= 2:
        tags.add('R')

    return tags if tags else None


def scan_wordnet_carefully():
    print("Executing strict WordNet dictionary scan for pluralisations and spelling variants...")
    final_output_groups = set()

    for synset in wn.all_synsets():
        # Track word strings preserving case mappings
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

        # Cluster group unification pass
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

    # Output verified spelling lists systematically sorted from A to Z
    sorted_master_output = sorted(list(final_output_groups), key=lambda x: x.lower())

    print(f"\n--- Verified Spelling & Plural Groups ({len(sorted_master_output)} Found) ---\n")
    for group_line in sorted_master_output:
        print(group_line)


if __name__ == "__main__":
    scan_wordnet_carefully()
