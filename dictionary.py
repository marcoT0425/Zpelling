import ssl
import sys
import unicodedata

# Automatically bypass macOS SSL restrictions for downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk

print("Loading dictionary database into memory, please wait...")
nltk.download('wordnet', quiet=True)
nltk.download('cmudict', quiet=True)

from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict

# Initialize CMU Pronunciation Dictionary
d_pronounce = cmudict.dict()


def strip_diacritics(text):
    """Recursively drops accent configurations natively (e.g., coöperate -> cooperate)."""
    normalized = unicodedata.normalize('NFD', text.lower())
    return "".join(c for c in normalized if unicodedata.category(c) != 'Mn')


def get_flattened_baseline(text):
    """Strips all structural accents, hyphens, spaces, and dots to check structural identity."""
    return strip_diacritics(text).replace('-', '').replace(' ', '').replace('.', '')


def get_phonetics(word):
    """Fetches arpabet phonetic data from CMUDict."""
    word_clean = word.lower().strip()
    if word_clean in d_pronounce:
        return " / ".join(" ".join(syllable) for syllable in d_pronounce[word_clean])
    return "Phonetics database entry missing"


def is_legitimate_spelling_variant(target, candidate):
    """
    Linguistically verifies if a candidate word is a true alternative spelling
    of the target word, blocking unrelated synonyms, derivatives, and phrases.
    """
    t_l = target.lower().strip()
    c_l = candidate.lower().strip()

    if t_l == c_l:
        return True

    # Guard: If one has spaces/hyphens and the other doesn't, they must be compound variants
    if (' ' in t_l or ' ' in c_l) and t_l.replace(' ', '') != c_l.replace(' ', ''):
        return False

    t_flat = get_flattened_baseline(t_l)
    c_flat = get_flattened_baseline(c_l)

    # 1. Accent, hyphen, or spacing variations of the exact same word
    if t_flat == c_flat:
        return True

    # 2. Regional suffix variations (-or/-our, -ize/-ise, -er/-re, -og/-ogue)
    suffix_pairs = [
        ('or', 'our'), ('our', 'or'),
        ('ize', 'ise'), ('ise', 'ize'),
        ('er', 're'), ('re', 'er'),
        ('og', 'ogue'), ('ogue', 'og')
    ]
    for s1, s2 in suffix_pairs:
        if t_l.endswith(s1) and c_l.endswith(s2):
            if t_l[:-len(s1)] == c_l[:-len(s2)]:
                return True

    # 3. Consonant doubling variations (traveler/traveller)
    if len(t_l) != len(c_l):
        if t_l.replace('l', 'll') == c_l or c_l.replace('l', 'll') == t_l:
            return True

    return False


def dynamic_corpus_tagger(w1, w2):
    """Programmatically applies frequency and regional taxonomy labels."""
    w1_l, w2_l = w1.lower(), w2.lower()

    # -er vs -re regional swaps (e.g., center vs centre)
    if (w1_l.endswith('er') and w2_l.endswith('re')) or (w1_l.endswith('re') and w2_l.endswith('er')):
        er_form = w1_l if w1_l.endswith('er') else w2_l
        re_form = w1_l if w1_l.endswith('re') else w2_l
        return {er_form: "[=] (AmE)", re_form: "[=] (BrE)"}

    # -or vs -our regional swaps (e.g., color vs colour)
    if (w1_l.endswith('or') and w2_l.endswith('our')) or (w1_l.endswith('our') and w2_l.endswith('or')):
        or_form = w1_l if w1_l.endswith('or') else w2_l
        our_form = w1_l if w1_l.endswith('our') else w2_l
        return {or_form: "[==] (AmE)", our_form: "[==] (BrE)"}

    # -ize vs -ise regional swaps (e.g., organize vs organise)
    if (w1_l.endswith('ize') and w2_l.endswith('ise')) or (w1_l.endswith('ise') and w2_l.endswith('ize')):
        ize_form = w1_l if w1_l.endswith('ize') else w2_l
        ise_form = w1_l if w1_l.endswith('ise') else w2_l
        return {ize_form: "[C] (AmE)", ise_form: "[C] (BrE)"}

    # Hyphen variations
    if w1_l.replace('-', '') == w2_l.replace('-', ''):
        hyphen_form = w1_l if '-' in w1_l else w2_l
        flat_form = w1_l if '-' not in w1_l else w2_l
        return {flat_form: "[=]", hyphen_form: "[=]"}

    return {w1_l: "[=]", w2_l: "[=]"}


def query_dynamic_dictionary(search_term):
    """Executes search queries with structural filtering to block false synonyms."""
    term_lower = search_term.lower().strip()
    wordnet_key = term_lower.replace(' ', '_')
    synsets = wn.synsets(wordnet_key)

    redirected = False
    original_search = term_lower

    # 📌 RECURSIVE REDIRECTION LAYER
    if not synsets:
        search_flat = get_flattened_baseline(term_lower)
        found_redirect = False

        for syn in wn.all_synsets():
            for lemma in syn.lemmas():
                lemma_clean = lemma.name().replace('_', ' ')
                if get_flattened_baseline(lemma_clean) == search_flat:
                    wordnet_key = lemma.name()
                    term_lower = lemma_clean
                    synsets = wn.synsets(wordnet_key)
                    redirected = True
                    found_redirect = True
                    break
            if found_redirect:
                break

    print("\n" + "=" * 80)
    print(f"🔍 SEARCH RESULTS FOR: '{original_search.upper()}'")
    print("=" * 80)
    print(f"🎙️ Phonetics (ARPAbet): {get_phonetics(term_lower)}")
    print("-" * 80)

    if redirected:
        print(f"🔀 WordNet Redirection: Variant spelling structure not found natively.")
        print(f"   Redirected input form: '{original_search}' ----> root form: '{term_lower}'\n")

    # Discover and FILTER synonyms out from the current Synset concept pool
    all_discovered_variants = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            lemma_clean = lemma.name().replace('_', ' ')
            # STRICT GUARD: Only accept true spelling variants, drop generic synonyms like 'tinge'
            if is_legitimate_spelling_variant(term_lower, lemma_clean):
                all_discovered_variants.add(lemma_clean)

    all_discovered_variants.add(term_lower)
    all_discovered_variants.add(original_search)
    sorted_variants = sorted(list(all_discovered_variants), key=lambda x: (x.lower(), x))

    # Generate dynamic custom tags
    display_tags = {}
    if len(sorted_variants) >= 2:
        for i in range(len(sorted_variants)):
            for j in range(i + 1, len(sorted_variants)):
                resolved_tags = dynamic_corpus_tagger(sorted_variants[i], sorted_variants[j])
                display_tags.update(resolved_tags)

        formatted_list = []
        for word in sorted_variants:
            tag = display_tags.get(word.lower(), "[=]")
            formatted_list.append(f"{word} {tag}")
        display_variants_str = " / ".join(formatted_list)
    else:
        display_variants_str = f"{term_lower} [=]"

    # Print clean block mapping
    print(f"{original_search}\n")
    print(f"variants: {display_variants_str}")
    print("-" * 80)

    # Output definitions
    if synsets:
        print(f"💡 MEANINGS & SENTENCE EXAMPLES (WordNet Reference Target: '{term_lower.upper()}'):")
        pos_names = {'n': 'Noun', 'v': 'Verb', 'a': 'Adjective', 'r': 'Adverb', 's': 'Adjective Satellite'}
        for idx, syn in enumerate(synsets, 1):
            print(f"   {idx}. [{pos_names.get(syn.pos(), syn.pos().upper())}] {syn.definition()}")
            for ex in syn.examples():
                print(f"      • Ex: \"{ex}\"")
    else:
        print("💡 MEANINGS: Term definitions missing from WordNet database matrices.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n--- Cleaned Variant Redirection Engine Initialized ---")
    print("Type 'exit' into the prompt to terminate the dictionary application.\n")

    while True:
        try:
            word_query = input("Enter an English word to analyse its meanings: ")
            if word_query.lower().strip() == 'exit':
                print("Closing system framework.")
                break
            if word_query.strip():
                query_dynamic_dictionary(word_query)
            else:
                print("⚠️ Warning: Input stream cannot be blank.")
        except (KeyboardInterrupt, SystemExit):
            print("\nApplication closed.")
            break
