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
from nltk.metrics.distance import edit_distance

# Initialise CMU Pronunciation Dictionary
try:
    d_pronounce = cmudict.dict()
except LookupError:
    # Fallback if cmudict downloading failed or needs explicit loading
    nltk.download('cmudict')
    d_pronounce = cmudict.dict()

print("Optimizing WordNet vocabulary index for fast typo-correction...")
# Pre-compile a flat vocabulary list of all lowercase WordNet lemmas for instant access
WORDNET_VOCAB = set()
for synset in wn.all_synsets():
    for lemma in synset.lemmas():
        WORDNET_VOCAB.add(lemma.name().replace('_', ' ').lower())
WORDNET_VOCAB = sorted(list(WORDNET_VOCAB))


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
    """Linguistically verifies if a candidate word is a true alternative spelling."""
    t_l = target.lower().strip()
    c_l = candidate.lower().strip()

    if t_l == c_l:
        return True

    if (' ' in t_l or ' ' in c_l) and t_l.replace(' ', '') != c_l.replace(' ', ''):
        return False

    t_flat = get_flattened_baseline(t_l)
    c_flat = get_flattened_baseline(c_l)

    if t_flat == c_flat:
        return True

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

    if len(t_l) != len(c_l):
        if t_l.replace('l', 'll') == c_l or c_l.replace('l', 'll') == t_l:
            return True

    return False


def dynamic_corpus_tagger(w1, w2):
    """Programmatically applies frequency and regional taxonomy labels."""
    w1_l, w2_l = w1.lower(), w2.lower()

    if (w1_l.endswith('er') and w2_l.endswith('re')) or (w1_l.endswith('re') and w2_l.endswith('er')):
        er_form = w1_l if w1_l.endswith('er') else w2_l
        re_form = w1_l if w1_l.endswith('re') else w2_l
        return {er_form: "[=] (AmE)", re_form: "[=] (BrE)"}

    if (w1_l.endswith('or') and w2_l.endswith('our')) or (w1_l.endswith('our') and w2_l.endswith('or')):
        or_form = w1_l if w1_l.endswith('or') else w2_l
        our_form = w1_l if w1_l.endswith('our') else w2_l
        return {or_form: "[==] (AmE)", our_form: "[==] (BrE)"}

    if (w1_l.endswith('ize') and w2_l.endswith('ise')) or (w1_l.endswith('ise') and w2_l.endswith('ize')):
        ize_form = w1_l if w1_l.endswith('ize') else w2_l
        ise_form = w1_l if w1_l.endswith('ise') else w2_l
        return {ize_form: "[C] (AmE)", ise_form: "[C] (BrE)"}

    if w1_l.replace('-', '') == w2_l.replace('-', ''):
        hypor_form = w1_l if '-' in w1_l else w2_l
        flat_form = w1_l if '-' not in w1_l else w2_l
        return {flat_form: "[=]", hypor_form: "[=]"}

    return {w1_l: "[=]", w2_l: "[=]"}


def query_dynamic_dictionary(search_term):
    """Executes searches with structural normalization and interactive typo handling."""
    term_lower = search_term.lower().strip()
    wordnet_key = term_lower.replace(' ', '_')
    synsets = wn.synsets(wordnet_key)

    redirect_message = ""
    original_search = term_lower

    # 📌 1. RECURSIVE EXACT BASELINE REDIRECTION (For true hidden unhyphenated standards)
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
                    redirect_message = f"🔀 Exact Structural Redirect: '{original_search}' ➔ '{term_lower}'"
                    found_redirect = True
                    break
            if found_redirect:
                break

    # 📌 2. INTELLIGENT TYPO-CORRECTION & INTERACTIVE MATRIX LAYER
    if not synsets:
        # Pre-filter lexicon space by length step limits to optimse performance
        len_candidates = [w for w in WORDNET_VOCAB if abs(len(w) - len(term_lower)) <= 2]
        best_matches = []
        min_dist = 999

        for cand in len_candidates:
            dist = edit_distance(term_lower, cand)
            if dist < min_dist:
                min_dist = dist
                best_matches = [cand]
            elif dist == min_dist:
                best_matches.append(cand)

        # Only process high-confidence adjustments (edit distance 1 or 2)
        if min_dist <= 2 and best_matches:
            if len(best_matches) == 1:
                # Type A: Highly certain single choice typo -> Auto-redirect
                term_lower = best_matches[0]
                wordnet_key = term_lower.replace(' ', '_')
                synsets = wn.synsets(wordnet_key)
                redirect_message = f"🔧 Autocorrected Typo: '{original_search}' ➔ '{term_lower}'"
            else:
                # Type B: Uncertain ambiguous typo -> Prompt user selection menu
                print(f"\n❓ Uncertain term entry. Did you mean one of these for '{original_search}'?")
                choices_preview = best_matches[:10]  # Cap interactive view at 10 items
                for idx, cand in enumerate(choices_preview, 1):
                    print(f"   {idx}. {cand}")
                print("   0. None of the above (Skip correction)")

                try:
                    selection = input("\nSelect a menu option number: ").strip()
                    if selection.isdigit() and 1 <= int(selection) <= len(choices_preview):
                        term_lower = choices_preview[int(selection) - 1]
                        wordnet_key = term_lower.replace(' ', '_')
                        synsets = wn.synsets(wordnet_key)
                        redirect_message = f"Interactive Selection: Mapped to '{term_lower}'"
                    else:
                        print("❌ Correction bypassed.")
                except (KeyboardInterrupt, SystemExit):
                    return

    # Print clean formatted results block
    print("\n" + "=" * 80)
    print(f"🔍 SEARCH RESULTS FOR: '{original_search.upper()}'")
    print("=" * 80)
    print(f"🎙️ Phonetics (ARPAbet): {get_phonetics(term_lower)}")
    print("-" * 80)

    if redirect_message:
        print(f"{redirect_message}\n")

    # Discover and clean variants array
    all_discovered_variants = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            lemma_clean = lemma.name().replace('_', ' ')
            if is_legitimate_spelling_variant(term_lower, lemma_clean):
                all_discovered_variants.add(lemma_clean)

    all_discovered_variants.add(term_lower)
    all_discovered_variants.add(original_search)
    sorted_variants = sorted(list(all_discovered_variants), key=lambda x: (x.lower(), x))

    # Render layout labels tracking variables dynamically
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

    print(f"{original_search}\n")
    print(f"variants: {display_variants_str}")
    print("-" * 80)

    # Render word meanings
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
    print("\n--- Autocorrecting Variant Redirection Engine Initialised ---")
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
