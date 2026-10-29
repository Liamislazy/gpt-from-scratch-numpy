import re
from collections import Counter
'''
def normalizer(texts):
    if isinstance(texts, str):
        texts = [texts]

    normalized = []
    for text in texts:
        # not using this yet
        # 1. HTML tag removal
        text = re.sub(r'<[^>]+>', '', text)

        # 2. Contraction expansion
        text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcan't\b", "can not", text, flags=re.IGNORECASE)
        text = re.sub(r"n't\b", " not", text, flags=re.IGNORECASE)
        text = re.sub(r"'re\b", " are", text, flags=re.IGNORECASE)
        text = re.sub(r"'s\b", " is", text, flags=re.IGNORECASE)
        text = re.sub(r"'ve\b", " have", text, flags=re.IGNORECASE)
        text = re.sub(r"'ll\b", " will", text, flags=re.IGNORECASE)
        text = re.sub(r"'d\b", " would", text, flags=re.IGNORECASE)

        # 3. Lowercasing
        text = text.lower()

        # 4. Remove special characters or punctuation
        text = re.sub(r'[^a-z0-9\s]', '', text)

        # 5. Whitespace normalization
        text = re.sub(r'\s+', ' ', text).strip()

        normalized.append(text)

    return normalized

'''

def pre_tokenizer(text):
    # Convert lowercase
    if isinstance(text, list):
        text = " ".join(text)

    # Tokenize
    words = text.split()
    counts = Counter(words)

    # Turn the vocab to tuple
    vocab = {}
    for word, freq in counts.items():
        word_tuple = tuple(list(word) + ['</w>'])
        vocab[word_tuple] = freq

    return vocab

def merge_tuple(word, pair):
    merged_tuple = []
    word_length = len(word)
    i = 0
    while i < word_length:
        if (
            i < word_length - 1 and
            word[i] == pair[0] and
            word[i + 1] == pair[1]
        ):
            merged_tuple.append(pair[0] + pair[1])
            i += 2
        else:
            merged_tuple.append(word[i])
            i += 1
    return tuple(merged_tuple)

def get_pairs(corpus):
    pairs = {}
    for word, freq in corpus.item():
        for i in range(len(word)-1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
    return pairs

def byte_pair_encoding(corpus, num_merges):
    merges = {}

    for _ in range(num_merges):
        pairs = get_pairs(corpus)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        merges[best_pair] = pairs[best_pair]

        for item in corpus:
            item[0] = merge_tuple(item[0], best_pair)

    return merges

