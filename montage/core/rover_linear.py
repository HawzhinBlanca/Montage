from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

Word = Tuple[float, float, str, float]   # (start, end, text, conf)
_JITTER = 0.050                          # seconds

def rover_merge(transcripts: List[Iterable[Word]],
                jitter: float = _JITTER) -> str:
    """Linear-time ROVER merge algorithm as specified in Tasks.md"""
    pool: List[Word] = [w for t in transcripts for w in t]
    if not pool:                                   # ← edge-case guard
        return ""

    pool.sort(key=lambda w: w[0])                  # O(N log N), N ≤ 12 k

    merged: List[str] = []
    i, n = 0, len(pool)
    while i < n:
        anchor = pool[i][0]
        j = i + 1
        while j < n and pool[j][0] - anchor <= jitter:
            j += 1
        best = max(pool[i:j], key=lambda w: (w[3], -len(w[2])))
        merged.append(best[2])
        i = j
    return " ".join(merged)


def rover_merge_adapter(fw_words: List[Dict[str, Any]], dg_words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adapter function that maintains backward compatibility with existing codebase.
    Converts dictionary format to tuple format, runs linear ROVER, then converts back.
    """
    if not fw_words and not dg_words:
        return []
    if not fw_words:
        return dg_words
    if not dg_words:
        return fw_words

    # Convert dict format to tuple format for linear algorithm
    fw_tuples = [(w["start"], w["end"], w["word"], w.get("confidence", 0.8)) for w in fw_words]
    dg_tuples = [(w["start"], w["end"], w["word"], w.get("confidence", 0.8)) for w in dg_words]

    # Run linear-time ROVER merge
    merged_text = rover_merge_original([fw_tuples, dg_tuples])

    # Convert merged text back to word list with timestamps
    # We need to reconstruct the word list from the merged text
    merged_words_text = merged_text.split()

    # Create a mapping of words to their best metadata
    word_to_data = {}
    for word_list in [fw_words, dg_words]:
        for w in word_list:
            word_text = w["word"]
            if word_text not in word_to_data or w.get("confidence", 0) > word_to_data[word_text].get("confidence", 0):
                word_to_data[word_text] = w

    # Build result maintaining original structure
    result = []
    used_words = set()

    # First pass: add words in order they appear in merged text
    for word in merged_words_text:
        if word in word_to_data and word not in used_words:
            result.append(word_to_data[word].copy())
            used_words.add(word)

    # Sort by start time to ensure proper order
    result.sort(key=lambda w: w["start"])

    return result


# Export the adapter as rover_merge for backward compatibility
rover_merge_original = rover_merge
rover_merge = rover_merge_adapter
