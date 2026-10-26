import re

import g2p_en


def error_rate(ref, hyp):
    """Levenshtein distance between token sequences, normalized by len(ref)."""
    R, H = len(ref), len(hyp)
    # d[i][j] = edit distance between ref[:i] and hyp[:j]
    d = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        d[i][0] = i          # delete all i ref tokens
    for j in range(H + 1):
        d[0][j] = j          # insert all j hyp tokens

    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,        # deletion
                d[i][j-1] + 1,        # insertion
                d[i-1][j-1] + cost,   # substitution / match
            )

    return d[R][H] / R if R > 0 else float(H > 0)


def wer(reference, hypothesis):
    return error_rate(str(reference).split(), str(hypothesis).split())


def cer(reference, hypothesis):
    return error_rate(list(str(reference)), list(str(hypothesis)))


class G2PTranslator:
    def __init__(self):
        self.g2p = g2p_en.G2p()

    def text_to_cmu(self, text) -> list[str]:
        text = re.sub(r"[^A-Za-z0-9 ]", "", str(text))
        return self.g2p(text)

    def per(self, reference, hypothesis):
        ref = [x for x in self.text_to_cmu(reference) if x != ' ']
        hyp = [x for x in self.text_to_cmu(hypothesis) if x != ' ']
        return error_rate(ref, hyp)
