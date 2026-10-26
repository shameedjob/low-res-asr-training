import argparse
from pathlib import Path

import pandas

import metrics


def score(path, translator, ref_col, pred_col=None, contains=None, save=False):
    df = pandas.read_csv(path)
    if pred_col is None:
        # wav2vec2 exports use e.g. "zero_shot_prediction" / "finetuned_prediction"
        pred_col = next(c for c in df.columns if c.endswith('prediction'))
    if contains:
        df = df[df[ref_col].str.contains(contains, case=False, regex=False, na=False)]
    print(path, "rows:", len(df))

    df["per"] = df.apply(lambda x: translator.per(x[ref_col], x[pred_col]), axis=1)
    df["wer"] = df.apply(lambda x: metrics.wer(x[ref_col], x[pred_col]), axis=1)
    df["cer"] = df.apply(lambda x: metrics.cer(x[ref_col], x[pred_col]), axis=1)
    if save:
        out = Path(path).with_name(Path(path).stem + "_scored.csv")
        df.to_csv(out, index=False)
        print("wrote", out)
    return df["per"].mean(), df["wer"].mean(), df["cer"].mean()


def main():
    parser = argparse.ArgumentParser(description="Score and compare baseline vs. finetuned transcriptions (PER/WER/CER).")
    parser.add_argument("baseline", help="CSV of baseline predictions")
    parser.add_argument("finetuned", help="CSV of finetuned predictions")
    parser.add_argument("--contains", help="only score rows whose reference contains this substring (case-insensitive)")
    parser.add_argument("--ref-col", default="reference", help="reference column name (default: reference)")
    parser.add_argument("--pred-col", help="prediction column name (default: first column ending in 'prediction')")
    parser.add_argument("--save", action="store_true", help="write <input>_scored.csv next to each input")
    args = parser.parse_args()

    translator = metrics.G2PTranslator()
    kwargs = dict(ref_col=args.ref_col, pred_col=args.pred_col, contains=args.contains, save=args.save)
    original_per, original_wer, original_cer = score(args.baseline, translator, **kwargs)
    new_per, new_wer, new_cer = score(args.finetuned, translator, **kwargs)

    print(f"{'':10}{'baseline':>10}{'finetuned':>10}{'ratio':>10}")
    for name, old, new in [("PER", original_per, new_per), ("WER", original_wer, new_wer), ("CER", original_cer, new_cer)]:
        print(f"{name:10}{old:10.4f}{new:10.4f}{old / new:10.4f}")
    print(f"{'PER/WER':10}{original_per / original_wer:10.4f}{new_per / new_wer:10.4f}")


if __name__ == "__main__":
    main()
