# active-lowres-asr-training

get_data.py
 - scores a baseline and a finetuned prediction CSV (PER/WER/CER) and compares them
 - `python get_data.py BASELINE FINETUNED [--contains TEXT] [--ref-col COL] [--pred-col COL] [--save]`
   - `BASELINE`, `FINETUNED`: CSVs with a reference column and a prediction column
   - `--contains TEXT`: only score rows whose reference contains TEXT (case-insensitive)
   - `--ref-col COL`: reference column name (default: `reference`)
   - `--pred-col COL`: prediction column name (default: first column ending in `prediction`)
   - `--save`: write `<input>_scored.csv` next to each input
 - e.g. `python get_data.py data/whisper_baseline.csv data/whisper_finetuned.csv --contains ing`

metrics.py
 - shared WER, CER and PER (via g2p_en phonemes) metrics

wav2vec2_CORAAL_finetune.ipynb
 - training the wav2vec2 model
Whisper_CORAAL_finetune.ipynb
 - training the whisper model