import os
import torch
import torchaudio
import soundfile as sf
import pandas as pd
from transformers import AutoModel
from jiwer import wer, cer
from tqdm import tqdm

# Load Indic Conformer multilingual model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

# Define dataset base path and output path
base_dir = "/teamspace/studios/this_studio/Indic_dataset/Indic_dataset"
output_dir = "/teamspace/studios/this_studio/Indic_dataset_results"
os.makedirs(output_dir, exist_ok=True)

# Mapping folder -> language code
lang_map = {
    "Assamese": "as", "Bengali": "bn", "Dogri": "doi", "Gujarati": "gu",
    "Hindi": "hi", "Kannada": "kn", "Maithili": "mai", "Malayalam": "ml",
    "Manipuri": "mni", "Nepali": "ne", "Punjabi": "pa", "Rajasthani": "raj",
    "Sanskrit": "sa", "Tamil": "ta", "Telugu": "te"
}

unsupported_languages = []
summary = []
global_refs, global_preds = [], []

target_sample_rate = 16000

# Loop through each language folder
for lang_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, lang_folder)
    if not os.path.isdir(folder_path):
        continue

    lang_name = lang_folder.split("_")[0].capitalize()
    lang_code = lang_map.get(lang_name)

    if lang_code is None:
        unsupported_languages.append(lang_folder)
        continue

    wav_dir = os.path.join(folder_path, "wav")
    txt_dir = os.path.join(folder_path, "txt")
    if not os.path.isdir(wav_dir) or not os.path.isdir(txt_dir):
        continue

    print(f"\n Running ASR on: {lang_folder} | Lang Code: {lang_code}")

    rows, lang_refs, lang_preds = [], [], []

    for file in tqdm(sorted(os.listdir(wav_dir))):
        if not file.endswith(".wav"):
            continue

        audio_path = os.path.join(wav_dir, file)
        txt_path = os.path.join(txt_dir, file.replace(".wav", ".txt"))

        try:
            # Load audio (use soundfile for safety)
            wav, sr = sf.read(audio_path)
            wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
                wav = resampler(wav)

            # Run inference (CTC decoding)
            predicted = model(wav, lang_code, "ctc")
            predicted = predicted.strip()

            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    reference = f.read().strip()

                lang_refs.append(reference)
                lang_preds.append(predicted)
                global_refs.append(reference)
                global_preds.append(predicted)

                rows.append({
                    "filename": file,
                    "reference": reference,
                    "prediction": predicted
                })

        except Exception as e:
            print(f" Error on {file}: {e}")
            continue

    # Save per-language results
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, f"{lang_name}_asr_output.csv"), index=False)

    # Compute WER/CER per language
    if lang_refs:
        lang_wer = wer(lang_refs, lang_preds)
        lang_cer = cer(lang_refs, lang_preds)
        print(f" {lang_name} | WER: {lang_wer:.2f}, CER: {lang_cer:.2f}")
        summary.append({"language": lang_name, "WER": lang_wer, "CER": lang_cer})
    else:
        print(f"[!] Skipping {lang_name}, no valid transcripts")

# Save summary metrics
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

# Save unsupported languages
with open(os.path.join(output_dir, "unsupported_languages.txt"), "w") as f:
    for lang in unsupported_languages:
        f.write(f"{lang}\n")

# Compute global WER/CER
if global_refs:
    overall_wer = wer(global_refs, global_preds)
    overall_cer = cer(global_refs, global_preds)
    print(f"\n Overall WER: {overall_wer:.2f}, CER: {overall_cer:.2f}")

    with open(os.path.join(output_dir, "overall_metrics.txt"), "w") as f:
        f.write(f"Overall WER: {overall_wer:.4f}\n")
        f.write(f"Overall CER: {overall_cer:.4f}\n")

print("\n  All language processing complete. Results saved in:", output_dir)
