import os
import whisper
from jiwer import wer, cer
from tqdm import tqdm
import pandas as pd

# Base directory where language folders are stored
base_dir = "/media/linux/Seagate/Other_users/Saroj/Indic_dataset"
model = whisper.load_model("medium")
lang_map = {
    "Assamese": "as", "Bengali": "bn", "Dogri": "doi", "Gujarati": "gu",
    "Hindi": "hi", "Kannada": "kn", "Maithili": "mai", "Malayalam": "ml",
    "Manipuri": "mni", "Nepali": "ne", "Punjabi": "pa", "Rajasthani": "raj",
    "Sanskrit": "sa", "Tamil": "ta", "Telugu": "te"
}

unsupported_languages = []
summary = []
global_refs = []
global_preds = []

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

    print(f" Running ASR on: {lang_folder} | Lang Code: {lang_code}")

    rows, lang_refs, lang_preds = [], [], []

    for file in tqdm(sorted(os.listdir(wav_dir))[:100]):
        if not file.endswith(".wav"):
            continue

        audio_path = os.path.join(wav_dir, file)
        txt_path = os.path.join(txt_dir, file.replace(".wav", ".txt"))

        try:
            result = model.transcribe(audio_path, language=lang_code)
            predicted = result["text"].strip()

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

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(f"{lang_name}_asr_output.csv", index=False)

    if lang_refs:
        lang_wer = wer(lang_refs, lang_preds)
        lang_cer = cer(lang_refs, lang_preds)
        print(f" {lang_name} | WER: {lang_wer:.2f}, CER: {lang_cer:.2f}")
        summary.append({"language": lang_name, "WER": lang_wer, "CER": lang_cer})
    else:
        print(f"[!] Skipping {lang_name}, no valid transcripts")

# Save summary metrics
summary_df = pd.DataFrame(summary)
summary_df.to_csv("summary_metrics.csv", index=False)

# Save unsupported
with open("unsupported_languages.txt", "w") as f:
    for lang in unsupported_languages:
        f.write(f"{lang}\n")

# Overall metrics
if global_refs:
    overall_wer = wer(global_refs, global_preds)
    overall_cer = cer(global_refs, global_preds)
    print(f"\n Overall WER: {overall_wer:.2f}, CER: {overall_cer:.2f}")

    with open("overall_metrics.txt", "w") as f:
        f.write(f"Overall WER: {overall_wer:.4f}\n")
        f.write(f"Overall CER: {overall_cer:.4f}\n")

print("All language processing complete.")

