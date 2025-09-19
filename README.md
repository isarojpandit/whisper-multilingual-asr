
# Whisper-Based Multilingual ASR for Indic Languages

This project evaluates **Whisper models** (medium, turbo, large) on **15 Indic languages** for Automatic Speech Recognition (ASR).  
Evaluation is based on **3,000 speech samples** (200 per language, balanced male/female).  

## 📌 Features
- Benchmarked **WER, CER, PER**
- Languages: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Nepali, Punjabi, Sanskrit, Tamil, Telugu, Dogri, Maithili, Manipuri, Rajasthani
- Compared with other ASR frameworks, AI4Bharat (IndianConformer)
- Paper Accepted to **PReMI 2025, IIT Delhi**

## ⚙️ Setup
```bash
git clone https://github.com/isarojpandit/whisper-multilingual-asr.git
cd whisper-multilingual-asr
pip install -r requirements.txt
