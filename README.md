# PEGASUS News Summarization

A full pipeline for abstractive news summarization using the PEGASUS transformer, including data preprocessing, tokenization, model training, evaluation, and a Streamlit-powered web interface.

---

## Project Overview

This project fine-tunes the `google/pegasus-xsum` model on English news articles.  
It covers robust cleaning, tokenization, efficient sequence-to-sequence training, performance evaluation with ROUGE metrics, and a ready-to-use web demo app for generating summaries.

---

## Project Structure

```News Summarization/
├── App/
│ └── streamlit_pegasus_app.py  # Streamlit app for interactive summarization
├── Data/
│ ├── Raw/ # Raw, original datasets
│ └── Processed/
│ ├── english_train_cleaned.csv
│ ├── english_test_cleaned.csv
│ └── tokenized_pegasus_english/
│ ├── train/ # Tokenized training data
│ └── test/ # Tokenized test data
├── Models/
│ └── pegasus_english_finetuned/
│ ├── checkpoint-*/ # Model checkpoints
│ └── final_model/ # Final fine-tuned model and tokenizer
├── Notebooks/
│ ├── Preprocessing/
│ │ └── preprocessing_english.ipynb # Data cleaning and exploratory analysis
│ ├── Tokenization/
│ │ └── pegasus_tokenize.ipynb # Prepare datasets for training
│ └── Training/
│ └── train_pegasus_final.ipynb # Fine-tune model and monitor progress
│ ├── Evaluation/
│ │ └── evaluation.ipynb # Compute ROUGE and review sample summaries
├── requirements.txt # List of project dependencies
└── README.md # Project info and instructions


## Dataset & Trained Model

The data and model are too large to include directly in this repository.

- **Dataset (Raw + Processed + Tokenized)**  
  [Download from Google Drive](https://drive.google.com/file/d/1T9mx5l-4CLOw7nRQAujT3CSGl3F-rXII/view?usp=drive_link)  
  After downloading, unzip and place the folder in `Data/`.

- **Trained Model (Final Fine-Tuned PEGASUS)**  
  [Download from Google Drive](https://drive.google.com/file/d/167KtBRJPTy_LgxAhKMDiC8HXcs9QLkdP/view?usp=drive_link)  
  After downloading, unzip and place the folder in `Models/`.

---

## Contributors
-Avirat Murade (https://github.com/AviratMurade2802)
- Abhishek Shete (https://github.com/Abhishekshete0808)  

