import re
import pickle
import streamlit as st
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Inisialisasi stemmer & stopword
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

factory_sw = StopWordRemoverFactory()
stopwords = set(factory_sw.get_stop_words())

# Kata penting untuk sentimen
sentiment_words = {'tidak', 'bukan', 'belum', 'kurang', 'terlalu', 'kecewa', 'senang', 
                   'marah', 'kesal', 'puas', 'muas', 'parah', 'error', 'gagal',
                   'ok', 'oke', 'bagus', 'baik', 'bisa', 'mantap', 'jelek', 'buruk', 
                   'lama', 'cepat', 'lambat', 'hanya', 'malah', 'sempat', 'sering', 
                   'jarang', 'susah', 'mudah', 'mampu'}

# Load kamus normalisasi offline
with open("kamus.pkl", "rb") as f:
    kamus = pickle.load(f)

def hapus_stopwords(tokens):
    return [kata for kata in tokens if (kata in sentiment_words) or (kata not in stopwords)]

def stemming(tokens):
    return [stemmer.stem(kata) for kata in tokens]

def preprocess_text(text, tokenizer, max_len, return_tokens=False):
    if not isinstance(text, str):
        text = ""

    # 1. Case folding
    text = text.lower().strip()

    # 2. Hapus URL
    text = re.sub(r'http\S+|www.\S+', '', text)

    # 3. Hapus angka jika perlu (training hapus)
    text = ''.join(char for char in text if not char.isdigit())

    # 4. Hapus tanda baca & karakter non-ASCII
    text = ''.join(char for char in text if char.isalnum() or char.isspace())

    # 5. Hapus spasi berlebih
    text = ' '.join(text.split())

    # 6. Normalisasi
    words = []
    for kata in text.split():
        if kata in kamus:
            words.append(kamus[kata])
        else:
            words.append(kata)

    # 7. Stopword removal & stemming
    tokens = hapus_stopwords(words)
    tokens = stemming(tokens)

    # 8. Gabungkan kembali menjadi string agar sesuai dengan training
    processed_text = ' '.join(tokens)

    # 9. Token ke sequence + padding
    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    if return_tokens:
        return tokens, seq, padded
    else:
        return seq, padded
