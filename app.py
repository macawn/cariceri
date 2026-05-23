import pickle
import os
import string
import itertools
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_CRAWLING  = os.path.join(BASE_DIR, "hasil_crawling.pkl")
PATH_PROCESSED = os.path.join(BASE_DIR, "processed_news.pkl")
PATH_THESAURUS = os.path.join(BASE_DIR, "thesaurus.pkl")

paper           = []
processed_paper = []
thesaurus       = {}
vectorizer      = None
tfidf_matrix    = None


def muat_data():
    global paper, processed_paper, thesaurus, vectorizer, tfidf_matrix

    if not os.path.exists(PATH_CRAWLING):
        print(f"[ERROR] File tidak ditemukan: {PATH_CRAWLING}")
        return
    with open(PATH_CRAWLING, "rb") as f:
        raw = pickle.load(f)
    paper = raw.values.tolist() if hasattr(raw, 'values') else raw
    print(f"[INFO] hasil_crawling.pkl dimuat → {len(paper)} dokumen")

    if not os.path.exists(PATH_PROCESSED):
        print(f"[ERROR] File tidak ditemukan: {PATH_PROCESSED}")
        return
    with open(PATH_PROCESSED, "rb") as f:
        raw_processed = pickle.load(f)
    processed_paper = raw_processed.values.tolist() if hasattr(raw_processed, 'values') else raw_processed
    print(f"[INFO] processed_news.pkl dimuat → {len(processed_paper)} teks")

    if os.path.exists(PATH_THESAURUS):
        with open(PATH_THESAURUS, "rb") as f:
            thesaurus = pickle.load(f)
        print(f"[INFO] thesaurus.pkl dimuat → {len(thesaurus)} kata")
    else:
        print("[WARNING] thesaurus.pkl tidak ditemukan.")

    vectorizer   = TfidfVectorizer(use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(processed_paper)
    print(f"[INFO] TF-IDF matrix siap → shape: {tfidf_matrix.shape}")


def preprocess_query(query_text):
    text = query_text.lower()
    remove_punct = dict((ord(c), None) for c in string.punctuation)
    text = text.translate(remove_punct)
    tokens = text.split()
    return tokens


def doc_to_dict(doc, idx, skor=None, query_used=None, isSearch=False):
    import math

    def bersih(val):
        if val is None:
            return ""
        try:
            if math.isnan(float(val)):
                return ""
        except:
            pass
        return str(val)

    return {
        "no"        : bersih(doc[0]),
        "judul"     : bersih(doc[2]),
        "link"      : bersih(doc[1]),
        "tanggal"   : bersih(doc[4]),
        "preview"   : bersih(doc[3])[:300] + "...",
        "skor"      : round(float(skor), 4) if skor is not None else None,
        "query_used": query_used
    }


def cari_tanpa_ekspansi(query_tokens, top_n=10):
    query_str = " ".join(query_tokens)
    query_vec = vectorizer.transform([query_str])
    skor      = cosine_similarity(query_vec, tfidf_matrix).flatten()
    urutan    = np.argsort(skor)[::-1]
    hasil = []
    for i in urutan[:top_n]:
        if skor[i] > 0:
            hasil.append(doc_to_dict(paper[i], i, skor[i], query_str))
    return hasil


def cari_dengan_ekspansi(query_tokens, top_n=10):
    list_synonym = []
    for q in query_tokens:
        if q in thesaurus:
            sinonim = list(dict.fromkeys([q] + thesaurus[q]))[:3]
            list_synonym.append(sinonim)
        else:
            list_synonym.append([q])

    kombinasi = [" ".join(combo) for combo in itertools.product(*list_synonym)]
    print(f"[INFO] Kombinasi ekspansi: {kombinasi}")

    max_result = {}
    for query_str in kombinasi:
        query_vec = vectorizer.transform([query_str])
        skor      = cosine_similarity(query_vec, tfidf_matrix).flatten()
        for i, s in enumerate(skor):
            if s > 0:
                if i not in max_result or s > max_result[i]["skor"]:
                    max_result[i] = {"skor": s, "query_used": query_str}

    terurut = sorted(max_result.items(), key=lambda x: x[1]["skor"], reverse=True)
    hasil = []
    for i, info in terurut[:top_n]:
        hasil.append(doc_to_dict(paper[i], i, info["skor"], info["query_used"]))
    return hasil

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/cari", methods=["POST"])
def cari():
    try:
        if vectorizer is None:
            return jsonify({"error": "Data belum dimuat, cek pickle files."}), 500

        data     = request.get_json()
        query    = data.get("query", "").strip()
        ekspansi = data.get("ekspansi", False)
        semua    = data.get("semua", False)

        if isinstance(semua, str):
            semua = semua.lower() == "true"

        if semua:
            hasil = []
            for i, doc in enumerate(paper):
                hasil.append(doc_to_dict(doc, i))
            return jsonify({"query": "", "ekspansi": False, "jumlah": len(hasil), "hasil": hasil})

        if not query:
            return jsonify({"query": "", "ekspansi": False, "jumlah": 0, "hasil": []})

        tokens = preprocess_query(query)
        if ekspansi and thesaurus:
            hasil = cari_dengan_ekspansi(tokens, top_n=10)
        else:
            hasil = cari_tanpa_ekspansi(tokens, top_n=10)

        return jsonify({"query": query, "ekspansi": ekspansi, "jumlah": len(hasil), "hasil": hasil})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    
@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "total_dokumen"  : len(paper),
        "total_vocab"    : tfidf_matrix.shape[1] if tfidf_matrix is not None else 0,
        "thesaurus_words": len(thesaurus),
        "status"         : "siap" if vectorizer is not None else "belum dimuat"
    })

muat_data()
if __name__ == "__main__":
    app.run(debug=False)
