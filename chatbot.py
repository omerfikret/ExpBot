"""
ML Tabanlı Türkçe ChatBot  v3.0
Pandas(chunksize) | TF-IDF(word+char hstack) | NumPy | Bigram
"""

import sys, re, random, time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from collections import defaultdict

# ─────────────────────────────────────────
# YARDIMCI
# ─────────────────────────────────────────
def _norm(t: str) -> str:
    t = str(t).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ─────────────────────────────────────────
# BÖLÜM 1 — CHUNK BAZLI OKUMA
# ─────────────────────────────────────────
# pd.read_csv() normalde tüm dosyayı tek seferde RAM'e yükler.
# chunksize=N ise dosyayı N satırlık dilimler halinde okuyup
# iterator döndürür; her dilim ayrı işlenir. Böylece bellek
# kullanımı O(toplam) yerine O(chunk) sabit kalır.
CHUNK_N = 256

def veri_yukle(yol: str) -> pd.DataFrame:
    try:
        reader = pd.read_csv(yol, encoding="utf-8",
                             chunksize=CHUNK_N, on_bad_lines="skip")
    except FileNotFoundError:
        sys.exit(f"[HATA] '{yol}' bulunamadı.")

    gorduk  = set()   # O(1) duplicate lookup
    parclar = []

    for chunk in reader:
        chunk.columns = chunk.columns.str.strip().str.lower()
        if "girdi" not in chunk.columns or "cevap" not in chunk.columns:
            continue

        chunk = (chunk
                 .dropna(subset=["girdi","cevap"])
                 .assign(girdi=chunk["girdi"].str.strip(),
                         cevap=chunk["cevap"].str.strip()))
        chunk["gn"] = chunk["girdi"].apply(_norm)

        # chunk içinde hızlı filtre
        chunk = chunk[chunk["gn"].str.len() >= 2]
        chunk = chunk[chunk["cevap"].str.len() >= 10]
        chunk = chunk[~chunk["gn"].isin(gorduk)]
        gorduk.update(chunk["gn"].tolist())
        parclar.append(chunk[["girdi","cevap","gn"]])

    if not parclar:
        sys.exit("[HATA] CSV boş veya sütun yapısı hatalı.")

    df = pd.concat(parclar, ignore_index=True)
    df.drop_duplicates("gn", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ─────────────────────────────────────────
# BÖLÜM 2 — VEKTÖRLEŞTİRME  (word + char)
# ─────────────────────────────────────────
# Neden iki vektörizer birden?
#
#  SORUN — sadece char_wb(1,3) kullanınca:
#   "yapay zeka nedir" ←→ "Yapay zeka nedir?" → cosine ≈ 1.00
#   Çünkü karakter n-gramları büyük/küçük harf ve noktalama farkını
#   görmezden gelir; "nasıl kullanılıyor" ile "nasıl kullanılır"
#   neredeyse aynı karakter dizisi → yanlış eşleşme.
#
#  ÇÖZÜM — word(1,2) + char_wb(2,3) scipy.sparse.hstack ile birleştir:
#   • word n-gram → kelime düzeyinde ANLAM farkını yakalar
#     ("kullanılıyor" ≠ "kullanılır" → farklı token)
#   • char n-gram → yazım hatası toleransı sağlar
#   • hstack → iki seyrek matris RAM'de birleşir, yoğunlaştırmaz
#   • max_features tavanı → matris boyutu kontrol altında

def vektorlestir(df: pd.DataFrame):
    vw = TfidfVectorizer(analyzer="word",    ngram_range=(1,2),
                         sublinear_tf=True,  max_features=2500)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,3),
                         sublinear_tf=True,  max_features=2500)
    Mw = vw.fit_transform(df["gn"])   # seyrek CSR
    Mc = vc.fit_transform(df["gn"])   # seyrek CSR
    M  = hstack([Mw, Mc], format="csr")   # birleşik, hâlâ seyrek
    return (vw, vc), M

def kullanici_vec(girdi_norm, vecs):
    vw, vc = vecs
    return hstack([vw.transform([girdi_norm]),
                   vc.transform([girdi_norm])], format="csr")

# ─────────────────────────────────────────
# BÖLÜM 3 — BİGRAM ÜRETİCİ
# ─────────────────────────────────────────
class Uretici:
    def __init__(self):
        self.bg: dict = defaultdict(list)

    def egit(self, df):
        for c in df["cevap"]:
            ws = str(c).split()
            for i in range(len(ws)-1):
                self.bg[ws[i].lower()].append(ws[i+1])

    def uret(self, anahtar, maks=12):
        ak = anahtar.lower()
        bas = ak if ak in self.bg else next(
              (k for k in self.bg if ak in k or k in ak), None)
        if not bas:
            return None
        c = [bas]; sim = bas
        for _ in range(maks-1):
            sec = self.bg.get(sim.lower(), [])
            if not sec: break
            sim = random.choice(sec); c.append(sim)
            if sim[-1] in ".!?": break
        m = " ".join(c)
        return (m if m[-1] in ".!?" else m+".").capitalize()

# ─────────────────────────────────────────
# BÖLÜM 4 — CEVAP BULMA
# ─────────────────────────────────────────
_BILMIYORUM = [
    "Bunu anlayamadım, farklı sormayı dener misiniz?",
    "Bu konuda eğitilmedim, başka şey sorabilirsiniz.",
    "Tam anlayamadım, biraz daha açar mısınız?",
]

def cevap(girdi, df, vecs, M, uretici,
          esik=0.30, top_k=3, uretim=False):
    gn = _norm(girdi)
    v  = kullanici_vec(gn, vecs)
    sk = cosine_similarity(v, M)[0]               # ndarray (n,)
    idx = np.argsort(sk)[::-1][:top_k]
    skor = sk[idx[0]]

    if skor < esik:
        return random.choice(_BILMIYORUM)

    ana = df.loc[idx[0], "cevap"]

    if uretim:
        for k in [w for w in gn.split() if len(w) > 3]:
            u = uretici.uret(k)
            if u and u.lower() != ana.lower():
                return f"{ana}\n  ↳ Üretilen: {u}"

    if skor < 0.45 and len(idx) > 1 and sk[idx[1]] >= esik:
        iki = df.loc[idx[1], "cevap"]
        if iki != ana:
            return f"{ana} — Ayrıca: {iki}"

    return ana

# ─────────────────────────────────────────
# BÖLÜM 5 — ANA DÖNGÜ
# ─────────────────────────────────────────
YARDIM = """
  /uretim ac|kapat   Bigram üretimini aç/kapat
  /uret <kelime>     Kelimeden cümle üret
  /info              Model istatistikleri
  q / /cikis         Çıkış
"""

def main():
    print("\n" + "="*52)
    print("  ML Tabanlı Türkçe ChatBot  v3.0")
    print("  Pandas(chunk) | word+char TF-IDF | Bigram")
    print("="*52+"\n")

    yol = sys.argv[1] if len(sys.argv) > 1 else "egitim_verisi_temiz.csv"

    t0 = time.perf_counter()
    df = veri_yukle(yol)
    print(f"[✓] {len(df)} satır yüklendi")
    vecs, M = vektorlestir(df)
    print(f"[✓] Matris: {M.shape[0]}×{M.shape[1]}  "
          f"doluluk %{M.nnz/(M.shape[0]*M.shape[1])*100:.2f}")
    ur = Uretici(); ur.egit(df)
    print(f"[✓] Bigram: {len(ur.bg)} kelime")
    print(f"[✓] Hazır  ({time.perf_counter()-t0:.2f}s)\n")
    print("Yardım için /yardim  |  Çıkış için q")
    print("-"*52)

    mod = False
    while True:
        try:
            g = input("Sen : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGörüşmek üzere!"); break

        if not g: continue
        gl = g.lower()

        if gl in ("q","/cikis","exit"):
            print("Bot : Görüşmek üzere!"); break
        if gl == "/yardim":
            print(YARDIM); continue
        if gl == "/info":
            print(f"  Satır={len(df)}  Matris={M.shape}  "
                  f"Bigram={len(ur.bg)}"); continue
        if "/uretim" in gl:
            mod = "ac" in gl or "aç" in gl
            print(f"Bot : Üretim {'AÇIK ✅' if mod else 'KAPALI ❌'}"); continue
        if gl.startswith("/uret "):
            print(f"Bot : {ur.uret(g[6:].strip()) or 'Üretemedi.'}"); continue

        print(f"Bot : {cevap(g, df, vecs, M, ur, uretim=mod)}")
if __name__ == "__main__":
    main()