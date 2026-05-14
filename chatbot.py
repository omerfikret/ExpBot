"""
ML Tabanlı Türkçe ChatBot  v4.0
LinearSVC Etiketleme | Etiket-Havuz Sistemi | TF-IDF(word+char) | Token Sayacı
"""

import sys, re, random, time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, vstack
from collections import defaultdict

# ─────────────────────────────────────────
# BÖLÜM 0 — TOKEN SAYACI
# ─────────────────────────────────────────
# Gerçek BPE/WordPiece tokenizer yokken basit ama anlamlı bir
# yaklaşım: her kelimeyi ortalama 1.3 token (Türkçe eklemeli dil
# olduğu için İngilizce'ye göre daha yüksek), noktalama ve özel
# karakterleri ayrı birer token sayar.
# Bu tahmin, GPT-4 / Claude tokenizer'larına yaklaşık uyar.

def token_say(metin: str) -> int:
    """Heuristik Türkçe token tahmini."""
    # Noktalama / sayı ayrıştırması
    parcalar = re.findall(r"[a-zA-ZğüşıöçĞÜŞİÖÇ]+|[0-9]+|[^\s\w]", metin)
    sayim = 0
    for p in parcalar:
        if re.fullmatch(r"[a-zA-ZğüşıöçĞÜŞİÖÇ]+", p):
            # Uzun Türkçe kelimeler birden fazla token alabilir
            # Her 5 karakter ≈ 1 ekstra token (ek / kök ayrımı)
            sayim += 1 + max(0, (len(p) - 4) // 5)
        else:
            sayim += 1
    return max(1, sayim)

# --- bak ---

def token_ozeti(girdi: str, cevap: str) -> str:
    """Girdi + cevap için token istatistikleri döndürür."""
    g_tok = token_say(girdi)
    c_tok = token_say(cevap)
    return (f"[Token → Girdi: ~{g_tok}  |  Cevap: ~{c_tok}  |  "
            f"Toplam: ~{g_tok + c_tok}]")

# ─────────────────────────────────────────
# BÖLÜM 1 — NORMALIZASYON
# ─────────────────────────────────────────
def _norm(t: str) -> str:
    t = str(t).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ─────────────────────────────────────────
# BÖLÜM 2 — ETİKET TANIMLAMA
# ─────────────────────────────────────────
# Her etiket için çekirdek (seed) kelimeler tanımlanır.
# LinearSVC bu tohumlardan vektörel genelleme yapar;
# yani tohumda geçmeyen ama anlamca benzer girdileri de sınıflar.
# Tohumlar eğitim verisindeki terimlerle örtüşmeli.

ETIKETLER = {
    "sohbet":      ["merhaba", "selam", "nasılsın", "iyi günler", "teşekkür",
                    "günaydın", "hoşgeldin", "görüşürüz", "naber", "ne haber",
                    "hava", "günlük", "hayat", "his", "duygu", "yalnız",
                    "üzgün", "mutlu", "yorgun", "hayal", "ümit", "kaygı"],
    "yapay_zeka":  ["yapay zeka", "makine öğrenmesi", "derin öğrenme", "sinir ağı",
                    "chatgpt", "algoritma", "nlp", "veri bilimi", "otomasyon",
                    "robot öğrenmesi", "büyük dil modeli", "llm", "ai"],
    "teknoloji":   ["yazılım", "donanım", "kod", "programlama", "python",
                    "bilgisayar", "internet", "siber", "uygulama", "veritabanı",
                    "bulut", "blockchain", "iot", "nesnelerin interneti",
                    "akıllı saat", "telefon", "tablet", "işlemci"],
    "araba":       ["araba", "otomobil", "elektrikli araç", "motor", "fren",
                    "yakıt", "dizel", "benzin", "hybrid", "tesla", "trafik",
                    "otonom araç", "şarj", "emisyon", "egzoz"],
    "uzay":        ["uzay", "gezegen", "yıldız", "galaksi", "nasa", "roket",
                    "kara delik", "astrofizik", "kozmoloji", "mars", "ay",
                    "güneş sistemi", "meteor", "asteroid", "teleskop", "ışık yılı"],
    "fizik":       ["fizik", "kuantum", "enerji", "kuvvet", "ivme", "hız",
                    "elektromanyetizma", "termodinamik", "parçacık", "foton",
                    "atom", "nükleer", "görelilik", "einstein", "newton"],
    "tarih":       ["tarih", "osmanlı", "cumhuriyet", "savaş", "imparatorluk",
                    "antik", "roma", "mısır", "pers", "medeniyet", "çağ",
                    "devrim", "kolonizasyon", "iskender", "atatürk"],
    "saglik":      ["sağlık", "hastalık", "tedavi", "ilaç", "doktor", "beslenme",
                    "egzersiz", "spor", "diyet", "kilo", "kalori", "protein",
                    "vitamin", "mental", "uyku", "stres", "yoga"],
    "ekonomi":     ["ekonomi", "enflasyon", "döviz", "borsa", "faiz", "büyüme",
                    "stagflasyon", "resesyon", "gdp", "vergi", "bütçe",
                    "ticaret", "yatırım", "kripto", "bitcoin"],
    "sanat":       ["sanat", "müzik", "resim", "heykel", "sinema", "edebiyat",
                    "şiir", "roman", "pop art", "klasik", "opera", "tiyatro",
                    "dans", "fotoğraf", "warhol", "picasso"],
    "doga":        ["doğa", "iklim", "çevre", "deprem", "volkan", "okyanus",
                    "orman", "biyoloji", "evrim", "tür", "ekosistem",
                    "bitki", "hayvan", "enerji kaynağı", "sürdürülebilir"],
}

# ─────────────────────────────────────────
# BÖLÜM 3 — VERİ YÜKLEME (chunk)
# ─────────────────────────────────────────
CHUNK_N = 256

def veri_yukle(yol: str) -> pd.DataFrame:
    try:
        reader = pd.read_csv(yol, encoding="utf-8",
                             chunksize=CHUNK_N, on_bad_lines="skip")
    except FileNotFoundError:
        sys.exit(f"[HATA] '{yol}' bulunamadı.")

    gorduk, parclar = set(), []
    for chunk in reader:
        chunk.columns = chunk.columns.str.strip().str.lower()
        if "girdi" not in chunk.columns or "cevap" not in chunk.columns:
            continue
        chunk = (chunk.dropna(subset=["girdi","cevap"])
                      .assign(girdi=chunk["girdi"].str.strip(),
                              cevap=chunk["cevap"].str.strip()))
        chunk["gn"] = chunk["girdi"].apply(_norm)
        chunk = chunk[chunk["gn"].str.len() >= 2]
        chunk = chunk[chunk["cevap"].str.len() >= 5]
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
# BÖLÜM 4 — LINEAR SVC ETİKETLEME
# ─────────────────────────────────────────
# Nasıl çalışır?
#
# 1. Her etiketin tohum cümlelerini "yapay eğitim seti" olarak kullan.
# 2. Gerçek veriyi YALNIZCA tahmin için (predict) kullan.
# 3. LinearSVC her girdi için vektör uzayında en yakın sınıf
#    hiperplanını bulur; yani tohum kelimeler vektörü o yönde iter.
#
# LinearSVC seçim nedeni:
#  • Yüksek boyutlu TF-IDF matrislerinde en iyi genelleme yapan model
#  • LogisticRegression'dan daha hızlı eğitim, benzer doğruluk
#  • Tohum-veri eğitiminde aşırı öğrenmeyi önler (margin maximization)

def svc_etiketle(df: pd.DataFrame):
    """LinearSVC ile her satıra etiket ata."""
    
    # ── Tohum veri seti oluştur ──
    tohum_giris, tohum_etiket = [], []
    for etiket, kelimeler in ETIKETLER.items():
        for k in kelimeler:
            # Her tohum kelimeyi birkaç varyasyonla genişlet
            tohum_giris.extend([k, f"{k} nedir", f"{k} hakkında", 
                                 f"{k} nasıl", f"{k} neden"])
            tohum_etiket.extend([etiket] * 5)

    # ── Birleşik vektörleştirici (tohum + gerçek veri) ──
    # Fit'i tüm veri üzerinde yaparız ki OOV (out-of-vocabulary)
    # sorununu önleyelim; sadece tahmin aşamasında gerçek veriyi veririz.
    tum_metin = [_norm(g) for g in tohum_giris] + df["gn"].tolist()

    vw = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                         sublinear_tf=True, max_features=3000)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,3),
                         sublinear_tf=True, max_features=3000)
    vw.fit(tum_metin); vc.fit(tum_metin)

    def _vec(metinler):
        return hstack([vw.transform(metinler),
                       vc.transform(metinler)], format="csr")

    X_tohum = _vec([_norm(g) for g in tohum_giris])
    X_gercek = _vec(df["gn"].tolist())

    # ── LinearSVC eğit ──
    svc = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
    svc.fit(X_tohum, tohum_etiket)

    df["etiket"] = svc.predict(X_gercek)
    return df, (vw, vc), svc

# ─────────────────────────────────────────
# BÖLÜM 5 — ETİKET BAZLI VEKTÖRLEŞTİRME
# ─────────────────────────────────────────
# Her etiket için ayrı TF-IDF matrisi tutulur.
# Neden?
#  • "fizik nedir" sorusu "sohbet" havuzuyla değil yalnızca
#    "fizik" havuzuyla karşılaştırılır → gürültü düşer, doğruluk artar
#  • Havuzlar küçük olduğundan bellek etkisi minimumdur

def etiket_matrisleri_olustur(df: pd.DataFrame, vecs):
    """Her etiket için satır indeksleri ve sparse matris döndür."""
    vw, vc = vecs
    havuzlar = {}
    for etiket in df["etiket"].unique():
        alt = df[df["etiket"] == etiket].copy()
        Mw = vw.transform(alt["gn"])
        Mc = vc.transform(alt["gn"])
        M  = hstack([Mw, Mc], format="csr")
        havuzlar[etiket] = {"df": alt.reset_index(drop=True), "M": M}
    return havuzlar

def kullanici_vec(girdi_norm, vecs):
    vw, vc = vecs
    return hstack([vw.transform([girdi_norm]),
                   vc.transform([girdi_norm])], format="csr")

# ─────────────────────────────────────────
# BÖLÜM 6 — ETİKET TAHMİNİ (inference)
# ─────────────────────────────────────────
def etiket_tahmin(girdi_norm: str, svc, vecs) -> str:
    """LinearSVC ile girdi etiketini tahmin et."""
    v = kullanici_vec(girdi_norm, vecs)
    return svc.predict(v)[0]

# ─────────────────────────────────────────
# BÖLÜM 7 — CEVAP BULMA
# ─────────────────────────────────────────
_BILMIYORUM = [
    "Bunu anlayamadım, farklı sormayı dener misiniz?",
    "Bu konuda eğitilmedim, başka şey sorabilirsiniz.",
    "Tam anlayamadım, biraz daha açar mısınız?",
]

def cevap_bul(girdi: str, havuzlar: dict, svc, vecs,
              esik: float = 0.4, top_k: int = 3) -> tuple[str, str, float]:
    """
    1. LinearSVC ile etiket tahmin et
    2. Yalnızca o etiketin havuzunda kosinüs benzerliği ara
    3. (etiket, cevap_metni, skor) döndür
    """
    gn = _norm(girdi)
    etiket = etiket_tahmin(gn, svc, vecs)

    # Etiket havuzda yoksa (uç durum) rastgele seç
    if etiket not in havuzlar:
        etiket = random.choice(list(havuzlar.keys()))

    havuz = havuzlar[etiket]
    v  = kullanici_vec(gn, vecs)
    sk = cosine_similarity(v, havuz["M"])[0]
    idx = np.argsort(sk)[::-1][:top_k]
    skor = sk[idx[0]]

    if skor < esik:
        # Eşiği geçemediyse tüm havuzlarda dene (fallback)
        tum_df, tum_M = _tum_havuz(havuzlar)
        sk2 = cosine_similarity(v, tum_M)[0]
        idx2 = np.argsort(sk2)[::-1][:top_k]
        skor2 = sk2[idx2[0]]
        if skor2 < esik:
            return "?", random.choice(_BILMIYORUM), 0.0
        etiket = tum_df.iloc[idx2[0]]["etiket"]
        ana = tum_df.iloc[idx2[0]]["cevap"]
        return etiket, ana, float(skor2)

    ana = havuz["df"].iloc[idx[0]]["cevap"]

    # İkincil önerileri ekle (skor düşükse)
    if skor < 0.40 and len(idx) > 1 and sk[idx[1]] >= esik:
        iki = havuz["df"].iloc[idx[1]]["cevap"]
        if iki != ana:
            return etiket, f"{ana} — Ayrıca: {iki}", float(skor)

    return etiket, ana, float(skor)

# Tüm havuzları birleştir (fallback için)
def _tum_havuz(havuzlar: dict):
    dfs = []
    Ms  = []
    for et, h in havuzlar.items():
        d = h["df"].copy(); d["etiket"] = et
        dfs.append(d); Ms.append(h["M"])
    df_all = pd.concat(dfs, ignore_index=True)
    M_all  = vstack(Ms, format="csr")
    return df_all, M_all

# ─────────────────────────────────────────
# BÖLÜM 8 — YARDIM & ANA DÖNGÜ
# ─────────────────────────────────────────
YARDIM = """
  /etiketler           Tüm etiketleri ve satır sayısını göster
  /token               Token sayacını aç/kapat
  /goster <etiket>     O etiketteki örnek girdileri listele
  /info                Model istatistikleri
  q / /cikis           Çıkış
"""

def main():
    print("\n" + "="*58)
    print("  ML Tabanlı Türkçe ChatBot  v4.0")
    print("  LinearSVC Etiket | Etiket-Havuz | word+char TF-IDF")
    print("="*58+"\n")

    yol = sys.argv[1] if len(sys.argv) > 1 else "egitim_verisi_temiz.csv"

    t0 = time.perf_counter()
    print("[…] Veri yükleniyor…")
    df = veri_yukle(yol)
    print(f"[✓] {len(df)} satır yüklendi")

    print("[…] LinearSVC etiketleme başlıyor…")
    df, vecs, svc = svc_etiketle(df)
    dagılım = df["etiket"].value_counts()
    print(f"[✓] Etiket dağılımı:\n{dagılım.to_string()}")

    print("[…] Etiket havuzları oluşturuluyor…")
    havuzlar = etiket_matrisleri_olustur(df, vecs)
    print(f"[✓] {len(havuzlar)} havuz hazır")
    print(f"[✓] Süre: {time.perf_counter()-t0:.2f}s\n")
    print("Yardım için /yardim  |  Çıkış için q")
    print("-"*58)

    token_mod = True   # Başlangıçta token gösterimi açık

    while True:
        try:
            g = input("Sen : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGörüşmek üzere!"); break

        if not g:
            continue
        gl = g.lower()

        if gl in ("q", "/cikis", "exit"):
            print("Bot : Görüşmek üzere!"); break

        if gl in ("/yardim", "/help"):
            print(YARDIM); continue

        if gl == "/info":
            print(f"  Satır={len(df)}  Etiket={len(havuzlar)}  "
                  f"Kelime-vocab={len(vecs[0].vocabulary_)}")
            continue

        if gl == "/etiketler":
            print("\n  Etiket Dağılımı:")
            for et, s in dagılım.items():
                bar = "█" * (s // 5)
                print(f"  {et:<16} {s:>4} satır  {bar}")
            print()
            continue

        if gl == "/token":
            token_mod = not token_mod
            print(f"Bot : Token sayacı {'AÇIK ✅' if token_mod else 'KAPALI ❌'}")
            continue

        if gl.startswith("/goster "):
            et = gl[8:].strip()
            if et not in havuzlar:
                print(f"Bot : '{et}' etiketi bulunamadı. "
                      f"Etiketler: {list(havuzlar.keys())}")
            else:
                ornekler = havuzlar[et]["df"]["girdi"].head(8).tolist()
                print(f"\n  [{et}] örnekleri:")
                for o in ornekler:
                    print(f"   • {o}")
                print()
            continue

        etiket, yanit, skor = cevap_bul(g, havuzlar, svc, vecs)
        etiket_goster = f"[{etiket}|{skor:.2f}]"
        print(f"Bot {etiket_goster}: {yanit}")

        if token_mod:
            print("  " + token_ozeti(g, yanit))

if __name__ == "__main__":
    main()
