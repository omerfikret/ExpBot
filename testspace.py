import sys, re, random, time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from fuzzywuzzy import fuzz

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1 — DÖNEM ETİKETLERİ (sadece referans, eğitimde kullanılmaz)
# ─────────────────────────────────────────────────────────────
# CSV'deki dönem sütunu kullanılacak. Bu sözlük sadece /donemler komutu için.
DONEM_ACIKLAMA = {
    "kuruluş":   "1299–1402  |  Osman Gazi'den Fetret Devri'ne",
    "fetret":    "1402–1453  |  Timur yenilgisinden İstanbul fethine",
    "yükseliş":  "1453–1566  |  Fatih'ten Kanuni'nin ölümüne",
    "duraklama": "1566–1699  |  II. Selim'den Karlofça'ya",
    "gerileme":  "1699–1839  |  Karlofça'dan Tanzimat'a",
    "çöküş":     "1839–1924  |  Tanzimat'tan Cumhuriyet'e",
}

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2 — STOP WORDS (değişmedi)
# ─────────────────────────────────────────────────────────────
STOP_WORDS = {
    "ve", "ile", "da", "de", "te", "ya", "ki", "bu", "bir",
    "o", "ama", "ancak", "fakat", "ne", "hem", "ya", "veya",
    "için", "ise", "bile", "dahi", "kadar", "gibi", "göre",
    "sonra", "önce", "ardından", "üzerine", "karşı", "olarak",
    "her", "hiç", "en", "çok", "az", "daha", "çok", "bazı",
    "tüm", "bütün", "son", "ilk", "yeni", "büyük", "küçük",
    "önemli", "kalıcı", "gerçek", "tam", "kısa",
    "etti", "oldu", "olarak", "aldı", "geldi", "geçti",
    "kurdu", "verdi", "çıktı", "başladı", "başlattı",
    "yapıldı", "edildi", "katıldı", "sağladı",
    "kaybetti", "yenildi", "kazandı", "bıraktı", "kaldı",
    "alındı", "indirildi", "çekildi", "durduruldu",
    "tanındı", "imzalandı", "ilan", "yaşandı", "sonuçlandı",
    "uğratıldı", "pekiştirildi", "tamamlandı", "sürdü",
    "kaldırıldı", "bastırıldı", "gönderildi",
    "i", "ı", "u", "ü", "a", "e",
    "nın", "nin", "nun", "nün",
    "nda", "nde", "nde", "nda",
    "nın", "daki", "deki", "taki", "teki",
    "ya", "ye", "yı", "yi", "yu", "yü",
    "un", "ün", "in", "ın",
    "stanbul", "ngiltere", "syanı",
    "ii", "iii",
    "hale", "getirdi",
    "ele", "geçirdi",
}

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3 — METİN NORMALİZASYONU (değişmedi)
# ─────────────────────────────────────────────────────────────
def _norm(t: str) -> str:
    t = str(t).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    tokenlar = t.split()
    tokenlar = [tok for tok in tokenlar if tok not in STOP_WORDS and len(tok) > 1]
    if len(tokenlar) < 2:
        ham = str(t).lower().strip()
        ham = re.sub(r"[^\w\s]", " ", ham)
        return re.sub(r"\s+", " ", ham).strip()
    return " ".join(tokenlar).strip()

# ─────────────────────────────────────────────────────────────
# BÖLÜM 4 — VERİ YÜKLEME (soru, cevap, dönem)
# ─────────────────────────────────────────────────────────────
def veri_yukle(yol: str) -> pd.DataFrame:
    """
    CSV formatı: 'soru'; 'cevap'; 'dönem' sütunları.
    """
    try:
        df = pd.read_csv(yol, encoding="utf-8")
    except FileNotFoundError:
        sys.exit(f"[HATA] '{yol}' bulunamadı.")
    except Exception as e:
        sys.exit(f"[HATA] CSV okunamadı: {e}")

    if not {"soru", "cevap", "dönem"}.issubset(df.columns):
        sys.exit("[HATA] CSV'de 'soru'; 'cevap'; 'dönem' sütunları bulunmalıdır.")

    df = df.dropna(subset=["soru", "cevap", "dönem"])
    df["soru"] = df["soru"].astype(str).str.strip()
    df["cevap"] = df["cevap"].astype(str).str.strip()
    df["dönem"] = df["dönem"].astype(str).str.strip().str.lower()
    df = df[df["soru"].str.len() >= 5]
    df = df[df["cevap"].str.len() >= 5]
    df["gn"] = df["soru"].apply(_norm)
    df = df[df["gn"].str.len() >= 3]
    df = df.drop_duplicates(subset=["gn"], keep="first").reset_index(drop=True)
    # Yeniden adlandırma: 'girdi' = soru
    df = df.rename(columns={"soru": "girdi"})
    return df[["girdi", "cevap", "dönem", "gn"]]

# ─────────────────────────────────────────────────────────────
# BÖLÜM 5 — LINEAR SVC EĞİTİMİ (doğrudan CSV'deki dönem sütunu kullanılır)
# ─────────────────────────────────────────────────────────────
def svc_egit(df: pd.DataFrame):
    """
    CSV'deki dönem sütununu hedef alarak LinearSVC eğitir.
    """
    vw = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        sublinear_tf=True, max_features=4000,
        stop_words=list(STOP_WORDS),
    )
    vc = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        sublinear_tf=True, max_features=4000,
    )
    Xw = vw.fit_transform(df["gn"])
    Xc = vc.fit_transform(df["gn"])
    X = hstack([Xw, Xc], format="csr")

    svc = LinearSVC(C=1.2, max_iter=3000, class_weight="balanced")
    svc.fit(X, df["dönem"])
    return svc, vw, vc

# ─────────────────────────────────────────────────────────────
# BÖLÜM 6 — ETİKET BAZLI HAVUZLAR (sadece metin, vektör yok)
# ─────────────────────────────────────────────────────────────
def havuz_olustur(df: pd.DataFrame):
    havuzlar = {}
    for donem in df["dönem"].unique():
        havuzlar[donem] = df[df["dönem"] == donem][["girdi", "cevap", "gn"]].reset_index(drop=True)
    return havuzlar

# ─────────────────────────────────────────────────────────────
# BÖLÜM 7 — FUZZY EŞLEME İLE CEVAP BULMA (fuzzywuzzy token_set_ratio)
# ─────────────────────────────────────────────────────────────
from fuzzywuzzy import fuzz

_BILMIYORUM = [
    "Bu konuda veri setimde bilgi bulamadım. Osmanlı tarihi hakkında başka bir şey sorabilirsiniz.",
    "Aradığınız bilgi veri setimde yok. Farklı bir konu dener misiniz?",
    "Bu soruya yanıt verecek kayıt bulamadım. Dönem adı ya da olay adıyla tekrar sorabilirsiniz.",
]

FUZZY_ESIK = 45   # token_set_ratio 0-100 arası, %45 altında cevap verme

def _fuzzy_ara(df: pd.DataFrame, gn_sorgu: str, esik: int):
    """
    fuzzywuzzy token_set_ratio ile sıralamadan bağımsız eşleme yapar.
    Skor 0-100 arası, en yüksek skorlu eşik üstü cevabı döndürür.
    """
    best_idx = -1
    best_score = 0
    for idx, row in df.iterrows():
        skor = fuzz.token_set_ratio(gn_sorgu, row["gn"])
        if skor > best_score:
            best_score = skor
            best_idx = idx
    if best_idx != -1 and best_score >= esik:
        return df.iloc[best_idx]["cevap"], best_score / 100.0
    return None, 0.0

def cevap_bul(girdi: str, havuzlar: dict, svc, vw, vc):
    gn = _norm(girdi)
    v = hstack([vw.transform([gn]), vc.transform([gn])], format="csr")
    donem = svc.predict(v)[0]

    if donem not in havuzlar:
        donem = random.choice(list(havuzlar.keys()))

    havuz = havuzlar[donem]
    cevap, skor = _fuzzy_ara(havuz, gn, FUZZY_ESIK)

    if cevap is not None:
        return donem, cevap, skor

    tum_df = pd.concat([h for h in havuzlar.values()], ignore_index=True)
    cevap, skor = _fuzzy_ara(tum_df, gn, FUZZY_ESIK)
    if cevap is not None:
        return "?", cevap, skor

    return "?", random.choice(_BILMIYORUM), 0.0

# ─────────────────────────────────────────────────────────────
# BÖLÜM 8 — YARDIM & ANA DÖNGÜ (küçük uyarlamalar)
# ─────────────────────────────────────────────────────────────
YARDIM = """
  /donemler          Tüm dönemleri ve satır sayılarını göster
  /goster <dönem>    O döneme ait örnek soruları listele
  /info              Model istatistikleri
  q / /cikis         Çıkış

  Örnek sorular:
    Osman Gazi ne zaman beyliği kurdu?
    Fatih İstanbul'u nasıl fethetti?
    Karlofça Antlaşması ne zaman imzalandı?
    Tanzimat Fermanı nedir?
    Çanakkale Savaşı hakkında ne biliyorsun?
"""

def main():
    print("\n" + "=" * 60)
    print("Osmanlı Tarihi ChatBot")
    print("=" * 60 + "\n")

    yol = sys.argv[1] if len(sys.argv) > 1 else "egitim_verisi_temiz.csv"

    t0 = time.perf_counter()
    print("[…] Veri yükleniyor…")
    df = veri_yukle(yol)
    print(f"[✓] {len(df)} soru-cevap çifti yüklendi")

    print("[…] Dönem sınıflandırıcısı eğitiliyor (LinearSVC)...")
    svc, vw, vc = svc_egit(df)
    dagilim = df["dönem"].value_counts()
    print(f"[✓] Dönem dağılımı:\n{dagilim.to_string()}")

    print("[…] Havuzlar oluşturuluyor…")
    havuzlar = havuz_olustur(df)
    print(f"[✓] {len(havuzlar)} dönem havuzu hazır")
    print(f"[✓] Süre: {time.perf_counter() - t0:.2f}s\n")
    print("Yardım için /yardim  |  Çıkış için q")
    print("-" * 60)

    while True:
        try:
            g = input("Sen : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGörüşmek üzere!")
            break

        if not g:
            continue
        gl = g.lower()

        if gl in ("q", "/cikis", "exit"):
            print("Bot : Görüşmek üzere!")
            break

        if gl in ("/yardim", "/help"):
            print(YARDIM)
            continue

        if gl == "/info":
            print(
                f"  Toplam soru    : {len(df)}\n"
                f"  Dönem sayısı   : {len(havuzlar)}\n"
                f"  Kelime vocab   : {len(vw.vocabulary_)}\n"
                f"  Char vocab     : {len(vc.vocabulary_)}"
            )
            continue

        if gl == "/donemler":
            print("\n  Dönem Dağılımı:")
            for et, acik in DONEM_ACIKLAMA.items():
                sayi = dagilim.get(et, 0)
                bar  = "█" * (sayi // 3)
                print(f"  {et:<12} {acik:<45} {sayi:>3} soru   {bar}")
            print()
            continue

        if gl.startswith("/goster "):
            et = gl[8:].strip()
            if et not in havuzlar:
                print(
                    f"Bot : '{et}' dönemi bulunamadı. "
                    f"Geçerli dönemler: {list(havuzlar.keys())}"
                )
            else:
                ornekler = havuzlar[et]["girdi"].head(6).tolist()
                print(f"\n  [{et}] dönemi örnek sorular:")
                for o in ornekler:
                    print(f"   • {o}")
                print()
            continue

        etiket, yanit, skor = cevap_bul(g, havuzlar, svc, vw, vc)
        etiket_goster = f"[{etiket}|{skor:.2f}]" if etiket != "?" else "[?]"
        print(f"Bot {etiket_goster}: {yanit}\n")

if __name__ == "__main__":
    main()