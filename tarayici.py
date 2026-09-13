"""
BIST T / P / N tarayicisi — Pine gostergesinin Python karsiligi.
Tum BIST hisselerini tarar, AL / TUT / SAT / BEKLE durumu uretir.

Veri kaynagi: yfinance (BIST sembolleri '.IS' ekiyle, or. THYAO.IS)
Kullanim:    python tarayici.py
Cikti:       sonuclar.csv  +  ozet.md  (+ istege bagli Telegram mesaji)
"""

from __future__ import annotations

import os
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────── Ayarlar (Pine input'larinin karsiligi) ───────────────────────────
CFG = {
    "rvol_len":   10,      # Goreceli hacim ortalama uzunlugu
    "rvol_min":   1.2,     # Goreceli hacim esigi
    "roll_len":   20,      # Kayan VWAP uzunlugu
    "t_yon":      "yukari",          # "yukari" | "asagi"
    "giris_modu": "T veya P",        # "Sadece T" | "Sadece P" | "T veya P" | "T + P"
    "cikis_modu": "N veya kural",    # "Sadece N" | "Sadece kural" | "N veya kural" | "N ve kural"
    "ard_len":    3,       # Kural 1 — ust uste dusen bar sayisi
    "zayif_len":  2,       # Kural 2 — endeksten zayif kalinan bar sayisi
    "zayif_sart": True,    # Kural 2'de hissenin de dusmus olmasi sarti
    "endeks":     "XU100.IS",
    "gun":        400,     # Kac gunluk gecmis cekilecek (durum makinesi icin gerekli)
    "min_hacim":  1_000_000,   # TL bazli ortalama islem hacmi filtresi (0 = kapali)
}

DIZIN = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────── Gosterge yardimcilari ───────────────────────────
def rsi(seri: pd.Series, uzunluk: int) -> pd.Series:
    """Wilder RSI — Pine'daki ta.rsi() ile ayni (RMA tabanli)."""
    fark = seri.diff()
    kazanc = fark.clip(lower=0)
    kayip = (-fark).clip(lower=0)
    ort_kazanc = kazanc.ewm(alpha=1 / uzunluk, adjust=False, min_periods=uzunluk).mean()
    ort_kayip = kayip.ewm(alpha=1 / uzunluk, adjust=False, min_periods=uzunluk).mean()
    rs = ort_kazanc / ort_kayip.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def sinyalleri_hesapla(df: pd.DataFrame, endeks_getiri: pd.Series) -> pd.DataFrame:
    """f_motor + f_zayif + f_cikis fonksiyonlarinin Python karsiligi."""
    d = df.copy()
    hlc3 = (d["High"] + d["Low"] + d["Close"]) / 3

    # Kayan (rolling) VWAP
    n = CFG["roll_len"]
    d["vwap"] = (hlc3 * d["Volume"]).rolling(n).sum() / d["Volume"].rolling(n).sum()

    # Goreceli hacim
    d["rvol"] = d["Volume"] / d["Volume"].rolling(CFG["rvol_len"]).mean()
    hacim_ok = d["rvol"] > CFG["rvol_min"]

    r3, r5, r10 = rsi(d["Close"], 3), rsi(d["Close"], 5), rsi(d["Close"], 10)

    ust = (d["Close"] > d["vwap"]) & (d["Close"].shift(1) <= d["vwap"].shift(1))
    alt = (d["Close"] < d["vwap"]) & (d["Close"].shift(1) >= d["vwap"].shift(1))
    kesis = ust if CFG["t_yon"] == "yukari" else alt

    d["T"] = kesis & hacim_ok & (r5 >= r10)
    d["P"] = (d["Close"] > d["vwap"]) & hacim_ok & (r5 >= r10)
    d["N"] = (d["Close"] < d["vwap"]) & hacim_ok & (r3 < r5)

    mod = CFG["giris_modu"]
    d["giris"] = (d["T"] if mod == "Sadece T"
                  else d["P"] if mod == "Sadece P"
                  else (d["T"] | d["P"]) if mod == "T veya P"
                  else (d["T"] & d["P"]))

    # Kural 1 — ard_len bar ust uste dusus
    dusus = (d["Close"] < d["Close"].shift(1)).astype(int)
    d["kural1"] = dusus.rolling(CFG["ard_len"]).sum() == CFG["ard_len"]

    # Kural 2 — zayif_len bar ust uste endeksten kotu performans
    getiri = d["Close"].pct_change() * 100
    e = endeks_getiri.reindex(d.index)          # tarih bazli hizalama
    zayif_bar = getiri < e
    if CFG["zayif_sart"]:
        zayif_bar &= getiri < 0
    d["kural2"] = zayif_bar.rolling(CFG["zayif_len"]).sum() == CFG["zayif_len"]

    kural = d["kural1"] | d["kural2"]
    cmod = CFG["cikis_modu"]
    d["cikis"] = (d["N"] if cmod == "Sadece N"
                  else kural if cmod == "Sadece kural"
                  else (d["N"] | kural) if cmod == "N veya kural"
                  else (d["N"] & kural))
    return d


def durum_makinesi(d: pd.DataFrame) -> dict:
    """f_durum() karsiligi — son bardaki pozisyon durumunu dondurur."""
    poz = False
    durum = 0                    # 0 bekle, 1 AL, 2 SAT, 3 tut
    giris_fiyat = np.nan
    giris_index = 0
    giris_tarih = None

    girisler = d["giris"].to_numpy()
    cikislar = d["cikis"].to_numpy()
    kapanislar = d["Close"].to_numpy()

    for i in range(len(d)):
        if poz and cikislar[i]:
            poz, durum = False, 2
            # giris bilgisini SILMIYORUZ — SAT satirinda raporlanacak
        elif (not poz) and girisler[i]:
            poz, durum = True, 1
            giris_fiyat, giris_index = kapanislar[i], i
            giris_tarih = d.index[i]
        else:
            durum = 3 if poz else 0

    son = kapanislar[-1]
    # AL/TUT'ta acik pozisyon, SAT'ta yeni kapanan islem raporlanir
    raporla = durum in (1, 2, 3) and not np.isnan(giris_fiyat)
    return {
        "durum": {0: "BEKLE", 1: "AL", 2: "SAT", 3: "TUT"}[durum],
        "bar": (len(d) - 1 - giris_index) if raporla else 0,
        "giris_fiyat": round(float(giris_fiyat), 2) if raporla else None,
        "giris_tarih": giris_tarih.date().isoformat() if (raporla and giris_tarih is not None) else None,
        "fiyat": round(float(son), 2),
        "kz_yuzde": round(float((son - giris_fiyat) / giris_fiyat * 100), 2) if raporla else None,
    }


# ─────────────────────────── Veri katmani ───────────────────────────
def sembolleri_oku(dosya: str = "semboller.txt") -> list[str]:
    yol = os.path.join(DIZIN, dosya)
    with open(yol, encoding="utf-8") as f:
        ham = [s.strip().upper() for s in f if s.strip() and not s.startswith("#")]
    return [s if s.endswith(".IS") else f"{s}.IS" for s in ham]


def veri_indir(semboller: list[str], gun: int, parca: int = 40) -> dict[str, pd.DataFrame]:
    """yfinance'ten toplu OHLCV indirir. Buyuk listeler parcalara bolunur."""
    sonuc: dict[str, pd.DataFrame] = {}
    baslangic = (dt.date.today() - dt.timedelta(days=int(gun * 1.6))).isoformat()

    for i in range(0, len(semboller), parca):
        grup = semboller[i:i + parca]
        print(f"  indiriliyor {i + 1}-{i + len(grup)} / {len(semboller)}", flush=True)
        try:
            ham = yf.download(grup, start=baslangic, auto_adjust=True,
                              group_by="ticker", progress=False, threads=True)
        except Exception as hata:
            print(f"  ! grup indirilemedi: {hata}")
            continue

        for s in grup:
            try:
                df = ham[s] if isinstance(ham.columns, pd.MultiIndex) else ham
                df = df.dropna(subset=["Close", "Volume"])
                if len(df) >= CFG["roll_len"] + CFG["ard_len"] + 5:
                    sonuc[s] = df
            except KeyError:
                pass
        time.sleep(1)          # nazik olalim, rate limit yemeyelim
    return sonuc


# ─────────────────────────── Ana akis ───────────────────────────
def tara() -> pd.DataFrame:
    semboller = sembolleri_oku()
    print(f"{len(semboller)} sembol taranacak")

    endeks = yf.download(CFG["endeks"], period="2y", auto_adjust=True, progress=False)
    if isinstance(endeks.columns, pd.MultiIndex):
        endeks.columns = endeks.columns.droplevel(1)
    endeks_getiri = endeks["Close"].pct_change() * 100

    veriler = veri_indir(semboller, CFG["gun"])
    print(f"{len(veriler)} sembol icin veri alindi")

    satirlar = []
    for sembol, df in veriler.items():
        try:
            if CFG["min_hacim"]:
                ort_tl = (df["Close"] * df["Volume"]).tail(20).mean()
                if ort_tl < CFG["min_hacim"]:
                    continue
            d = sinyalleri_hesapla(df, endeks_getiri).dropna(subset=["vwap"])
            if d.empty:
                continue
            satir = {"sembol": sembol.replace(".IS", ""), "tarih": d.index[-1].date().isoformat()}
            satir.update(durum_makinesi(d))
            satirlar.append(satir)
        except Exception as hata:
            print(f"  ! {sembol}: {hata}")

    tablo = pd.DataFrame(satirlar)
    if tablo.empty:
        return tablo
    sira = {"AL": 0, "SAT": 1, "TUT": 2, "BEKLE": 3}
    return tablo.sort_values(["durum", "kz_yuzde"], key=lambda c: c.map(sira) if c.name == "durum" else c,
                             ascending=[True, False]).reset_index(drop=True)


def ozet_yaz(tablo: pd.DataFrame) -> str:
    bugun = dt.date.today().isoformat()
    al = tablo[tablo["durum"] == "AL"]
    sat = tablo[tablo["durum"] == "SAT"]
    tut = tablo[tablo["durum"] == "TUT"]

    parcalar = [f"*BIST T/P/N taramasi — {bugun}*", ""]
    parcalar.append(f"AL: {len(al)}  |  SAT: {len(sat)}  |  TUT: {len(tut)}  |  toplam: {len(tablo)}")
    if not al.empty:
        parcalar += ["", "*AL sinyali*", ", ".join(al["sembol"].tolist())]
    if not sat.empty:
        parcalar += ["", "*SAT sinyali*", ", ".join(sat["sembol"].tolist())]
    if not tut.empty:
        satirlar = [f"{r.sembol} ({r.bar} bar, %{r.kz_yuzde})" for r in tut.itertuples()]
        parcalar += ["", "*Pozisyonda*", ", ".join(satirlar)]
    return "\n".join(parcalar)


def telegram_gonder(mesaj: str) -> None:
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    import urllib.request, urllib.parse, json
    veri = urllib.parse.urlencode(
        {"chat_id": chat, "text": mesaj[:4000], "parse_mode": "Markdown"}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", veri, timeout=20)
        print("Telegram mesaji gonderildi")
    except Exception as hata:
        print(f"Telegram hatasi: {hata}")


if __name__ == "__main__":
    tablo = tara()
    if tablo.empty:
        print("Sonuc uretilemedi.")
        sys.exit(1)

    tablo.to_csv(os.path.join(DIZIN, "sonuclar.csv"), index=False)
    ozet = ozet_yaz(tablo)
    with open(os.path.join(DIZIN, "ozet.md"), "w", encoding="utf-8") as f:
        f.write(ozet)

    print("\n" + ozet)
    telegram_gonder(ozet)
