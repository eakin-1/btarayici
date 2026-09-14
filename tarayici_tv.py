"""
BIST T / P / N tarayicisi — YENI TradingView uyumlu paralel surum.

Korunanlar:
- yfinance veri kaynagi
- semboller_tv.txt (yoksa semboller.txt yedegi)
- AL / TUT / SAT / BEKLE raporu
- sonuclar_tv.csv + ozet_tv.md
- opsiyonel Telegram bildirimi

Degisen kisim:
- AL giris mantigi TradingView kodundaki f_motor() ile uyumludur:
  * T sinyali
  * P sinyali ve pEdgeOnly
  * T veya P giris modu
  * opsiyonel EMA trend filtresi
  * opsiyonel ADX filtresi
  * opsiyonel cooldown
- Cikis mantigi artik varsayilan olarak "Sadece N": backtest'te "N veya kural"
  ayarinin kazanan islemleri cok erken kestigi, SAT sinyalinin bu yuzden pratikte
  hemen hic gorunmedigi tespit edildi. Kural 1/Kural 2 kodu, ileride tekrar
  denemek istersen diye SILINMEDI, sadece varsayilan cikis_modu degisti ve
  CIKIS_MODU ortam degiskeniyle GitHub Actions'tan degistirilebilir hale geldi.

Veri kaynagi: yfinance (BIST sembolleri '.IS' ekiyle, or. THYAO.IS)
Kullanim:    python tarayici_tv.py
Cikti:       sonuclar_tv.csv + ozet_tv.md (+ istege bagli Telegram mesaji)
"""

from __future__ import annotations

import os
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────── Ayarlar ───────────────────────────
CFG = {
    # TradingView / Filtreler
    "rvol_len": 10,
    "rvol_min": 1.2,
    "roll_len": 20,
    "t_yon": "yukari",              # "yukari" | "asagi"
    "giris_modu": "T veya P",       # "Sadece T" | "Sadece P" | "T veya P" | "T + P"

    # TradingView / Gelismis filtreler
    "p_edge_only": True,             # TV varsayilani: P sadece YENI olustugunda sinyal
    "trend_on": False,               # TV varsayilani: kapali
    "ema_len": 200,
    "adx_filtre_on": False,          # TV varsayilani: kapali
    "adx_len": 14,
    "adx_min": 20.0,
    "cooldown_on": False,            # TV varsayilani: kapali
    "cooldown_bars": 3,

    # Cikis mantigi — backtest sonuclarina gore degistirildi:
    # "N veya kural" kazananlari cok erken kesiyordu (bkz. backtest_tv_ozet.md,
    # Senaryo 9 "Sadece N ile cikis"). Varsayilan artik "Sadece N".
    "cikis_modu": "Sadece N",       # "Sadece N" | "Sadece kural" | "N veya kural" | "N ve kural"
    "ard_len": 3,
    "zayif_len": 2,
    "zayif_sart": True,
    "endeks": "XU100.IS",

    # Mevcut GitHub tarayici ayarlari — DEGISTIRILMEDI
    "gun": 400,
    "min_hacim": 1_000_000,         # TL bazli ortalama islem hacmi filtresi (0 = kapali)
}

DIZIN = os.path.dirname(os.path.abspath(__file__))


def _env_bool(ad: str, varsayilan: bool) -> bool:
    deger = os.getenv(ad)
    if deger is None or str(deger).strip() == "":
        return varsayilan
    return str(deger).strip().lower() in {"1", "true", "yes", "on", "evet", "acik"}


def _env_int(ad: str, varsayilan: int) -> int:
    deger = os.getenv(ad)
    if deger is None or str(deger).strip() == "":
        return varsayilan
    try:
        return int(deger)
    except ValueError:
        return varsayilan


def _env_float(ad: str, varsayilan: float) -> float:
    deger = os.getenv(ad)
    if deger is None or str(deger).strip() == "":
        return varsayilan
    try:
        return float(deger)
    except ValueError:
        return varsayilan


def _env_secim(ad: str, varsayilan: str, gecerliler: set[str]) -> str:
    """Sadece belirli seceneklere izin verilen metin bazli ayarlar icin (ornek: CIKIS_MODU)."""
    deger = os.getenv(ad)
    if deger is None or str(deger).strip() == "":
        return varsayilan
    deger = deger.strip()
    return deger if deger in gecerliler else varsayilan


def github_ayarlarini_uygula() -> None:
    """GitHub Actions'tan gelen ayarlari TradingView input'lari gibi CFG'ye uygular."""
    CFG["p_edge_only"] = _env_bool("P_EDGE_ONLY", CFG["p_edge_only"])
    CFG["trend_on"] = _env_bool("TREND_ON", CFG["trend_on"])
    CFG["ema_len"] = _env_int("EMA_LEN", CFG["ema_len"])
    CFG["adx_filtre_on"] = _env_bool("ADX_FILTRE_ON", CFG["adx_filtre_on"])
    CFG["adx_len"] = _env_int("ADX_LEN", CFG["adx_len"])
    CFG["adx_min"] = _env_float("ADX_MIN", CFG["adx_min"])
    CFG["cooldown_on"] = _env_bool("COOLDOWN_ON", CFG["cooldown_on"])
    CFG["cooldown_bars"] = _env_int("COOLDOWN_BARS", CFG["cooldown_bars"])
    CFG["cikis_modu"] = _env_secim(
        "CIKIS_MODU",
        CFG["cikis_modu"],
        {"Sadece N", "Sadece kural", "N veya kural", "N ve kural"},
    )


# GitHub Actions'ta secilen ayarlar varsa tum tarama motoruna uygulanir.
github_ayarlarini_uygula()


# ─────────────────────────── Gosterge yardimcilari ───────────────────────────
def rsi(seri: pd.Series, uzunluk: int) -> pd.Series:
    """Mevcut tarayicidaki Wilder/RMA tabanli RSI hesabi korunmustur."""
    fark = seri.diff()
    kazanc = fark.clip(lower=0)
    kayip = (-fark).clip(lower=0)
    ort_kazanc = kazanc.ewm(alpha=1 / uzunluk, adjust=False, min_periods=uzunluk).mean()
    ort_kayip = kayip.ewm(alpha=1 / uzunluk, adjust=False, min_periods=uzunluk).mean()
    rs = ort_kazanc / ort_kayip.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def adx_hesapla(df: pd.DataFrame, uzunluk: int) -> pd.Series:
    """
    Wilder ADX yaklasimi.
    Yalnizca CFG['adx_filtre_on'] = True ise AL filtresinde kullanilir.
    Varsayilan False oldugu icin mevcut davranisi etkilemez.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    yukari = high.diff()
    asagi = -low.diff()

    plus_dm = pd.Series(np.where((yukari > asagi) & (yukari > 0), yukari, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((asagi > yukari) & (asagi > 0), asagi, 0.0), index=df.index)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / uzunluk, adjust=False, min_periods=uzunluk).mean()
    plus_sm = plus_dm.ewm(alpha=1 / uzunluk, adjust=False, min_periods=uzunluk).mean()
    minus_sm = minus_dm.ewm(alpha=1 / uzunluk, adjust=False, min_periods=uzunluk).mean()

    plus_di = 100 * plus_sm / atr.replace(0, np.nan)
    minus_di = 100 * minus_sm / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / uzunluk, adjust=False, min_periods=uzunluk).mean()


def sinyalleri_hesapla(df: pd.DataFrame, endeks_getiri: pd.Series) -> pd.DataFrame:
    """
    TradingView f_motor() AL mantigi + guncellenmis cikis mantigi.

    AL tarafindaki temel fark:
      pSinRaw = close > VWAP and hacimOK and RSI5 >= RSI10
      pSin    = pEdgeOnly ? (pSinRaw and not pSinRaw[1]) : pSinRaw
    """
    d = df.copy()
    hlc3 = (d["High"] + d["Low"] + d["Close"]) / 3

    # Kayan (rolling) VWAP — TradingView varsayilani
    n = CFG["roll_len"]
    d["vwap"] = (hlc3 * d["Volume"]).rolling(n).sum() / d["Volume"].rolling(n).sum()

    # Goreceli hacim
    d["rvol"] = d["Volume"] / d["Volume"].rolling(CFG["rvol_len"]).mean()
    hacim_ok = d["rvol"] > CFG["rvol_min"]

    r3 = rsi(d["Close"], 3)
    r5 = rsi(d["Close"], 5)
    r10 = rsi(d["Close"], 10)

    # T: VWAP kesisi + hacim + RSI teyidi
    ust = (d["Close"] > d["vwap"]) & (d["Close"].shift(1) <= d["vwap"].shift(1))
    alt = (d["Close"] < d["vwap"]) & (d["Close"].shift(1) >= d["vwap"].shift(1))
    kesis = ust if CFG["t_yon"] == "yukari" else alt
    d["T"] = kesis & hacim_ok & (r5 >= r10)

    # P: TradingView pEdgeOnly mantigi
    d["P_raw"] = (d["Close"] > d["vwap"]) & hacim_ok & (r5 >= r10)
    if CFG["p_edge_only"]:
        onceki_p = d["P_raw"].shift(1).fillna(False).astype(bool)
        d["P"] = d["P_raw"] & ~onceki_p
    else:
        d["P"] = d["P_raw"]

    # N sinyali (cikis kurallarindan biri)
    d["N"] = (d["Close"] < d["vwap"]) & hacim_ok & (r3 < r5)

    # TradingView girisTemel
    mod = CFG["giris_modu"]
    if mod == "Sadece T":
        giris_temel = d["T"]
    elif mod == "Sadece P":
        giris_temel = d["P"]
    elif mod == "T veya P":
        giris_temel = d["T"] | d["P"]
    else:  # "T + P"
        giris_temel = d["T"] & d["P"]

    # Opsiyonel EMA trend filtresi — TradingView varsayilani KAPALI
    d["ema"] = d["Close"].ewm(span=CFG["ema_len"], adjust=False).mean()
    if CFG["trend_on"]:
        trend_ok = (d["Close"] > d["ema"]) if CFG["t_yon"] == "yukari" else (d["Close"] < d["ema"])
    else:
        trend_ok = pd.Series(True, index=d.index)

    # Opsiyonel ADX filtresi — TradingView varsayilani KAPALI
    if CFG["adx_filtre_on"]:
        d["adx"] = adx_hesapla(d, CFG["adx_len"])
        adx_ok = d["adx"] > CFG["adx_min"]
    else:
        d["adx"] = np.nan
        adx_ok = pd.Series(True, index=d.index)

    # TradingView girisSinyal
    d["giris"] = giris_temel & trend_ok & adx_ok

    # ── Cikis kurallari ──
    # Kural 1 — ard_len bar ust uste dusus
    dusus = (d["Close"] < d["Close"].shift(1)).astype(int)
    d["kural1"] = dusus.rolling(CFG["ard_len"]).sum() == CFG["ard_len"]

    # Kural 2 — zayif_len bar ust uste endeksten kotu performans
    getiri = d["Close"].pct_change() * 100
    e = endeks_getiri.reindex(d.index)
    zayif_bar = getiri < e
    if CFG["zayif_sart"]:
        zayif_bar &= getiri < 0
    d["kural2"] = zayif_bar.rolling(CFG["zayif_len"]).sum() == CFG["zayif_len"]

    kural = d["kural1"] | d["kural2"]
    cmod = CFG["cikis_modu"]
    d["cikis"] = (
        d["N"] if cmod == "Sadece N"
        else kural if cmod == "Sadece kural"
        else (d["N"] | kural) if cmod == "N veya kural"
        else (d["N"] & kural)
    )
    return d


def durum_makinesi(d: pd.DataFrame) -> dict:
    """
    Mevcut AL/TUT/SAT/BEKLE durum makinesi.
    Opsiyonel cooldown destegi mevcut. cooldown_on=False iken eski davranis aynen korunur.
    """
    poz = False
    durum = 0                    # 0 BEKLE, 1 AL, 2 SAT, 3 TUT
    giris_fiyat = np.nan
    giris_index = 0
    giris_tarih = None
    son_cikis_index = None

    girisler = d["giris"].fillna(False).to_numpy(dtype=bool)
    cikislar = d["cikis"].fillna(False).to_numpy(dtype=bool)
    kapanislar = d["Close"].to_numpy()

    for i in range(len(d)):
        giris_izin = (
            not CFG["cooldown_on"]
            or son_cikis_index is None
            or (i - son_cikis_index) >= CFG["cooldown_bars"]
        )

        if poz and cikislar[i]:
            poz, durum = False, 2
            son_cikis_index = i
            # Eski GitHub rapor yapisi korunuyor: SAT satirinda son giris bilgisi gosterilebilir.
        elif (not poz) and girisler[i] and giris_izin:
            poz, durum = True, 1
            giris_fiyat, giris_index = kapanislar[i], i
            giris_tarih = d.index[i]
        else:
            durum = 3 if poz else 0

    son = kapanislar[-1]
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
def sembolleri_oku(dosya: str = "semboller_tv.txt") -> list[str]:
    # Yeni tarayici kendi sembol listesini kullanir.
    # Dosya yoksa eski tarayicinin semboller.txt dosyasina salt-okunur fallback yapar.
    yol = os.path.join(DIZIN, dosya)
    if not os.path.exists(yol):
        yol = os.path.join(DIZIN, "semboller.txt")
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
            ham = yf.download(
                grup,
                start=baslangic,
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
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
        time.sleep(1)
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

            satir = {
                "sembol": sembol.replace(".IS", ""),
                "tarih": d.index[-1].date().isoformat(),
            }
            satir.update(durum_makinesi(d))
            satirlar.append(satir)
        except Exception as hata:
            print(f"  ! {sembol}: {hata}")

    tablo = pd.DataFrame(satirlar)
    if tablo.empty:
        return tablo

    sira = {"AL": 0, "SAT": 1, "TUT": 2, "BEKLE": 3}
    return tablo.sort_values(
        ["durum", "kz_yuzde"],
        key=lambda c: c.map(sira) if c.name == "durum" else c,
        ascending=[True, False],
    ).reset_index(drop=True)


def ozet_yaz(tablo: pd.DataFrame) -> str:
    bugun = dt.date.today().isoformat()
    al = tablo[tablo["durum"] == "AL"]
    sat = tablo[tablo["durum"] == "SAT"]
    tut = tablo[tablo["durum"] == "TUT"]

    parcalar = [f"*BIST T/P/N taramasi — {bugun}*", ""]
    parcalar.append(
        "Ayarlar: "
        f"P-edge={'ACIK' if CFG['p_edge_only'] else 'KAPALI'} | "
        f"Trend EMA({CFG['ema_len']})={'ACIK' if CFG['trend_on'] else 'KAPALI'} | "
        f"ADX({CFG['adx_len']})>{CFG['adx_min']:g}={'ACIK' if CFG['adx_filtre_on'] else 'KAPALI'} | "
        f"Cooldown({CFG['cooldown_bars']})={'ACIK' if CFG['cooldown_on'] else 'KAPALI'} | "
        f"Cikis={CFG['cikis_modu']}"
    )
    parcalar.append(
        f"AL: {len(al)}  |  SAT: {len(sat)}  |  TUT: {len(tut)}  |  toplam: {len(tablo)}"
    )

    if not al.empty:
        parcalar += ["", "*AL sinyali*", ", ".join(al["sembol"].tolist())]
    if not sat.empty:
        parcalar += ["", "*SAT sinyali*", ", ".join(sat["sembol"].tolist())]
    if not tut.empty:
        satirlar = [f"{r.sembol} ({r.bar} bar, %{r.kz_yuzde})" for r in tut.itertuples()]
        parcalar += ["", "*Pozisyonda*", ", ".join(satirlar)]

    return "\n".join(parcalar)


def telegram_gonder(mesaj: str) -> None:
    token = os.getenv("TELEGRAM_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return

    import urllib.parse
    import urllib.request

    veri = urllib.parse.urlencode(
        {"chat_id": chat, "text": mesaj[:4000], "parse_mode": "Markdown"}
    ).encode()

    try:
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage",
            veri,
            timeout=20,
        )
        print("Telegram mesaji gonderildi")
    except Exception as hata:
        print(f"Telegram hatasi: {hata}")


if __name__ == "__main__":
    tablo = tara()
    if tablo.empty:
        print("Sonuc uretilemedi.")
        sys.exit(1)

    tablo.to_csv(os.path.join(DIZIN, "sonuclar_tv.csv"), index=False)
    ozet = ozet_yaz(tablo)

    with open(os.path.join(DIZIN, "ozet_tv.md"), "w", encoding="utf-8") as f:
        f.write(ozet)

    print("\n" + ozet)
    telegram_gonder(ozet)
