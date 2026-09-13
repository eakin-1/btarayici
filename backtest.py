"""
BIST T / P / N tarayicisi — geriye donuk olcum (backtest).

Ayni kurallari gecmise uygular, her islemi kaydeder ve sistemin gercekten
ise yarayip yaramadigini olcer. Farkli ayarlari (sadece T, kural 2 kapali...)
yan yana karsilastirir.

Gercekci varsayimlar:
  - Sinyal gun sonunda olusur, islem ERTESI GUN ACILISTA yapilir.
  - Her alim-satim icin komisyon + BSMV dusulur.

Kullanim: python backtest.py
Cikti:    backtest_islemler.csv  +  backtest_ozet.md
"""

from __future__ import annotations

import os
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

from tarayici import CFG, sinyalleri_hesapla, sembolleri_oku, veri_indir

DIZIN = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────── Olcum ayarlari ───────────────────────────
KOMISYON = 0.0004      # tek yon islem maliyeti (%0.04). Kendi oranini yaz.
GUN = 800              # kac gunluk gecmis uzerinde olculecek
ERTESI_ACILIS = True   # True = sinyalin ertesi gunu acilistan islem (gercekci)


def islemleri_cikar(d: pd.DataFrame, sembol: str) -> list[dict]:
    """Sinyal serisinden tamamlanmis islem listesi uretir."""
    girisler = d["giris"].to_numpy()
    cikislar = d["cikis"].to_numpy()
    kapanis = d["Close"].to_numpy()
    acilis = d["Open"].to_numpy() if "Open" in d else kapanis

    # Sinyal i. barda olustu -> islem i+1. barin acilisinda
    if ERTESI_ACILIS:
        fiyat = np.append(acilis[1:], np.nan)
    else:
        fiyat = kapanis

    islemler: list[dict] = []
    poz = False
    g_fiyat = g_index = None

    for i in range(len(d) - 1):
        if poz and cikislar[i]:
            c_fiyat = fiyat[i]
            if not np.isnan(c_fiyat) and g_fiyat:
                brut = (c_fiyat - g_fiyat) / g_fiyat * 100
                net = brut - KOMISYON * 2 * 100
                islemler.append({
                    "sembol": sembol,
                    "giris_tarih": d.index[g_index + 1].date().isoformat(),
                    "cikis_tarih": d.index[i + 1].date().isoformat(),
                    "giris_fiyat": round(float(g_fiyat), 2),
                    "cikis_fiyat": round(float(c_fiyat), 2),
                    "gun": i - g_index,
                    "brut_yuzde": round(float(brut), 2),
                    "net_yuzde": round(float(net), 2),
                })
            poz = False
        elif (not poz) and girisler[i]:
            if not np.isnan(fiyat[i]):
                poz, g_fiyat, g_index = True, fiyat[i], i
    return islemler


def istatistik(islemler: pd.DataFrame) -> dict:
    if islemler.empty:
        return {}
    net = islemler["net_yuzde"]
    kar = net[net > 0]
    zarar = net[net <= 0]
    # Bilesik getiri: her islemde sermayenin tamami kullanilmis gibi
    bilesik = (np.prod(1 + net / 100) - 1) * 100
    return {
        "islem_sayisi": len(net),
        "isabet_yuzde": round(len(kar) / len(net) * 100, 1),
        "ort_net_yuzde": round(net.mean(), 2),
        "medyan_net_yuzde": round(net.median(), 2),
        "ort_kar": round(kar.mean(), 2) if len(kar) else 0.0,
        "ort_zarar": round(zarar.mean(), 2) if len(zarar) else 0.0,
        "kar_zarar_orani": round(abs(kar.mean() / zarar.mean()), 2) if len(zarar) and len(kar) else None,
        "en_kotu": round(net.min(), 2),
        "en_iyi": round(net.max(), 2),
        "ort_gun": round(islemler["gun"].mean(), 1),
        "toplam_bilesik_yuzde": round(bilesik, 1),
        "beklenen_deger": round(net.mean(), 2),
    }


def senaryo_calistir(veriler: dict, endeks_getiri: pd.Series, ad: str, ayarlar: dict) -> tuple[str, pd.DataFrame]:
    """CFG'yi gecici degistirip tum sembolleri tarar."""
    eski = {k: CFG[k] for k in ayarlar}
    CFG.update(ayarlar)
    try:
        hepsi = []
        for sembol, df in veriler.items():
            try:
                d = sinyalleri_hesapla(df, endeks_getiri).dropna(subset=["vwap"])
                if len(d) > 30:
                    hepsi += islemleri_cikar(d, sembol.replace(".IS", ""))
            except Exception:
                pass
    finally:
        CFG.update(eski)
    return ad, pd.DataFrame(hepsi)


def endeks_getirisi(endeks: pd.DataFrame, ilk_tarih, son_tarih) -> float:
    kesit = endeks.loc[str(ilk_tarih):str(son_tarih), "Close"]
    if len(kesit) < 2:
        return float("nan")
    return round(float((kesit.iloc[-1] / kesit.iloc[0] - 1) * 100), 1)


def main() -> None:
    semboller = sembolleri_oku()
    print(f"{len(semboller)} sembol, {GUN} gunluk gecmis olculecek")

    endeks = yf.download(CFG["endeks"], period="5y", auto_adjust=True, progress=False)
    if isinstance(endeks.columns, pd.MultiIndex):
        endeks.columns = endeks.columns.droplevel(1)
    endeks_getiri = endeks["Close"].pct_change() * 100

    veriler = veri_indir(semboller, GUN)
    print(f"{len(veriler)} sembol icin veri alindi\n")

    senaryolar = [
        ("1. Mevcut ayarlar (T veya P)", {}),
        ("2. Sadece T sinyali (kesisim)", {"giris_modu": "Sadece T"}),
        ("3. Kural 2 kapali (endeks sarti yok)", {"zayif_len": 99}),
        ("4. Sadece N ile cikis", {"cikis_modu": "Sadece N"}),
        ("5. Daha yuksek hacim esigi (1.8)", {"rvol_min": 1.8}),
    ]

    sonuclar = {}
    for ad, ayar in senaryolar:
        ad, islemler = senaryo_calistir(veriler, endeks_getiri, ad, ayar)
        sonuclar[ad] = (istatistik(islemler), islemler)
        ist = sonuclar[ad][0]
        print(f"{ad}: {ist.get('islem_sayisi', 0)} islem, "
              f"isabet %{ist.get('isabet_yuzde', 0)}, "
              f"islem basi net %{ist.get('ort_net_yuzde', 0)}")

    # Karsilastirma icin al-tut (XU100) getirisi
    ana = sonuclar[senaryolar[0][0]][1]
    if not ana.empty:
        ilk = min(ana["giris_tarih"])
        son = max(ana["cikis_tarih"])
        xu = endeks_getirisi(endeks, ilk, son)
    else:
        ilk = son = "—"
        xu = float("nan")

    # ── Rapor ──
    sat = ["# Backtest sonuclari", "",
           f"Olcum tarihi: {dt.date.today().isoformat()}  ",
           f"Donem: {ilk} — {son}  ",
           f"Sembol sayisi: {len(veriler)}  ",
           f"Islem maliyeti: tek yon %{KOMISYON * 100:.2f} (gidis-donus %{KOMISYON * 200:.2f})  ",
           f"Uygulama: {'sinyalin ertesi gunu acilistan' if ERTESI_ACILIS else 'ayni gun kapanistan'}  ",
           f"**Karsilastirma — ayni donemde XU100: %{xu}**", "",
           "## Senaryolar", "",
           "| Senaryo | Islem | Isabet % | Islem basi net % | Ort. kar % | Ort. zarar % | K/Z orani | Ort. gun | En kotu % |",
           "|---|---|---|---|---|---|---|---|---|"]

    for ad, (ist, _) in sonuclar.items():
        if not ist:
            sat.append(f"| {ad} | 0 | — | — | — | — | — | — | — |")
            continue
        sat.append(f"| {ad} | {ist['islem_sayisi']} | {ist['isabet_yuzde']} | {ist['ort_net_yuzde']} | "
                   f"{ist['ort_kar']} | {ist['ort_zarar']} | {ist['kar_zarar_orani']} | "
                   f"{ist['ort_gun']} | {ist['en_kotu']} |")

    sat += ["", "## Nasil okunur", "",
            "- **Islem basi net %**: komisyon dusulmus ortalama sonuc. Sifirin altindaysa sistem para kaybettiriyor.",
            "- **Isabet %** tek basina yaniltir. Dusuk isabet + yuksek K/Z orani da kazandirir.",
            "- **K/Z orani**: ortalama karin ortalama zarara bolumu. 1'in altindaysa isabet oraninin yuksek olmasi sart.",
            "- **En kotu %**: tek bir islemde gorulen en buyuk kayip. Zarar kes olmadigi icin bu deger onemli.",
            "- XU100 satiri referans: sistem bundan iyi degilse ugrasmaya degmiyor demektir.", ""]

    ana_ist, ana_islem = sonuclar[senaryolar[0][0]]
    if not ana_islem.empty:
        en_iyi = ana_islem.nlargest(10, "net_yuzde")[["sembol", "giris_tarih", "cikis_tarih", "gun", "net_yuzde"]]
        en_kotu = ana_islem.nsmallest(10, "net_yuzde")[["sembol", "giris_tarih", "cikis_tarih", "gun", "net_yuzde"]]
        sat += ["## Mevcut ayarlarda en iyi 10 islem", "", en_iyi.to_markdown(index=False), "",
                "## En kotu 10 islem", "", en_kotu.to_markdown(index=False), ""]
        ana_islem.to_csv(os.path.join(DIZIN, "backtest_islemler.csv"), index=False)

    rapor = "\n".join(sat)
    with open(os.path.join(DIZIN, "backtest_ozet.md"), "w", encoding="utf-8") as f:
        f.write(rapor)
    print("\n" + rapor)


if __name__ == "__main__":
    main()
