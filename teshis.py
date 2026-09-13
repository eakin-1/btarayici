"""
Giris mi kotu, cikis mi? — teshis scripti.

Uc ayri olcum yapar:

  A) GIRIS KALITESI
     Sinyalden sonraki 1/3/5/10/20 gunluk getiri, ayni gunun piyasa ortalamasi
     cikarilarak. Boylece "o donemde her sey yukseliyordu" etkisi elenir.
     Artiysa girisin bir degeri var, sifir civariysa yok.

  B) CIKIS KALITESI
     Her islemde fiyat tutma suresi icinde en fazla ne kadar yukseldi (MFE),
     en fazla ne kadar dustu (MAE), cikista zirvenin yuzde kaci cebe girdi.
     Ayrica cikistan SONRAKI 5/10 gunde ne oldu: yukselmeye devam ettiyse
     erken cikilmis, dustuyse cikis dogru zamanlanmis demektir.

  C) KONTROL TESTI
     Ayni girisler, kurallar yerine SABIT SURE sonra cikis (5/10/20/40 gun).
     Sabit sure kurallardan iyiyse sorun cikista, kotuyse sorun giriste.

Kullanim: python teshis.py
Cikti:    teshis_ozet.md
"""

from __future__ import annotations

import os
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

from tarayici import CFG, sinyalleri_hesapla, sembolleri_oku, veri_indir

DIZIN = os.path.dirname(os.path.abspath(__file__))

GUN = 800
KOMISYON = 0.0004
UFUKLAR = [1, 3, 5, 10, 20]        # giris testi icin ileri bakis pencereleri
SABIT_SURELER = [5, 10, 20, 40]    # kontrol testi icin
SERT_DUSUS, SERT_YUKSELIS = -25.0, 45.0
ISLEM_GUNU = 252


def supheli_barlar(kapanis: np.ndarray) -> np.ndarray:
    """Bedelsiz/veri hatasi kaynakli sert hareketler."""
    g = np.append(np.nan, np.diff(kapanis) / kapanis[:-1] * 100)
    return (g < SERT_DUSUS) | (g > SERT_YUKSELIS)


def islem_indeksleri(d: pd.DataFrame) -> list[tuple[int, int]]:
    """Kurallara gore (giris_bar, cikis_bar) ciftleri."""
    girisler, cikislar = d["giris"].to_numpy(), d["cikis"].to_numpy()
    ciftler, poz, gi = [], False, None
    for i in range(len(d)):
        if poz and cikislar[i]:
            ciftler.append((gi, i))
            poz = False
        elif (not poz) and girisler[i]:
            poz, gi = True, i
    return ciftler


def giris_indeksleri(d: pd.DataFrame) -> list[int]:
    """Pozisyon durumundan bagimsiz, tum giris sinyali barlari."""
    return list(np.flatnonzero(d["giris"].to_numpy()))


# ─────────────────────────── A) Giris kalitesi ───────────────────────────
def giris_testi(sinyaller: dict, kapanis_tablo: pd.DataFrame) -> pd.DataFrame:
    ileri, piyasa = {}, {}
    for h in UFUKLAR:
        ileri[h] = (kapanis_tablo.shift(-h) / kapanis_tablo - 1) * 100
        piyasa[h] = ileri[h].mean(axis=1)          # o gunun piyasa ortalamasi

    satirlar = []
    for h in UFUKLAR:
        ham, fazla = [], []
        for sembol, d in sinyaller.items():
            if sembol not in ileri[h].columns:
                continue
            tarihler = d.index[giris_indeksleri(d)]
            tarihler = tarihler.intersection(ileri[h].index)
            if len(tarihler) == 0:
                continue
            g = ileri[h].loc[tarihler, sembol]
            p = piyasa[h].loc[tarihler]
            gecerli = g.notna() & p.notna()
            ham += list(g[gecerli])
            fazla += list((g - p)[gecerli])
        if not ham:
            continue
        ham_s, fazla_s = pd.Series(ham), pd.Series(fazla)
        satirlar.append({
            "ufuk_gun": h,
            "sinyal": len(ham_s),
            "ort_getiri_%": round(ham_s.mean(), 2),
            "piyasa_ustu_ort_%": round(fazla_s.mean(), 2),
            "piyasa_ustu_medyan_%": round(fazla_s.median(), 2),
            "piyasayi_gecme_%": round((fazla_s > 0).mean() * 100, 1),
        })
    return pd.DataFrame(satirlar)


# ─────────────────────────── B) Cikis kalitesi ───────────────────────────
def cikis_testi(sinyaller: dict) -> tuple[pd.DataFrame, dict]:
    kayitlar = []
    for sembol, d in sinyaller.items():
        kapanis = d["Close"].to_numpy()
        yuksek = d["High"].to_numpy() if "High" in d else kapanis
        dusuk = d["Low"].to_numpy() if "Low" in d else kapanis
        sup = supheli_barlar(kapanis)

        for gi, ci in islem_indeksleri(d):
            if sup[gi:ci + 1].any():
                continue
            g_fiyat = kapanis[gi]
            if not np.isfinite(g_fiyat) or g_fiyat <= 0:
                continue
            pencere = slice(gi + 1, ci + 1)
            if ci <= gi:
                continue
            mfe = (np.nanmax(yuksek[pencere]) - g_fiyat) / g_fiyat * 100   # zirve
            mae = (np.nanmin(dusuk[pencere]) - g_fiyat) / g_fiyat * 100    # dip
            gerceklesen = (kapanis[ci] - g_fiyat) / g_fiyat * 100

            sonra = {}
            for h in (5, 10):
                j = ci + h
                sonra[h] = (kapanis[j] - kapanis[ci]) / kapanis[ci] * 100 if j < len(kapanis) else np.nan

            kayitlar.append({
                "sembol": sembol, "gun": ci - gi,
                "gerceklesen_%": gerceklesen, "zirve_%": mfe, "dip_%": mae,
                "cikis_sonrasi_5g_%": sonra[5], "cikis_sonrasi_10g_%": sonra[10],
            })

    t = pd.DataFrame(kayitlar)
    if t.empty:
        return t, {}

    karli = t[t["gerceklesen_%"] > 0]
    zararli = t[t["gerceklesen_%"] <= 0]
    verim = (t["gerceklesen_%"] / t["zirve_%"].replace(0, np.nan) * 100).clip(-200, 200)

    ozet = {
        "islem": len(t),
        "ort_zirve_%": round(t["zirve_%"].mean(), 2),
        "ort_dip_%": round(t["dip_%"].mean(), 2),
        "ort_gerceklesen_%": round(t["gerceklesen_%"].mean(), 2),
        "cikis_verimi_%": round(verim.median(), 1),
        "zirveden_geri_verilen_%": round((t["zirve_%"] - t["gerceklesen_%"]).mean(), 2),
        "cikis_sonrasi_5g_ort_%": round(t["cikis_sonrasi_5g_%"].mean(), 2),
        "cikis_sonrasi_10g_ort_%": round(t["cikis_sonrasi_10g_%"].mean(), 2),
        "cikis_sonrasi_yukselen_%": round((t["cikis_sonrasi_10g_%"] > 0).mean() * 100, 1),
        "karlida_ort_zirve_%": round(karli["zirve_%"].mean(), 2) if len(karli) else None,
        "zararlida_ort_zirve_%": round(zararli["zirve_%"].mean(), 2) if len(zararli) else None,
        "zararlida_ort_dip_%": round(zararli["dip_%"].mean(), 2) if len(zararli) else None,
    }
    return t, ozet


# ─────────────────────────── C) Kontrol testi ───────────────────────────
def sabit_sure_testi(sinyaller: dict) -> pd.DataFrame:
    satirlar = []
    for sure in SABIT_SURELER:
        netler, gunler = [], []
        for sembol, d in sinyaller.items():
            kapanis = d["Close"].to_numpy()
            sup = supheli_barlar(kapanis)
            girisler = d["giris"].to_numpy()
            i, n = 0, len(d)
            while i < n - 1:
                if girisler[i]:
                    j = min(i + sure, n - 1)
                    if not sup[i:j + 1].any() and kapanis[i] > 0:
                        netler.append((kapanis[j] - kapanis[i]) / kapanis[i] * 100 - KOMISYON * 200)
                        gunler.append(j - i)
                    i = j                      # pozisyon bitene kadar yeni giris yok
                else:
                    i += 1
        if not netler:
            continue
        net = pd.Series(netler)
        toplam_gun = max(sum(gunler), 1)
        carpan = float(np.prod(1 + net / 100))
        gunluk = (carpan ** (1 / toplam_gun) - 1) * 100 if carpan > 0 else float("nan")
        satirlar.append({
            "cikis": f"sabit {sure} gun",
            "islem": len(net),
            "ort_net_%": round(net.mean(), 2),
            "medyan_%": round(net.median(), 2),
            "isabet_%": round((net > 0).mean() * 100, 1),
            "gunluk_%": round(gunluk, 3),
            "yillik_%": round(((1 + gunluk / 100) ** ISLEM_GUNU - 1) * 100, 1),
        })
    return pd.DataFrame(satirlar)


# ─────────────────────────── Ana akis ───────────────────────────
def main() -> None:
    semboller = sembolleri_oku()
    print(f"{len(semboller)} sembol, {GUN} gunluk gecmis")

    endeks = yf.download(CFG["endeks"], period="5y", auto_adjust=True, progress=False)
    if isinstance(endeks.columns, pd.MultiIndex):
        endeks.columns = endeks.columns.droplevel(1)
    endeks_getiri = endeks["Close"].pct_change() * 100

    veriler = veri_indir(semboller, GUN)
    print(f"{len(veriler)} sembol icin veri alindi\n")

    sinyaller, kapanislar = {}, {}
    for sembol, df in veriler.items():
        try:
            d = sinyalleri_hesapla(df, endeks_getiri).dropna(subset=["vwap"])
            if len(d) > 40:
                kisa = sembol.replace(".IS", "")
                sinyaller[kisa] = d
                kapanislar[kisa] = d["Close"]
        except Exception:
            pass
    kapanis_tablo = pd.DataFrame(kapanislar)

    print("A) giris testi...")
    a = giris_testi(sinyaller, kapanis_tablo)
    print("B) cikis testi...")
    _, b = cikis_testi(sinyaller)
    print("C) kontrol testi...")
    c = sabit_sure_testi(sinyaller)

    xu_gunluk = float(((endeks["Close"].iloc[-1] / endeks["Close"].iloc[-GUN])
                       ** (1 / GUN) - 1) * 100) if len(endeks) > GUN else float("nan")

    s = ["# Teshis: giris mi kotu, cikis mi?", "",
         f"Olcum tarihi: {dt.date.today().isoformat()}  |  Sembol: {len(sinyaller)}  ",
         f"Karsilastirma — XU100 gunluk: %{xu_gunluk:.3f}", "",
         "## A) Giris kalitesi", "",
         "Sinyalden sonraki getiriler. 'Piyasa ustu' sutunu, ayni gun tum hisselerin",
         "ortalamasi cikarilarak hesaplandi — donem etkisi bu sayede elenir.", ""]
    s += [a.to_markdown(index=False) if not a.empty else "veri yok", "",
          "**Nasil okunur:** 'piyasa ustu ort' sifira yakin veya eksiyse, giris sinyali",
          "rastgele bir gunden daha iyi bir an secmiyor demektir. Artiysa girisin degeri var",
          "ve sorun cikista aranmali. 'Piyasayi gecme %' 50'nin ustunde olmali.", "",
          "## B) Cikis kalitesi", ""]

    if b:
        s += ["| Olcum | Deger |", "|---|---|"]
        etiket = {
            "islem": "Islem sayisi",
            "ort_zirve_%": "Tutarken gorulen ortalama ZIRVE %",
            "ort_dip_%": "Tutarken gorulen ortalama DIP %",
            "ort_gerceklesen_%": "Cikista gerceklesen ortalama %",
            "cikis_verimi_%": "Cikis verimi (zirvenin yuzde kaci cebe girdi, medyan)",
            "zirveden_geri_verilen_%": "Zirveden geri verilen ortalama puan",
            "cikis_sonrasi_5g_ort_%": "Cikistan sonraki 5 gun ortalama %",
            "cikis_sonrasi_10g_ort_%": "Cikistan sonraki 10 gun ortalama %",
            "cikis_sonrasi_yukselen_%": "Cikistan sonra 10 gunde yukselen islem %",
            "karlida_ort_zirve_%": "Karli islemlerde ortalama zirve %",
            "zararlida_ort_zirve_%": "Zararli islemlerde ortalama zirve %",
            "zararlida_ort_dip_%": "Zararli islemlerde ortalama dip %",
        }
        for k, v in b.items():
            s.append(f"| {etiket.get(k, k)} | {v} |")
        s += ["", "**Nasil okunur:** Cikistan sonraki 5/10 gun ortalamasi belirgin ARTIysa",
              "erken cikiyorsun, kurallar kazanci kesiyor. EKSIyse cikis dogru calisiyor.",
              "Zirveden geri verilen puan buyukse, kar korumasi (trailing stop) eksik.",
              "Zararli islemlerde 'ortalama dip' cok derinse zarar kes eksigi var.", ""]
    else:
        s += ["veri yok", ""]

    s += ["## C) Kontrol testi — kurallar yerine sabit sure", "",
          c.to_markdown(index=False) if not c.empty else "veri yok", "",
          "**Nasil okunur:** Sabit sureli cikislarin gunluk getirisi, mevcut kurallarin",
          "gunluk getirisinden YUKSEKse sorun cikis kurallarinda. Hepsi birbirine yakin",
          "ve hepsi XU100'un altindaysa sorun giriste — cikisi degistirmek kurtarmaz.", ""]

    rapor = "\n".join(s)
    with open(os.path.join(DIZIN, "teshis_ozet.md"), "w", encoding="utf-8") as f:
        f.write(rapor)
    print("\n" + rapor)


if __name__ == "__main__":
    main()
