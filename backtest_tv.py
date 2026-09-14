"""
BIST T / P / N tarayicisi — YENI (TradingView uyumlu) motor icin
geriye donuk olcum (backtest), v2.

tarayici.py icin yazilan backtest.py ile ayni mantigi kullanir, farkli
olan tek sey: sinyaller artik tarayici_tv.py'deki f_motor() uyumlu
motordan geliyor (p-edge, EMA trend filtresi, ADX filtresi, cooldown
destekli). Cikis (SAT/N/kural) mantigi iki motorde de aynidir; bu
script de cikis tarafina dokunmaz.

Eklenen senaryolar (mevcut 5 senaryoya ek olarak):
  - p-edge KAPALI  -> eski motorun P mantigina esdeger, dogrudan
    karsilastirma icin
  - Trend filtresi ACIK (EMA200 uzerinde olma sarti)
  - ADX filtresi ACIK (trend gucu sarti)
  - Cooldown ACIK (SAT sonrasi bekleme suresi)
  - Tum yeni filtreler birlikte ACIK

Gercekci varsayimlar (backtest.py ile ayni):
  - Sinyal gun sonunda olusur, islem ERTESI GUN ACILISTA yapilir.
  - Her alim-satim icin komisyon + BSMV dusulur.
  - Cooldown, canli durum_makinesi()'ndeki mantikla ayni sekilde
    islem cikarma asamasinda da uygulanir (backtest.py'de yoktu,
    burada eklendi).

Kullanim: python backtest_tv.py
Cikti:    backtest_tv_islemler.csv  +  backtest_tv_ozet.md
"""

from __future__ import annotations

import os
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

from tarayici_tv import CFG, sinyalleri_hesapla, sembolleri_oku, veri_indir

DIZIN = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────── Olcum ayarlari ───────────────────────────
KOMISYON = 0.0004       # tek yon islem maliyeti (%0.04). Kendi oranini yaz.
GUN = 800               # kac gunluk gecmis uzerinde olculecek
ERTESI_ACILIS = True    # True = sinyalin ertesi gunu acilistan islem (gercekci)

# Bedelsiz/sermaye artirimi filtresi: tek barda bundan sert hareket varsa
# veri duzeltmesi eksik demektir, o islem olcume alinmaz.
SERT_DUSUS = -25.0      # tek gunde % kac dususu supheli sayalim
SERT_YUKSELIS = 45.0    # tek gunde % kac yukselisi supheli sayalim
ISLEM_GUNU = 252        # yillandirma icin

# Asagidaki senaryolar listesindeki index (0'dan baslar). Yil-yil kirilim ve
# en iyi/en kotu 10 islem tablosu BU senaryo icin uretilir. Baska bir
# senaryoyu detayli incelemek istersen sadece bu sayiyi degistir.
# 0 = "1. TV varsayilan", 10 = "11. ADX ACIK + Sadece N ile cikis" vb.
ANA_SENARYO_INDEX = 10


def islemleri_cikar(d: pd.DataFrame, sembol: str) -> list[dict]:
    """
    Sinyal serisinden tamamlanmis islem listesi uretir.

    tarayici_tv'nin canli durum_makinesi()'ndeki cooldown mantigi burada
    da uygulanir: CFG["cooldown_on"] acikken, bir SAT'tan sonra
    CFG["cooldown_bars"] kadar bar gecmeden yeni AL kabul edilmez.
    cooldown_on=False iken davranis backtest.py ile birebir aynidir.
    """
    girisler = d["giris"].to_numpy()
    cikislar = d["cikis"].to_numpy()
    kapanis = d["Close"].to_numpy()
    acilis = d["Open"].to_numpy() if "Open" in d else kapanis

    # Sinyal i. barda olustu -> islem i+1. barin acilisinda
    fiyat = np.append(acilis[1:], np.nan) if ERTESI_ACILIS else kapanis

    # Supheli bar tespiti (bedelsiz, rucu, veri hatasi)
    gunluk = np.append(np.nan, np.diff(kapanis) / kapanis[:-1] * 100)
    supheli = (gunluk < SERT_DUSUS) | (gunluk > SERT_YUKSELIS)

    cooldown_on = CFG["cooldown_on"]
    cooldown_bars = CFG["cooldown_bars"]

    islemler: list[dict] = []
    poz = False
    g_fiyat = g_index = None
    son_cikis_index = None

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
                    "yil": d.index[i + 1].year,
                    "giris_fiyat": round(float(g_fiyat), 2),
                    "cikis_fiyat": round(float(c_fiyat), 2),
                    "gun": i - g_index,
                    "brut_yuzde": round(float(brut), 2),
                    "net_yuzde": round(float(net), 2),
                    "supheli": bool(supheli[g_index + 1:i + 2].any()),
                })
            poz = False
            son_cikis_index = i
        elif (not poz) and girisler[i]:
            giris_izin = (
                not cooldown_on
                or son_cikis_index is None
                or (i - son_cikis_index) >= cooldown_bars
            )
            if giris_izin and not np.isnan(fiyat[i]):
                poz, g_fiyat, g_index = True, fiyat[i], i
    return islemler


def _gunluk_getiri(net: pd.Series, gunler: pd.Series) -> float:
    """Sermaye surekli yatirimdaymis gibi, islem gunu basina bilesik getiri."""
    toplam_gun = max(int(gunler.sum()), 1)
    carpan = float(np.prod(1 + net / 100))
    if carpan <= 0:
        return float("nan")
    return (carpan ** (1 / toplam_gun) - 1) * 100


def istatistik(islemler: pd.DataFrame) -> dict:
    if islemler.empty:
        return {}
    temiz = islemler[~islemler["supheli"]]
    if temiz.empty:
        return {}
    net, gunler = temiz["net_yuzde"], temiz["gun"]
    kar, zarar = net[net > 0], net[net <= 0]

    # Dayaniklilik: en iyi 10 islem cikarilinca
    kalan = temiz.drop(temiz.nlargest(10, "net_yuzde").index)
    g = _gunluk_getiri(net, gunler)

    return {
        "islem_sayisi": len(net),
        "elenen": int(islemler["supheli"].sum()),
        "isabet_yuzde": round(len(kar) / len(net) * 100, 1),
        "ort_net_yuzde": round(net.mean(), 2),
        "medyan_net_yuzde": round(net.median(), 2),
        "ort_kar": round(kar.mean(), 2) if len(kar) else 0.0,
        "ort_zarar": round(zarar.mean(), 2) if len(zarar) else 0.0,
        "kar_zarar_orani": round(abs(kar.mean() / zarar.mean()), 2) if len(zarar) and len(kar) else None,
        "en_kotu": round(net.min(), 2),
        "ort_gun": round(gunler.mean(), 1),
        "gunluk_yuzde": round(g, 3),
        "yillik_yuzde": round(((1 + g / 100) ** ISLEM_GUNU - 1) * 100, 1),
        "top10_haric_ort": round(kalan["net_yuzde"].mean(), 2) if len(kalan) else None,
        "top10_haric_gunluk": round(_gunluk_getiri(kalan["net_yuzde"], kalan["gun"]), 3) if len(kalan) else None,
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


def endeks_yillik(endeks: pd.DataFrame) -> pd.DataFrame:
    """XU100'un yil yil getirisi ve gunluk getirisi."""
    k = endeks["Close"]
    satir = []
    for yil, kesit in k.groupby(k.index.year):
        if len(kesit) < 2:
            continue
        getiri = (kesit.iloc[-1] / kesit.iloc[0] - 1) * 100
        gunluk = ((kesit.iloc[-1] / kesit.iloc[0]) ** (1 / len(kesit)) - 1) * 100
        satir.append({"yil": int(yil), "xu100_yuzde": round(float(getiri), 1),
                      "xu100_gunluk": round(float(gunluk), 3)})
    return pd.DataFrame(satir)


def main() -> None:
    semboller = sembolleri_oku()
    print(f"{len(semboller)} sembol, {GUN} gunluk gecmis olculecek")

    endeks = yf.download(CFG["endeks"], period="5y", auto_adjust=True, progress=False)
    if isinstance(endeks.columns, pd.MultiIndex):
        endeks.columns = endeks.columns.droplevel(1)
    endeks_getiri = endeks["Close"].pct_change() * 100

    veriler = veri_indir(semboller, GUN)
    print(f"{len(veriler)} sembol icin veri alindi\n")

    # ── Senaryolar: mevcut TV motorunun varsayilanlari + yeni filtrelerin
    #    tek tek ve birlikte acilmasi ──
    senaryolar = [
        ("1. TV varsayilan (p-edge ACIK, digerleri KAPALI)", {}),
        ("2. p-edge KAPALI (eski motor P mantigi)", {"p_edge_only": False}),
        ("3. Trend filtresi ACIK (EMA200 uzeri)", {"trend_on": True}),
        ("4. ADX filtresi ACIK (>20)", {"adx_filtre_on": True}),
        ("5. Cooldown ACIK (3 bar bekleme)", {"cooldown_on": True}),
        ("6. Tum yeni filtreler ACIK (trend+ADX+cooldown)",
         {"trend_on": True, "adx_filtre_on": True, "cooldown_on": True}),
        ("7. Sadece T sinyali (kesisim)", {"giris_modu": "Sadece T"}),
        ("8. Kural 2 kapali (endeks sarti yok)", {"zayif_len": 99}),
        ("9. Sadece N ile cikis", {"cikis_modu": "Sadece N"}),
        ("10. Daha yuksek hacim esigi (1.8)", {"rvol_min": 1.8}),
        ("11. ADX ACIK + Sadece N ile cikis",
         {"adx_filtre_on": True, "cikis_modu": "Sadece N"}),
    ]

    sonuclar = {}
    for ad, ayar in senaryolar:
        ad, islemler = senaryo_calistir(veriler, endeks_getiri, ad, ayar)
        sonuclar[ad] = (istatistik(islemler), islemler)
        ist = sonuclar[ad][0]
        print(f"{ad}: {ist.get('islem_sayisi', 0)} islem, "
              f"gunluk %{ist.get('gunluk_yuzde', 0)}, yillik %{ist.get('yillik_yuzde', 0)}")

    ana_ad = senaryolar[ANA_SENARYO_INDEX][0]
    ana_ist, ana_islem = sonuclar[ana_ad]
    ana_temiz = ana_islem[~ana_islem["supheli"]] if not ana_islem.empty else ana_islem

    # Donem ve XU100 karsilastirmasi
    if not ana_temiz.empty:
        ilk, son = min(ana_temiz["giris_tarih"]), max(ana_temiz["cikis_tarih"])
        kesit = endeks.loc[str(ilk):str(son), "Close"]
        xu_toplam = round(float((kesit.iloc[-1] / kesit.iloc[0] - 1) * 100), 1)
        xu_gunluk = round(float(((kesit.iloc[-1] / kesit.iloc[0]) ** (1 / len(kesit)) - 1) * 100), 3)
        xu_yillik = round(((1 + xu_gunluk / 100) ** ISLEM_GUNU - 1) * 100, 1)
    else:
        ilk = son = "—"
        xu_toplam = xu_gunluk = xu_yillik = float("nan")

    # ── Rapor ──
    s = ["# Backtest sonuclari — TradingView uyumlu motor (v2)", "",
         f"Olcum tarihi: {dt.date.today().isoformat()}  ",
         f"Donem: {ilk} — {son}  |  Sembol: {len(veriler)}  ",
         f"Maliyet: gidis-donus %{KOMISYON * 200:.2f}  |  Uygulama: ertesi gun acilis  ",
         f"Cikis (SAT/N/kural) mantigi eski motorla birebir aynidir; sadece giris (AL) tarafi test ediliyor.",
         f"Supheli (bedelsiz/veri hatasi) islemler olcum disi birakildi.",
         f"Yil-yil kirilim ve en iyi/en kotu islem tablolari **'{ana_ad}'** senaryosu icindir.", "",
         f"**XU100 ayni donemde: toplam %{xu_toplam} — gunluk %{xu_gunluk} — yillik %{xu_yillik}**", "",
         "## Senaryolar", "",
         "Gunluk ve yillik sutunlari, farkli tutma sureli senaryolari adil karsilastirmak icindir.",
         "Sermayenin surekli yatirimda oldugu varsayilir (gercekte sinyal beklerken para bosta kalir).", "",
         "| Senaryo | Islem | Isabet % | Ort. net % | **Medyan %** | Ort. gun | **Gunluk %** | **Yillik %** | Top10 haric gunluk % | En kotu % |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    for ad, (ist, _) in sonuclar.items():
        if not ist:
            s.append(f"| {ad} | 0 | — | — | — | — | — | — | — | — |")
            continue
        s.append(f"| {ad} | {ist['islem_sayisi']} | {ist['isabet_yuzde']} | {ist['ort_net_yuzde']} | "
                 f"{ist['medyan_net_yuzde']} | {ist['ort_gun']} | {ist['gunluk_yuzde']} | "
                 f"{ist['yillik_yuzde']} | {ist['top10_haric_gunluk']} | {ist['en_kotu']} |")

    s += ["", f"## Yil yil — {ana_ad} vs XU100", ""]
    if not ana_temiz.empty:
        yillik = []
        xu_y = endeks_yillik(endeks).set_index("yil")
        for yil, grup in ana_temiz.groupby("yil"):
            g = _gunluk_getiri(grup["net_yuzde"], grup["gun"])
            yillik.append({
                "yil": int(yil),
                "islem": len(grup),
                "ort_net_%": round(grup["net_yuzde"].mean(), 2),
                "medyan_%": round(grup["net_yuzde"].median(), 2),
                "isabet_%": round((grup["net_yuzde"] > 0).mean() * 100, 1),
                "sistem_gunluk_%": round(g, 3),
                "xu100_gunluk_%": xu_y["xu100_gunluk"].get(int(yil), float("nan")),
            })
        yil_tablo = pd.DataFrame(yillik)
        s += [yil_tablo.to_markdown(index=False), "",
              "Son iki sutun ayni birimde: sistemin gunluk getirisi XU100'un altindaysa",
              "o yil endeksi almak daha iyiydi demektir.", ""]

    s += ["## Nasil okunur", "",
          "- **Medyan %**: islemlerin tam ortasindaki sonuc. Ortalamadan cok dusukse,",
          "  sonucu birkac buyuk kazanan tasiyor ve tekrarlanmasi sanstan cok sey bekler.",
          "- **Top10 haric gunluk %**: en iyi 10 islem silinince ne kaliyor. Buyuk dususe ugruyorsa sistem kirilgan.",
          "- **Gunluk/yillik %**: sermaye hep yatirimda varsayimiyla. Gercek getiri bunun altinda kalir.",
          "- Senaryo 2 (p-edge KAPALI) ile Senaryo 1'i karsilastirmak, yeni motorun eski motora gore",
          "  islem sikligini/performansini nasil degistirdigini gosterir.",
          "- TL mevduat/para piyasasi getirisini de rakip olarak dusun: sistem onu da gecmeli.", ""]

    if not ana_temiz.empty:
        en_iyi = ana_temiz.nlargest(10, "net_yuzde")[["sembol", "giris_tarih", "cikis_tarih", "gun", "net_yuzde"]]
        en_kotu = ana_temiz.nsmallest(10, "net_yuzde")[["sembol", "giris_tarih", "cikis_tarih", "gun", "net_yuzde"]]
        s += [f"## En iyi 10 islem ({ana_ad})", "", en_iyi.to_markdown(index=False), "",
              f"## En kotu 10 islem ({ana_ad})", "", en_kotu.to_markdown(index=False), ""]
        elenen = ana_islem[ana_islem["supheli"]]
        if not elenen.empty:
            s += [f"## Olcum disi birakilan {len(elenen)} islem (supheli fiyat hareketi)", "",
                  elenen.nsmallest(10, "net_yuzde")[["sembol", "giris_tarih", "cikis_tarih", "net_yuzde"]]
                  .to_markdown(index=False), "",
                  "Bu islemlerdeki sert hareket bedelsiz/rucu kaynakli olabilir; veri duzeltmesi eksikse",
                  "gercek kayip/kazanc bu degil. Suphelenirsen ilgili hisseyi TradingView'de kontrol et.", ""]
        ana_islem.to_csv(os.path.join(DIZIN, "backtest_tv_islemler.csv"), index=False)

    rapor = "\n".join(s)
    with open(os.path.join(DIZIN, "backtest_tv_ozet.md"), "w", encoding="utf-8") as f:
        f.write(rapor)
    print("\n" + rapor)


if __name__ == "__main__":
    main()
