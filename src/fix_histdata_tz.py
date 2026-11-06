"""Исправление времени HistData: файлы записаны как (время_файла + 5 ч) = «UTC при EST круглый год»,
а время в файлах HistData — нью-йоркское С переходом на летнее. Восстанавливаем локальное время NY и переводим в UTC.
Неоднозначный час осенью (01:00–02:00 дважды) и несуществующий весенний — выбрасываются (ночь, не влияет на окна).
Запуск: PYTHONPATH=src.venv/bin/python src/fix_histdata_tz.py data/raw/NQH_M1.csv..."""
import sys
import pandas as pd

for path in sys.argv[1:]:
    df = pd.read_csv(path, header=None)
    local = pd.to_datetime(df[0] - 5 * 3600 * 1000, unit="ms")
    ny = local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    ok = ny.notna()
    df = df[ok.to_numpy()].copy()
    utc = ny[ok].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ms]")
    df[0] = utc.astype("int64").to_numpy()                 # миллисекунды (pandas 3 хранит не в нс — без деления)
    n0 = len(df)
    df = df.drop_duplicates(0).sort_values(0)
    assert len(df) > 0.99 * n0, f"слишком много дублей: {n0} → {len(df)}"
    df.to_csv(path, header=False, index=False)
    print(path, "строк", len(df), "выброшено", int((~ok).sum()))
