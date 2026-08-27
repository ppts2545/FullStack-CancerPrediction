"""Ad-hoc lake inspector (scratch)."""
import sys

sys.path.insert(0, "/opt/airflow/src")
from pyspark.sql import functions as F  # noqa: E402

from common.config import bronze_path, gold_path, silver_path  # noqa: E402
from common.spark_session import get_spark  # noqa: E402

what = sys.argv[1] if len(sys.argv) > 1 else "join"
s = get_spark("probe")
s.sparkContext.setLogLevel("ERROR")

if what == "join":
    cl = s.read.format("delta").load(silver_path("clinical"))
    mu = s.read.format("delta").load(silver_path("mutations"))
    rs = s.read.format("delta").load(silver_path("rna_seq"))
    ck = set(r[0] for r in cl.select("patient_key").distinct().collect())
    mk = set(r[0] for r in mu.select("patient_key").distinct().collect())
    rk = set(r[0] for r in rs.select("patient_key").distinct().collect())
    print(f"clinical rows={cl.count()} patients={len(ck)}")
    print(f"mutations rows={mu.count()} patients={len(mk)}")
    print(f"rna_seq  rows={rs.count()} patients={len(rk)}")
    print(f"clinical & mutations overlap: {len(ck & mk)}")
    print(f"clinical & rna_seq overlap:   {len(ck & rk)}")
    print(f"all three overlap:            {len(ck & mk & rk)}")
    print("clinical cols:", cl.columns)
elif what.startswith(("bronze:", "silver:", "gold:")):
    layer, dom = what.split(":", 1)
    fn = {"bronze": bronze_path, "silver": silver_path, "gold": gold_path}[layer]
    df = s.read.format("delta").load(fn(dom))
    print("rows:", df.count(), "cols:", len(df.columns))
    print(df.columns)
    df.show(5, truncate=25)

s.stop()
