"""Genomic data has far more features (genes) than samples (patients) - fuse
it raw and it drowns out every other modality and overfits. Reduce first.
"""
from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def top_variable_genes(long_df: DataFrame, gene_col: str, value_col: str, n: int = 2000) -> DataFrame:
    """RNA-Seq spans ~60k genes - well past Spark's pivotMaxValues guard, and
    mostly near-constant columns anyway. Keep the n highest-variance genes
    (where the expression signal PCA will compress actually lives) and drop the
    rest *before* the long->wide pivot.
    """
    ranked = (
        long_df.filter(F.col(gene_col).isNotNull())
        .groupBy(gene_col)
        .agg(F.variance(F.col(value_col).cast("double")).alias("_v"))
        .orderBy(F.col("_v").desc_nulls_last())
        .limit(n)
    )
    keep = [r[gene_col] for r in ranked.collect()]
    return long_df.filter(F.col(gene_col).isin(keep))


def reduce_expression_pca(df: DataFrame, gene_columns: list[str], k: int = 50, id_col: str = "patient_key") -> DataFrame:
    """RNA-Seq expression: one row per patient, one column per gene (wide
    format expected). Standardize before PCA since genes differ in scale by
    orders of magnitude, then keep only the top-k principal components as
    the genomic feature block fed into fusion.

    Output is exploded into flat `expression_pca_0..{k-1}` columns rather
    than a single vector column - training reads this via pandas/XGBoost,
    which can't consume a Spark ML VectorUDT cell.
    """
    assembler = VectorAssembler(inputCols=gene_columns, outputCol="_gene_vec")
    scaler = StandardScaler(inputCol="_gene_vec", outputCol="_gene_vec_scaled", withMean=True, withStd=True)
    pca = PCA(k=k, inputCol="_gene_vec_scaled", outputCol="expression_pca")

    assembled = assembler.transform(df)
    scaled = scaler.fit(assembled).transform(assembled)
    reduced = pca.fit(scaled).transform(scaled)

    reduced = reduced.withColumn("_pca_array", vector_to_array("expression_pca"))
    flat_cols = [F.col("_pca_array")[i].alias(f"expression_pca_{i}") for i in range(k)]
    return reduced.select(id_col, *flat_cols)


def aggregate_mutation_burden(
    mutations_df: DataFrame, gene_pathway_map_df: DataFrame, id_col: str = "patient_key"
) -> DataFrame:
    """Per-mutation-position data is too sparse to use directly (thousands of
    distinct positions, most patients hit none of the same one twice). Roll
    mutations up to pathway-level burden counts instead - a much lower-
    cardinality, more generalizable signal.
    """
    with_pathway = mutations_df.join(gene_pathway_map_df, on="gene", how="left").withColumn(
        "pathway", F.coalesce(F.col("pathway"), F.lit("unmapped"))
    )
    return (
        with_pathway.groupBy(id_col, "pathway")
        .agg(F.count("*").alias("mutation_count"))
        .groupBy(id_col)
        .pivot("pathway")
        .agg(F.first("mutation_count"))
        .na.fill(0)
    )
