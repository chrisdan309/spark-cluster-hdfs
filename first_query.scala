import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession.builder()
  .appName("ConsultaAgricultura")
  .master("spark://namenode:7077")
  .getOrCreate()

val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("hdfs://namenode:9000/data/productos_limpio.csv")
  .filter(col("PRODUCCION") > 0 && col("PRECIO_CHACRA") > 0)
  .repartition(6)

val mr1 = df.select(
  col("DEPARTAMENTO"), col("CULTIVO"),
  (col("PRECIO_CHACRA") * col("PRODUCCION")).alias("ponderado"),
  col("PRODUCCION")
)

val mr2 = mr1.groupBy("DEPARTAMENTO", "CULTIVO")
  .agg(
    sum("ponderado").alias("suma_ponderada"),
    sum("PRODUCCION").alias("suma_produccion")
  )

val mr3 = mr2.withColumn("promedio_ponderado",
  col("suma_ponderada") / col("suma_produccion")
)

println("--- Consulta 1: Promedio ponderado del precio chacra ---")
mr3.select("DEPARTAMENTO", "CULTIVO", "promedio_ponderado").show(20, false)

val mr1b = df.select("ANO", "CULTIVO", "PRECIO_CHACRA")

val mr2b = mr1b.groupBy("ANO", "CULTIVO")
  .agg(avg("PRECIO_CHACRA").alias("promedio_anual"))

val mr3b = mr2b.orderBy("CULTIVO", "ANO")
  .withColumn("prev_anio", lag("promedio_anual", 1).over(
    org.apache.spark.sql.expressions.Window.partitionBy("CULTIVO").orderBy("ANO")
  ))
  .withColumn("variacion_porcentual",
    when(col("prev_anio").isNotNull,
         ((col("promedio_anual") - col("prev_anio")) / col("prev_anio")) * 100)
  )

println("--- Consulta 2: Variación porcentual de precio anual por cultivo ---")
mr3b.select("ANO", "CULTIVO", "promedio_anual", "variacion_porcentual").show(20, false)

spark.stop()
