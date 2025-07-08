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

println("Cantidad de registros: " + df.count())