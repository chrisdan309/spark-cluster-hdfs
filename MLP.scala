import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.VectorAssembler

val spark = SparkSession.builder()
  .appName("ClasificacionAgricolaMLP")
// .master("spark://namenode:7077")
  .master("local[*]")
  .getOrCreate()

val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
// .csv("hdfs://namenode:9000/data/productos_limpio.csv")
  .csv("data/productos_limpio.csv")
  .filter(col("PRODUCCION").isNotNull && col("PRECIO_CHACRA").isNotNull && col("VERDE_ACTUAL").isNotNull)
  .cache()

val binarized = df.withColumn("label", when(col("PRODUCCION") > 0, 1).otherwise(0))

val assembler = new VectorAssembler()
  .setInputCols(Array("SIEMBRA", "VERDE_ACTUAL", "PRECIO_CHACRA"))
  .setOutputCol("features")

val prepared = assembler.transform(binarized).select("label", "features")

val layers = Array(3, 5, 2)

val mlp = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(4)
  .setSeed(42L)
  .setMaxIter(100)

val model = mlp.fit(prepared)
val predictions = model.transform(prepared)

println("\n====== Clasificación con Multilayer Perceptron ======")
predictions.select("features", "label", "prediction").show(10, truncate = false)

spark.stop()
