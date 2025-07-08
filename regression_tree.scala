import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.RegressionEvaluator

val spark = SparkSession.builder()
  .appName("RegresionAgricolaDT")
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

val assembler = new VectorAssembler()
  .setInputCols(Array("SIEMBRA", "PRODUCCION", "VERDE_ACTUAL"))
  .setOutputCol("features")

val prepared = assembler.transform(df).withColumnRenamed("PRECIO_CHACRA", "label")

val dt = new DecisionTreeRegressor()
  .setFeaturesCol("features")
  .setLabelCol("label")

val model = dt.fit(prepared)
val predictions = model.transform(prepared)

println("\n====== Regresión con Árbol de Decisión ======")
predictions.select("features", "label", "prediction").show(10, truncate = false)

val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

println(s"\n[RMSE]: ${evaluator.evaluate(predictions)}")

spark.stop()
