import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vector
import java.io._

val inicio = System.nanoTime()

val spark = SparkSession.builder()
  .appName("RegresionAgricolaDT")
// .master("spark://namenode:7077")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._


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

val rmse = evaluator.evaluate(predictions)
println(f"\n[RMSE]: $rmse%.4f")

val fin = System.nanoTime()
val duracionSeg = (fin - inicio) / 1e9d
println(f"\n⏱ Tiempo total de ejecución: $duracionSeg%.2f segundos")

// ✅ Guardar métricas en archivo
val writer = new PrintWriter(new File("resultados_dt.txt"))
writer.write("====== Regresión con Árbol de Decisión ======\n")
writer.write(f"[RMSE]: $rmse%.4f\n")
writer.write(f"⏱ Tiempo total de ejecución: $duracionSeg%.2f segundos\n")
writer.close()

// ✅ Guardar predicciones en CSV plano
val predFlat = predictions.select("features", "label", "prediction")
  .map { row =>
    val vec = row.getAs[Vector]("features").toArray.map(_.toString.toDouble)
    val label = row.getAs[Any]("label").toString.toDouble
    val prediction = row.getAs[Any]("prediction").toString.toDouble
    (vec(0), vec(1), vec(2), label, prediction)
  }.toDF("SIEMBRA", "PRODUCCION", "VERDE_ACTUAL", "label", "prediction")

predFlat
  .coalesce(1)
  .write
  .option("header", "true")
  .mode("overwrite")
  .csv("predicciones_dt.csv")

spark.stop()
