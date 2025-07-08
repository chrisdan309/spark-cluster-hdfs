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

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val evaluatorAcc = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val evaluatorRecall = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("weightedRecall")

val evaluatorF1 = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("f1")

val accuracy = evaluatorAcc.evaluate(predictions)
val recall = evaluatorRecall.evaluate(predictions)
val f1 = evaluatorF1.evaluate(predictions)

println(f"Accuracy: $accuracy%.4f")
println(f"Recall: $recall%.4f")
println(f"F1-score: $f1%.4f")

val loss = 1.0 - (accuracy)
println(f"Loss (1 - accuracy): $loss%.4f") 


println("\n====== Clasificación con Multilayer Perceptron ======")
predictions.select("features", "label", "prediction").show(10, truncate = false)
import java.io._
import org.apache.spark.ml.linalg.Vector
import spark.implicits._

// Guardar métricas en archivo de texto
val pw = new PrintWriter(new File("resultados_mlp.txt"))
pw.write(f"Accuracy: $accuracy%.4f\n")
pw.write(f"Recall: $recall%.4f\n")
pw.write(f"F1-score: $f1%.4f\n")
pw.write(f"Loss (1 - accuracy): $loss%.4f\n")
pw.close()

// Convertir features (Vector) a columnas escalares, con casteo seguro
val predFlat = predictions.select("features", "label", "prediction")
  .map { row =>
    val vec = row.getAs[Vector]("features").toArray.map(_.toString.toDouble)
    val label = row.getAs[Any]("label").toString.toDouble
    val prediction = row.getAs[Any]("prediction").toString.toDouble
    (vec(0), vec(1), vec(2), label, prediction)
  }.toDF("SIEMBRA", "VERDE_ACTUAL", "PRECIO_CHACRA", "label", "prediction")

predFlat
  .coalesce(1) // opcional: un solo CSV
  .write
  .option("header", "true")
  .mode("overwrite")
  .csv("predicciones_mlp.csv")

println("\n====== Clasificación con Multilayer Perceptron ======")
predFlat.show(10, truncate = false)
