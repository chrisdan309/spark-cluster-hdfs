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

spark.stop()
