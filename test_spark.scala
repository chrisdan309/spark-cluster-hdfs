val file = spark.read.text("hdfs://namenode:9000/user/nodo/README.md")
val count = file.count()
println(s"Cantidad de lineas: $count")
