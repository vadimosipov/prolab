package prolab

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Paths}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, IDF, RegexTokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s.DefaultFormats
import org.json4s.native.Json
import prolab.Lab10FindBestNeighbours.Info

import scala.collection.{Map, immutable, mutable}
import scala.collection.mutable.ArrayBuffer
import scala.math.BigDecimal.RoundingMode

object Lab10FindRecommends {


  def saveResult(itemVectorArray: Map[Int, Map[String, Int]], fileName: String):Unit = {
//{"item": "28759795", "recoms": {"28759801": 1, "28759817": 2, "28759803": 13}}

    val map = new mutable.HashMap[String, Any]()
    for ((k, v) <- itemVectorArray) {
      map.put("item", k.toString)
      map.put("recoms", v)
    }

    val json = Json(DefaultFormats).write(map)
    println(json)
    Files.write(Paths.get(fileName), json.getBytes(UTF_8))
  }

  def main(args: Array[String]): Unit = {
    val spark = Lab10.prepareEnv()
    spark.udf.register("cosineSimilarity", cosineSimilarity _)
    Lab10.loadItemAttrs(spark, Lab10.dataLocation)
    val vertexCatalogArray = Lab10.loadMergedTestDataAndCatalogs(spark)

//    val vertex = 28759795
//    val catalog = 1175217

    val countVectorizerMap = new scala.collection.mutable.HashMap[Int, Map[String, Int]]()
    val tfidfMap = new scala.collection.mutable.HashMap[Int, Map[String, Int]]()

    for ((vertex, catalog) <- vertexCatalogArray) {
      val summary = Lab10FindBestNeighbours.find(vertex, catalog)

      import scala.collection.JavaConversions._
      val bApplicants = spark.sparkContext.broadcast(summary.vertexes.toMap)

      val wordsData = transformItemAttrAndTokenize(spark, bApplicants, vertex)

      countVectorize(spark, wordsData)
      val itemVectorArray = estimateCosSimilarity(spark, vertex, "item_vectors")
      countVectorizerMap.put(vertex, itemVectorArray)

      estimateTfIDF(spark, wordsData)
      val tfidfArray = estimateCosSimilarity(spark, vertex, "tfidf")
      tfidfMap.put(vertex, tfidfArray)
    }
    saveResult(countVectorizerMap, "ozon-test_count_vect.txt")
    saveResult(tfidfMap, "ozon-test_tfidf.txt")

//    println(itemVectorArray.toSet.intersect(tfidfArray.toSet).size * 1.0 / tfidfArray.size)
  }

  def transformItemAttrAndTokenize(spark: SparkSession, applicants: Broadcast[immutable.Map[Int, Info]], vertex: Int): DataFrame = {
    import spark.implicits._

    val itemsAttrs = spark.sql(
      s"""
         |SELECT *
         |FROM ${Lab10.itemAttrsTable}
      """.stripMargin)
      .rdd
      .filter(row => {
        val item = row.getAs[String]("itemid").toInt
        applicants.value.contains(item) || item == vertex
      })
      .map(row => {
        val item = row.getAs[String]("itemid").toInt
        val parent = row.getAs[String]("parent_id")
        val notNullParent = if (parent == null) -1 else parent.toInt

        val buffer = new ArrayBuffer[String]()
        val builder = new StringBuilder(1024)
        var i = 1
        while (i <= 77) {
          var attr = row.getAs[String](s"attr$i")
          if (attr != null) {
            attr = attr.toLowerCase.trim
            builder.append(attr).append(" ")
          }
          buffer += attr
          i += 1
        }

        var level = 1
        var clickNumber = 0
        var rating = 0.0
        if (applicants.value.contains(item)) {
          val info = applicants.value.get(item).get
          level = info.level
          clickNumber = info.numberClickFrom
          rating = info.rating
        } else {
          println(s"no item in applicants ! : $item")
        }

        (item, notNullParent, builder.toString, buffer, level, clickNumber, rating)
      })
      .toDF("item", "parent", "sentence", "attributes", "level", "click", "rating")

    itemsAttrs.cache()
//    itemsAttrs.show(false)


    val wordsData = new RegexTokenizer()
      .setMinTokenLength(2)
      .setPattern("[^\\p{L}\\w\\d]+")
      .setInputCol("sentence")
      .setOutputCol("words")
      .transform(itemsAttrs)
      .map(row => {
        val item = row.getAs[Int]("item")
        val parent = row.getAs[Int]("parent")
        val sentence = row.getAs[String]("sentence")
        val attributes = row.getAs[mutable.WrappedArray[String]]("attributes")
        val words = row.getAs[mutable.WrappedArray[String]]("words").toSet
        val level = row.getAs[Int]("level")
        val click = row.getAs[Int]("click")
        val rating = row.getAs[Double]("rating")

        (item, parent, sentence, attributes, words.toArray, level, click, rating)
      })
      .toDF("item", "parent", "sentence", "attributes", "words", "level", "click", "rating")

    wordsData.cache()
    wordsData
      .select("item", "parent", "words", "level")
//      .show(false)

    wordsData
  }

  def countVectorize(spark: SparkSession, wordsData: DataFrame): Unit = {
    val cvmodel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      //      .setVocabSize(3)
      //      .setMinDF(2)
      .fit(wordsData)
      .transform(wordsData)
    cvmodel.createOrReplaceTempView("item_vectors")
    cvmodel.cache()
  }

  def estimateTfIDF(spark: SparkSession, wordsData: DataFrame): Unit = {
    val featurizedData = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .transform(wordsData)

    val rescaledData = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .fit(featurizedData)
      .transform(featurizedData)
    rescaledData.createOrReplaceTempView("tfidf")
    rescaledData.cache()
  }

  def estimateCosSimilarity(spark: SparkSession, vertex: Int, table: String): Map[String, Int] = {
    val result = spark.sql(
      s"""
         |SELECT
         |    iv.item,
         |    iv.parent,
         |    iv.click,
         |    iv.rating,
         |    iv.level,
         |    iv.click * iv.rating,
         |    100 + iv.click * iv.rating AS ordered,
         |    cosineSimilarity(u.features, iv.features) cos,
         |    iv.sentence
         |
         |FROM $table iv, (SELECT * FROM $table WHERE item = $vertex) u
         |WHERE iv.item != u.item AND iv.parent != u.parent
         |ORDER BY cos DESC, ordered DESC
      """.stripMargin)
        .cache()

//    result.show(200, false)

    import spark.implicits._
     result
       .map(row => (row.getAs[Int]("item").toString, row.getAs[Double]("ordered").toInt))
       .rdd
       .collectAsMap()
  }

  def cosineSimilarity(u: SparseVector, a: SparseVector): Double = {
    var numeratorSum = 0.0
    var denominatorSum1 = 0.0
    var denominatorSum2 = 0.0

    val intersection = u.indices.toSet.intersect(a.indices.toSet)
    if (intersection.isEmpty || intersection.size == 1) return 0.0

    for (wordsInx <- intersection) {
      val multu = u(wordsInx)
      val multa = a(wordsInx)

      numeratorSum += multu * multa
      denominatorSum1 += multu * multu
      denominatorSum2 += multa * multa
    }
    val cos = numeratorSum / (math.sqrt(denominatorSum1) * math.sqrt(denominatorSum2))
    cos
    //    BigDecimal.valueOf(cos).setScale(4, RoundingMode.HALF_DOWN).doubleValue()

  }
}
