package prolab

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Paths}
import java.time.LocalDateTime

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.json4s.DefaultFormats
import org.json4s.native.Json

import scala.collection.mutable.ArrayBuffer
import scala.math.BigDecimal.RoundingMode.HALF_DOWN


object Lab9Server {

  val userId = 201
  val neighbourLimit = 30
  val tableName = "data"

  def main(args: Array[String]): Unit = {
    val spark = prepareEnv(tableName)

    val (average_user_ratings, average_film_ratings, completeness) = estimatePart1(spark)
    val (pearsonNeighbours, pearsonTop10) = estimatePart2(spark)
    val averageRating = estimatePart3(spark)
    val (top10Predicators, top10PositivePredicators) = estimatePart4(spark)

    val result = Map[String, Any](
      "average_user_ratings" -> BigDecimal.valueOf(average_user_ratings).setScale(4, HALF_DOWN),
      "average_film_ratings" -> BigDecimal.valueOf(average_film_ratings).setScale(4, HALF_DOWN),
      "completeness" -> BigDecimal.valueOf(completeness).setScale(4, HALF_DOWN),
      "average_rating" -> BigDecimal.valueOf(averageRating).setScale(4, HALF_DOWN),
      "pearson_neighbours" -> pearsonNeighbours,
      "pearson_top10" -> pearsonTop10,
      "predicators_top10" -> top10Predicators,
      "predicators_positive_top10" -> top10PositivePredicators
    )
    saveResult(result)
  }

  def prepareEnv(tableName: String): SparkSession = {
    val dataLocation = "src/main/resources/ml-100k/u.data"
    val warehouseLocation = "spark-warehouse"

    val spark = SparkSession
      .builder()
      .appName("lab9")
      .master("local[*]")
      .config("spark.sql.crossJoin.enabled", true)
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      //      .config("spark.sql.warehouse.dir", Paths.get(warehouseLocation).toAbsolutePath.toString)
      //      .enableHiveSupport()
      .getOrCreate()

    spark.udf.register("pearsonCorrelation", pearsonCorrelation _)
    spark.sparkContext.setLogLevel("ERROR")
    Logger.getLogger("org.apache.parquet").setLevel(Level.OFF)

    loadData(spark, tableName, dataLocation)
    //    loadHiveData(spark, tableName, dataLocation)
    //    loadTest(spark, tableName)
    spark
  }

  def pearsonCorrelation(u: Map[Int, Int], a: Map[Int, Int]): Double = {
    val ru = u.values.sum * 1.0 / u.values.size
    val ra = a.values.sum * 1.0 / a.values.size

    var numeratorSum = 0.0
    var denominatorSum1 = 0.0
    var denominatorSum2 = 0.0

    val intersection = u.keys.toSet.intersect(a.keys.toSet)
    if (intersection.isEmpty || u.values.toSet.size == 1 || a.values.toSet.size == 1) return 0.0

    for (movieId <- intersection) {
      val multu = u(movieId) - ru
      val multa = a(movieId) - ra

      numeratorSum += multu * multa
      denominatorSum1 += multu * multu
      denominatorSum2 += multa * multa
    }
    val coeff = math.min(intersection.size * 1.0 / 50, 1.0)
    coeff * numeratorSum / (math.sqrt(denominatorSum1) * math.sqrt(denominatorSum2))
  }

  def estimatePart1(spark: SparkSession): (Double, Double, Double) = {
    val data = spark.sql(
      s"""
         |SELECT
         |       COUNT(rating) / COUNT(DISTINCT user_id) average_user_ratings,
         |       COUNT(rating) / COUNT(DISTINCT movie_id) average_film_ratings,
         |       COUNT(rating) / (COUNT(DISTINCT user_id) * COUNT(DISTINCT movie_id)) completeness
         |FROM $tableName
      """.stripMargin)
      .first()
    val average_user_ratings = data.getAs[Double]("average_user_ratings")
    val average_film_ratings = data.getAs[Double]("average_film_ratings")
    val completeness = data.getAs[Double]("completeness")
    (average_user_ratings, average_film_ratings, completeness)
  }

  def estimatePart2(spark: SparkSession): (Array[Int], Array[Int]) = {
    import spark.implicits._

    spark.sql(
      s"""
         |SELECT user_id, movie_id, rating
         |FROM $tableName
      """.stripMargin)
      .rdd
      .map(row => (row.getAs[Int]("user_id"), row.getAs[Int]("movie_id") -> row.getAs[Int]("rating")))
      .aggregateByKey(new ArrayBuffer[(Int, Int)]())((acc, e) => acc += e, (acc1, acc2) => acc1 ++ acc2)
      .mapValues(v => v.toMap)
      .toDF("user_id", "ratings")
      .createOrReplaceTempView("user_ratings")

    val neighbours = spark.sql(
      s"""
         |SELECT u.user_id, pearsonCorrelation(u.ratings, a.ratings) pc
         |FROM user_ratings u, (SELECT user_id, ratings FROM user_ratings WHERE user_id = $userId) a
         |WHERE u.user_id != a.user_id
         |ORDER BY pc DESC
         |LIMIT $neighbourLimit
      """.stripMargin)
    neighbours.createOrReplaceTempView("neighbours")

    spark.sql(
      s"""
         |SELECT t.movie_id, t.user_id, t.rating, n.pc
         |FROM $tableName t INNER JOIN neighbours n ON t.user_id = n.user_id
         |                  LEFT JOIN (SELECT movie_id FROM $tableName WHERE user_id = $userId) u ON t.movie_id = u.movie_id
         |WHERE u.movie_id IS NULL
      """.stripMargin)
      .createOrReplaceTempView("neighbour_movies")

    spark.sql(
      s"""
         |SELECT
         |   nm.movie_id,
         |   (nm.rating - (SELECT AVG(rating) FROM $tableName WHERE user_id = nm.user_id GROUP BY user_id)) * nm.pc AS mark,
         |   nm.pc
         |FROM neighbour_movies nm
      """.stripMargin)
      .createOrReplaceTempView("neighbour_movie_marks")

    val pearsonTop10 = spark.sql(
      s"""
         |SELECT
         |   movie_id,
         |   (SELECT AVG(rating) FROM $tableName WHERE user_id = $userId)
         |   +
         |   SUM(mark) / SUM(ABS(pc))
         |   AS neighbour_rating
         |FROM neighbour_movie_marks nmm
         |GROUP BY movie_id
         |ORDER BY neighbour_rating DESC, movie_id ASC
         |LIMIT 10
      """.stripMargin)
      .map(row => row.getAs[Int]("movie_id"))
      .collect()

    spark.sqlContext.dropTempTable("user_ratings")
    spark.sqlContext.dropTempTable("neighbours")
    spark.sqlContext.dropTempTable("neighbour_movies")
    spark.sqlContext.dropTempTable("neighbour_movie_marks")

    val pearsonNeighbours = neighbours.map(row => row.getAs[Int]("user_id")).collect()
    (pearsonNeighbours, pearsonTop10)
  }

  def estimatePart3(spark: SparkSession): Double = {
    import spark.implicits._

    spark.sql(
      s"""
         |SELECT user_id, SUM(variance) / (COUNT(movie_id) + 10) AS value
         |FROM
         |(
         |SELECT user_id, movie_id, rating - (SELECT AVG(rating) FROM $tableName) variance
         |FROM $tableName
         |)
         |GROUP BY user_id
      """.stripMargin)
    .createOrReplaceTempView("user_base_predictors")

    spark.sql(
      s"""
         |SELECT movie_id, SUM(diff) / (COUNT(movie_id) + 25) value
         |FROM
         |(
         |SELECT
         |      movie_id,
         |      rating - ubp.value - (SELECT AVG(rating) FROM $tableName) AS diff
         |FROM $tableName t, user_base_predictors ubp
         |WHERE t.user_id = ubp.user_id
         |)
         |GROUP BY movie_id
      """.stripMargin)
      .createOrReplaceTempView("movie_base_predictors")

    spark.sql(
      s"""
         |SELECT ubp.user_id, mbp.movie_id, ubp.value + mbp.value + (SELECT AVG(rating) FROM $tableName) value
         |FROM user_base_predictors ubp, movie_base_predictors mbp
      """.stripMargin)
    .createOrReplaceTempView("bu")

    spark.sqlContext.dropTempTable("user_base_predictors")
    spark.sqlContext.dropTempTable("movie_base_predictors")
    spark.sqlContext.cacheTable("bu")

    val average_rating = spark.sql(
      s"""
         |SELECT AVG(rating) avg_rating
         |FROM $tableName
      """.stripMargin)
      .map(row => row.getAs[Double]("avg_rating"))
      .first()
    average_rating
  }

  def estimatePart4(spark: SparkSession): (Array[Int], Array[Int]) = {

    println("before new_data", LocalDateTime.now())
    val newData = spark.sql(
      s"""
         |SELECT bu.user_id, bu.movie_id, COALESCE(t.rating, bu.value) - bu.value AS new_rating
         |FROM bu LEFT JOIN $tableName t ON bu.user_id = t.user_id AND bu.movie_id = t.movie_id
      """.stripMargin)
    newData.createOrReplaceTempView("new_data")
    spark.sqlContext.cacheTable("new_data")
    newData.write.parquet("parquet/new_data")
    println("after new_data", LocalDateTime.now())

    spark.sql(
      s"""
         |SELECT movie_id, rating
         |FROM $tableName
         |WHERE user_id = $userId
      """.stripMargin)
      .createOrReplaceTempView("user_movies")

    val unwatchedMovies = spark.sql(
      s"""
         |SELECT all_movies.movie_id
         |FROM
         |(SELECT movie_id
         |FROM $tableName
         |GROUP BY movie_id) all_movies
         |LEFT JOIN user_movies
         |ON all_movies.movie_id = user_movies.movie_id
         |WHERE user_movies.movie_id IS NULL
      """.stripMargin)
    unwatchedMovies.createOrReplaceTempView("unwatched_movies")
    spark.sqlContext.cacheTable("unwatched_movies")
    println("before unwatched_movies", LocalDateTime.now())
    unwatchedMovies.write.parquet("parquet/unwatched_movies")
    println("after unwatched_movies", LocalDateTime.now())

    val similarities = spark.sql(
      """
        |SELECT
        |  f1.movie_id m1,
        |  f2.movie_id m2,
        |  SUM(f1.new_rating * f2.new_rating) /
        |  (sqrt(SUM(f1.new_rating * f1.new_rating)) * sqrt(SUM(f2.new_rating * f2.new_rating)))
        |  AS sim
        |FROM new_data f1 INNER JOIN new_data f2 ON f1.user_id = f2.user_id
        |WHERE f1.movie_id != f2.movie_id AND f1.movie_id IN (SELECT movie_id FROM unwatched_movies)
        |GROUP BY f1.movie_id, f2.movie_id
      """.stripMargin)
    similarities.createOrReplaceTempView("similarities")

    spark.sqlContext.cacheTable("similarities")
    similarities.write.parquet("parquet/similarities")

    spark.sqlContext.dropTempTable("new_data")
    spark.sqlContext.dropTempTable("unwatched_movies")

    (estimateSimilarity(spark, "cut_similarities", "1 = 1"), estimateSimilarity(spark, "pos_cut_similarities", "new_sim.sim >= 0"))
  }

  def estimateSimilarity(spark: SparkSession, similarityTable: String, additionalCondition: String): Array[Int] = {
    val cut_similarities = spark.sql(
      s"""
         |SELECT
         |    new_sim.m1,
         |    new_sim.sim,
         |    new_sim.sim * (user_movies.rating - bu.value) AS diff
         |FROM
         |  (
         |     SELECT
         |         ROW_NUMBER() OVER (PARTITION BY m1 ORDER BY sim DESC, m2 ASC) rank,
         |         m1,
         |         m2,
         |         sim
         |     FROM similarities
         |  ) new_sim INNER JOIN user_movies ON new_sim.m2 = user_movies.movie_id
         |            INNER JOIN bu          ON new_sim.m2 = bu.movie_id
         |
         |WHERE
         |     new_sim.rank <= 30
         |     AND bu.user_id = $userId
         |     AND $additionalCondition
      """.stripMargin)
    cut_similarities.createOrReplaceTempView(similarityTable)

    spark.sqlContext.cacheTable(similarityTable)
    cut_similarities.write.parquet(s"parquet/$similarityTable")

    import spark.implicits._
    val top10Predicators = spark.sql(
      s"""
         |SELECT
         |  m1 AS movie_id,
         |  AVG(bu.value) + SUM(diff) / SUM(ABS(sim)) AS estimate
         |FROM cut_similarities cs INNER JOIN bu ON cs.m1 = bu.movie_id
         |WHERE bu.user_id = $userId
         |GROUP BY m1
         |ORDER BY estimate DESC, m1 ASC
         |LIMIT 10
      """.stripMargin)
      .map(row => row.getAs[Int]("movie_id"))
      .collect()

    spark.sqlContext.dropTempTable(similarityTable)

    top10Predicators
  }

  def loadTestData(spark: SparkSession, tableName: String): Unit = {
    val data = "1\t242\t3\t881250949\n1\t302\t3\t891717742\n2\t302\t1\t878887116\n3\t302\t2\t880606923\n3\t242\t1\t886397596\n3\t474\t4\t884182806\n3\t265\t2\t881171488\n4\t302\t5\t891628467\n4\t242\t3\t886324817\n6\t302\t3\t883603013\n4\t257\t2\t879372434\n5\t302\t5\t879781125\n5\t222\t5\t876042340\n5\t242\t3\t891035994\n6\t29\t3\t888104457\n6\t785\t3\t879485318\n6\t242\t5\t879270459\n6\t274\t2\t879539794\n7\t302\t4\t874834944\n8\t242\t2\t892079237\n8\t302\t4\t886176814\n8\t486\t4\t892738452\n9\t302\t4\t877881320\n7\t118\t2\t874833878"
    val seq = data
      .split("\n")
      .map(x => x.split("\t"))
      .map(x => (x(0).toInt, x(1).toInt, x(2).toInt))
      .toSeq
    val df = spark.createDataFrame(seq).toDF("user_id", "movie_id", "rating")
    df.createOrReplaceTempView(tableName)
  }

  def loadData(spark: SparkSession, tableName: String, dataLocation: String): Unit = {
    val data = spark.read
      .option("header", true)
      .option("delimiter", "\t")
      .option("inferSchema", true)
      .csv(dataLocation)
    data.createOrReplaceTempView(tableName)
  }

  def loadDataAndInitHive(spark: SparkSession, tableName: String, dataLocation: String): Unit = {
    if (spark.sql(s"SHOW TABLES LIKE '$tableName'").collect().length == 0) {
      val createTableSql =
        s"""CREATE TABLE IF NOT EXISTS $tableName (user_id integer, movie_id integer, rating integer, timestamp integer)
           |row format delimited fields terminated by '\t'"""
          .stripMargin
      val loadDataSql = s"LOAD DATA LOCAL INPATH '$dataLocation' INTO TABLE $tableName"
      spark.sql(createTableSql)
      spark.sql(loadDataSql)
    }
  }

  def saveResult(result: Map[String, Any]): Unit = {
    val json = Json(DefaultFormats).write(result)
    println(json)
    Files.write(Paths.get("lab09.json"), json.getBytes(UTF_8))
  }
}
