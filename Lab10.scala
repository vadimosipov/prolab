package prolab

import java.io.{FileOutputStream, ObjectOutputStream}
import java.nio.file.{Files, Paths}
import java.time.LocalDateTime

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

import scala.collection.{Map, mutable}
import scala.collection.mutable.ArrayBuffer

object Lab10 {

  case class Vertex(id: Int, rating: Double, catalog: Int, numberClickFrom: Int)

  val ozonTrainTable = "ozon_train"
  val ozonTestTable = "ozon_test"
  val catalogsTable = "catalogs"
  val catalogPathTable = "catalog_path"
  val ratingsTable = "ratings"
  val itemAttrsTable = "items" // ozon_items_full
  val mergedTable = "merge"

  val emptyArray = new ArrayBuffer[Vertex]()

  val isThereSavedMergedTrainData = true
  val pathToParquetData = "parquet/merged_train_data"

  val dataLocation = s"/home/vadim/other/prolab/lab10/"

  def main(args: Array[String]): Unit = {
    println("start:", LocalDateTime.now())

    val spark = prepareEnv()
    if (!isThereSavedMergedTrainData) {
      loadDataOzonTrain(spark, dataLocation)
      loadDataCatalogs(spark, dataLocation)
      loadRatings(spark, dataLocation)
      saveMergedTrainDataToParquet(spark)
    }
//    readMergedTrainData(spark)

    saveCatalogPath(spark, dataLocation)

//    test(spark)

    loadItemAttrs(spark, dataLocation)
    spark.sql(
      s"""
        |SELECT parent_id, COUNT(1)
        |FROM $itemAttrsTable
        |GROUP BY parent_id
        |HAVING COUNT(parent_id) > 3
      """.stripMargin)
      .show(false)


    println("end:", LocalDateTime.now())
  }

  def saveCatalogPath(spark: SparkSession, dataLocation: String): Unit = {
    if (Files.notExists(Paths.get("catalogpathmap"))) {
      println("aaa")
      loadCatalogPath(spark, dataLocation)
      import spark.implicits._
      val catalogPathMap = spark.sql(
        s"""
           |SELECT *
           |FROM $catalogPathTable
      """.stripMargin)
        .rdd
        .map(row => {
          val catalog = row.getAs[Int]("catalog")
          val brothers = row.getAs[mutable.WrappedArray[Int]]("brothers")
          (catalog, brothers.toSet)
        })
        .collectAsMap()

      val out = new ObjectOutputStream(new FileOutputStream("catalogpathmap"))
      out.writeObject(catalogPathMap)
      out.close()
    }
  }

  def test(spark: SparkSession): Unit = {
    println("start sql", LocalDateTime.now())

    spark.sql(
      s"""
         |SELECT *
         |FROM $mergedTable
         |WHERE catalog1 IS NULL OR catalog2 IS NULL
      """.stripMargin)
      .show(false)

    spark.sql(
      s"""
         |SELECT COUNT(1)
         |FROM $mergedTable
         |WHERE catalog1 IS NULL OR catalog2 IS NULL
      """.stripMargin)
      .show(false)

    spark.sql(
      s"""
         |SELECT *
         |FROM $mergedTable
      """.stripMargin)
      .describe("rating1", "rating2", "click")
      .show(50, false)


    spark.sql(
      s"""
         |SELECT *
         |FROM $mergedTable
         |WHERE item1 = 28759795 OR item2 = 28759795
      """.stripMargin)
      .show(false)

  }

  def saveOutgoingMap(spark: SparkSession): Unit = {
    val incoming = null
    //getIncomingItemsMap(spark)
    val outgoing = getOutgoingItemsMap(spark)
    val out = new ObjectOutputStream(new FileOutputStream("outgoingmap"))
    out.writeObject(outgoing)
    out.close()
  }


  def getIncomingItemsMap(spark: SparkSession): collection.Map[Int, ArrayBuffer[Int]] = {
    import spark.implicits._

    spark.sql(
      s"""
         |SELECT item1, item2
         |FROM $mergedTable
         |GROUP BY item1, item2
      """.stripMargin)
      .map(row => (row.getAs[Int]("item1"), row.getAs[Int]("item2")))
      .rdd
      .aggregateByKey(new ArrayBuffer[Int]())((acc, v) => acc += v, (acc1, acc2) => acc1 ++ acc2)
      .collectAsMap()
  }

  def getOutgoingItemsMap(spark: SparkSession): collection.Map[Int, ArrayBuffer[Vertex]] = {
    import spark.implicits._

    spark.sql(
      s"""
         |SELECT item1, rating1, catalog1, item2, click
         |FROM $mergedTable
         |GROUP BY item1, rating1, catalog1, item2, click
      """.stripMargin)
      .map(row => (row.getAs[Int]("item2"),
        Vertex(row.getAs[Int]("item1"), row.getAs[Double]("rating1"), row.getAs[Int]("catalog1"), row.getAs[Int]("click"))
      ))
      .rdd
      .aggregateByKey(new ArrayBuffer[Vertex]())((acc, v) => acc += v, (acc1, acc2) => acc1 ++ acc2)
      .collectAsMap()
  }

  def prepareEnv(): SparkSession = {
    val spark = SparkSession
      .builder()
      .appName("lab10")
      .master("local[2]")
      .config("spark.sql.crossJoin.enabled", true)
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    Logger.getLogger("org.apache.parquet").setLevel(Level.OFF)

    spark
  }

  def loadDataOzonTrain(spark: SparkSession, dataLocation: String): Unit = {
    val schema = StructType(Array(
      StructField("item", DataTypes.StringType, false),
      StructField("true_recoms", DataTypes.createMapType(DataTypes.StringType, DataTypes.IntegerType, false), false)
    ))

    import spark.implicits._
    val data = spark.read
      .schema(schema)
      .json(dataLocation + ozonTrainTable + ".txt")


    data.flatMap(row => {
      val buffer = new ArrayBuffer[(Int, Int, Int)]()
      val item = row.getAs[String]("item")
      val trueRecoms = row.getAs[Map[String, Int]]("true_recoms")

      if (item != "") {
        val intItem = item.toInt
        for ((k, v) <- trueRecoms) {
          buffer += ((intItem, k.toInt, v))
        }
      }
      buffer
    })
      .toDF("item", "neighbour", "click")
      .createOrReplaceTempView(ozonTrainTable)
  }

  def loadDataCatalogs(spark: SparkSession, dataLocation: String): Unit = {

    val schema = StructType(Array(
      StructField("itemid", DataTypes.StringType, false),
      StructField("catalogid", DataTypes.StringType, false)
    ))

    val data = spark.read
      .schema(schema)
      .json(dataLocation + catalogsTable)

    import spark.implicits._
    data.map(row => {
      val itemid = row.getAs[String]("itemid").toInt
      val catalogid = row.getAs[String]("catalogid").toInt
      (itemid, catalogid)
    })
      .toDF("item", "catalog")
      .createOrReplaceTempView(catalogsTable)
  }

  def loadCatalogPath(spark: SparkSession, dataLocation: String): Unit = {

    val schema = StructType(Array(
      StructField("catalogid", DataTypes.StringType, false),
      StructField("catalogpath",
        DataTypes.createArrayType(MapType(DataTypes.StringType, DataTypes.StringType, false)),
        false)
    ))

    val data = spark.read
      .schema(schema)
      .json(dataLocation + catalogPathTable)

    import spark.implicits._
    data.map(row => {
      val catalogid = row.getAs[String]("catalogid").toInt
      val catalogpath = row.getAs[mutable.WrappedArray[Map[String, String]]]("catalogpath")

      val brothers = new mutable.ArrayBuffer[Int]()
      for (catalog <- catalogpath) {
        for ((k, v) <- catalog) {
          brothers += k.toInt
        }
      }
      (catalogid, brothers)
    })
      .toDF("catalog", "brothers")
      .createOrReplaceTempView(catalogPathTable)
  }

  def loadRatings(spark: SparkSession, dataLocation: String): Unit = {
    val schema = StructType(Array(
      StructField("itemid", DataTypes.IntegerType, false),
      StructField("rating", DataTypes.DoubleType, false)
    ))

    spark.read
      .schema(schema)
      .json(dataLocation + ratingsTable)
      .createOrReplaceTempView(ratingsTable)
  }

  def loadDataOzonTest(spark: SparkSession, dataLocation: String): Unit = {
    val schema = StructType(Array(
      StructField("item", DataTypes.StringType, false),
      StructField("recoms", DataTypes.IntegerType, false)
    ))

    val data = spark.read
      .schema(schema)
      .json(dataLocation + ozonTestTable + ".txt")

    import spark.implicits._
    data
      .map(row => {
        row.getAs[String]("item").toInt
      })
      .toDF("item")
      .createOrReplaceTempView(ozonTestTable)
  }

  def loadItemAttrs(spark: SparkSession, dataLocation: String): Unit = {
    val fields = new ArrayBuffer[StructField]()
    var i = 1
    while (i <= 77) {
      fields += StructField(s"attr$i", DataTypes.StringType)
      i += 1
    }
    fields += StructField("itemid", DataTypes.StringType, false)
    fields += StructField("parent_id", DataTypes.IntegerType, false)

    val schema = StructType(fields)
    val data = spark.read.parquet(dataLocation + itemAttrsTable)

    data.createOrReplaceTempView(itemAttrsTable)
  }

  def loadMergedTestDataAndCatalogs(spark: SparkSession): Array[(Int, Int)] = {
    loadDataCatalogs(spark, dataLocation)
    loadDataOzonTest(spark, dataLocation)

    spark.sql(
      s"""
        |SELECT c.item, c.catalog
        |FROM $catalogsTable c INNER JOIN $ozonTestTable t ON c.item = t.item
      """.stripMargin)
      .rdd
      .map(row => (row.getAs[Int]("item"), row.getAs[Int]("catalog")))
      .collect()
  }


  def saveMergedTrainDataToParquet(spark: SparkSession): Unit = {
    spark.sql(
      s"""
         |SELECT
         |   tr.item      AS item1, ct1.catalog AS catalog1, rt1.rating AS rating1,
         |   tr.neighbour AS item2, ct2.catalog AS catalog2, rt2.rating AS rating2,
         |   tr.click
         |FROM $ozonTrainTable tr LEFT JOIN $catalogsTable ct1 ON tr.item = ct1.item
         |                        LEFT JOIN $catalogsTable ct2 ON tr.neighbour = ct2.item
         |                        LEFT JOIN $ratingsTable rt1 ON tr.item = rt1.itemid
         |                        LEFT JOIN $ratingsTable rt2 ON tr.neighbour = rt2.itemid
      """.stripMargin)
      .write
      .parquet(pathToParquetData)
  }

  def readMergedTrainData(spark: SparkSession): Unit = {
    spark
      .read
      .parquet(pathToParquetData)
      .createTempView(mergedTable)
    spark.sqlContext.cacheTable(mergedTable)
  }

}
