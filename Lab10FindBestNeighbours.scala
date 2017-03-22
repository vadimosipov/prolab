package prolab

import java.io.{FileInputStream, ObjectInputStream}
import java.util.concurrent.{ConcurrentHashMap, ConcurrentSkipListSet}

import prolab.Lab10.Vertex

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.forkjoin.ForkJoinPool

object Lab10FindBestNeighbours {


  case class Info(id: Int, rating: Double, catalog: Int, numberClickFrom: Int, level: Int)

  case class Summary(vertexes: ConcurrentHashMap[Int, Info], vertexesIds: ConcurrentSkipListSet[Integer])
  case class Library(outgoing: collection.Map[Int, ArrayBuffer[Vertex]],
                     catalogpath: collection.Map[Int, Set[Int]],
                     nearestCatalogs: ConcurrentSkipListSet[Int])

  def main(args: Array[String]): Unit = {
    val fromVertex = 28759795
    val fromCatalog = 1175217
    find(fromVertex, fromCatalog)
  }

  def find(fromVertex: Int, fromCatalog: Int): Summary = {
    val outgoingmapIn = new ObjectInputStream(new FileInputStream("outgoingmap"))
    val outgoing = outgoingmapIn.readObject().asInstanceOf[collection.Map[Int, ArrayBuffer[Vertex]]]
    outgoingmapIn.close()

    val catalogpathIn = new ObjectInputStream(new FileInputStream("catalogpathmap"))
    val catalogpath = catalogpathIn.readObject().asInstanceOf[collection.Map[Int, Set[Int]]]
    catalogpathIn.close()



    val library = Library(outgoing, catalogpath, new ConcurrentSkipListSet[Int]())
    val summary = findSourceVerteces(fromVertex, fromCatalog, library)

    println("---------------")
    println(library.nearestCatalogs.size)
    println(library.nearestCatalogs)

    //    SaveNumericValues.write(fromVertex, sourceVerteces)
    //    val neighbours = buildGraph(incoming, outgoing)
    //    println("neighbours", neighbours)
    summary
  }

  def findSourceVerteces(fromVertex: Int, fromCatalog: Int, library: Library): Summary = {
    val pool = new ForkJoinPool(6)
    val summary = Summary(new ConcurrentHashMap[Int, Info](), new ConcurrentSkipListSet[Integer]())

    for (catalog <- library.catalogpath(fromCatalog)) {
      library.nearestCatalogs.add(catalog)
    }
    val initialParents = library.outgoing.getOrElse(fromVertex, Lab10.emptyArray)
    val initialLevel = 0
    val task = new OutgoingTask(initialParents, initialLevel, library, summary)
    val answer = pool.invoke(task)

    println("final size: ", summary.vertexes.size(), "answer: ", answer)
    findAverageAndMedianClick(library)
    summary
  }

  def findAverageAndMedianClick(library: Library): Unit = {
    var sum = 0
    var count = 0
    val allClicks = new ArrayBuffer[Int](5000)
    val allRatings = new ArrayBuffer[Double](5000)
    for (parents <- library.outgoing.values) {
      for (p <- parents) {
        sum += p.numberClickFrom
        count += 1
        allClicks += p.numberClickFrom
        if (p.rating != 0.0) allRatings += p.rating
      }
    }
    val a = allClicks.sorted
    val sortedRatings = allRatings.sorted
    println("50%: ", a(allClicks.size / 2),
      "75%: ", a(allClicks.size / 2 + allClicks.size / 4),
      "average: ", sum * 1.0 / count,
      "50% ratings: ", sortedRatings(sortedRatings.size / 2),
      "75% ratings: ", sortedRatings(sortedRatings.size / 2 + sortedRatings.size / 4)
    )


  }

}
