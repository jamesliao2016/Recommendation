package org.jd.datamill.rec.train

import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel}

import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map

class XgboostSort {

  def train(spark: SparkSession, x_train: DataFrame, testPath: String, paramStr: String, modelPath: String, modelAuc: Map[String, Double]): Unit = {
    var paramList = new ListBuffer[(String, Any)]()

    val paramArr = paramStr.split(",")
    var num_round = 0
    var nworkers = 0

    paramArr.foreach { row =>
      val arr = row.split(":")
      val (param, value) = (arr(0), arr(1))
      if (param != "num_round" & param != "num_workers") {
        paramList += (s"$param" -> value)
      } else if (param == "num_workers") {
        nworkers = value.toInt
      } else if (param == "num_round") {
        num_round = value.toInt
      }
    }

    paramList += ("objective" -> "binary:logistic")

    //OpenMP optimization in each worker, need to (--conf spark.task.cpus=4)
    paramList += ("nthread" -> 8)


    val paramMap = paramList.toList.toMap
    println("before train paramMap : "+paramMap)

    val xgboostModel: XGBoostModel = XGBoost.trainWithDataFrame(
      x_train, paramMap, num_round, nWorkers = nworkers, useExternalMemory = true)


    val saveModelPath = paramStr.stripSuffix(",")
      //.replace("=", "_")
      .replace(":", "_")
      .replace(",", "_")

    xgboostModel.saveModelAsHadoopFile(s"$modelPath/$saveModelPath")(spark.sparkContext)

    println(s"xgbClassificationModel.params  : ${xgboostModel.params}")

    modelAuc += (saveModelPath -> nworkers)


  }


  def main(args: Array[String]): Unit = {

    val trainPath = args(0)
    val testPath = args(1)
    val paramStr = args(2)
    val modelPath = args(3)

    //val paramStr = "eta:0.05,max_depth:5"

    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("sort")
      .enableHiveSupport()
      .getOrCreate()

    var modelAuc: Map[String, Double] = Map()


    //train(spark, trainPath, testPath, paramStr, modelPath, modelAuc)

  }

}

