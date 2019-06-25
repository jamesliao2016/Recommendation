package org.jd.datamill.rec.feature
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.linalg.{Vector, Vectors}
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DoubleType, StringType}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.joda.time.DateTime
import scala.collection.mutable

import org.jd.datamill.rec.Utils

object FeatureEngineerNoUserAttr {

  //final step : generate features
  def generateFeatures(spark: SparkSession, df: DataFrame, tree: JsonNode, jobType: String): DataFrame = {

    //final cols after process
    val colsAll = ArrayBuffer[Column]()

    /** 0. user brand cate ...*/
    if (jobType == "train") {
      colsAll.append(col("y_value"))
      colsAll.append(col("user_log_acct"))
      colsAll.append(col("main_brand_code"))
      colsAll.append(col("item_third_cate_cd"))
      colsAll.append(col("sample_type"))
    } else if (jobType == "predict") {
      colsAll.append(col("user_log_acct"))
      colsAll.append(col("main_brand_code"))
      colsAll.append(col("item_third_cate_cd"))
    } else {
      println("Error feature generator type!")
      sys.exit(1)
    }

    /** 1. continue value, so put in directly */
    val orgFeature = tree.get("org_feature").iterator()
    //cols no need to process, put into directly
    while (orgFeature.hasNext) {
      val cur = orgFeature.next().asText()
      println(cur)
      colsAll.append(col(cur).cast(DoubleType))
    }
    //No user attr
    //val orgUserFeature = tree.get("org_user_feature").iterator()

    /** 2. ratio features */
    val colsRatioFeatures = calcRatioFeatures(tree,"ratio_feature")
    colsAll.appendAll(colsRatioFeatures)

    ///** 3. user_cat_attr: onehot or indexed or setMapping */

    val resDF = df
      .select(colsAll.toArray: _*)
    resDF
  }

  //calculate ratio features: numerators/denominators
  def calcRatioFeatures(tree:JsonNode,name:String):Array[Column] = {

    val arrBuf = new ArrayBuffer[Column]

    val it = tree.path(name).iterator()
    while (it.hasNext) {
      val node = it.next()
      //fenzi
      val numerators = node.path("numerators").iterator()
      var colNume: Column = (lit(0)).as("zero")
      var colNumeName = "numerator"
      while (numerators.hasNext) {
        val numerator = numerators.next().asText()
        colNumeName = colNumeName+"_"+numerator
        val f = udf { (add1: Double,add2: Double) =>
          add1 + add2
        }
        colNume = f(colNume.cast(DoubleType),col(numerator).cast(DoubleType))
      }
      //colNume = colNume.as(colNumeName)
      //fenmu
      val denominators = node.path("denominators").iterator()
      var colDeno = (lit(0)).as("zero")
      var colDenoName = "denominator"
      while (denominators.hasNext) {
        val denominator = denominators.next().asText()
        colDenoName = colDenoName+"_"+denominator
        val f = udf { (add1: Double,add2: Double) =>
          add1 + add2
        }
        colDeno = f(colDeno.cast(DoubleType),col(denominator).cast(DoubleType))
      }
      //colDeno = colDeno.as(colDenoName)

      val newName = colNumeName+"_devide_"+colDenoName
      //devide
      arrBuf.append(Utils.ratioColumn(colNume,colDeno,newName))
    }

    arrBuf.toArray
  }

}
