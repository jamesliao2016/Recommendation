package org.jd.datamill.rec

import java.net.URL
import java.sql.Date
import java.text.SimpleDateFormat
import java.util.Calendar

import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.joda.time.DateTime

import scala.collection.mutable
import scala.io.Source

object Utils {

  /** I. tables general use*/
  val tableActUBC = "app.wiwin_rec_user_brand_cate_act_merge_without_cate2"
  val tableActUB = "app.wiwin_rec_feature_user_brand_act_d"
  val tableActUC = "app.wiwin_rec_user_cate_act"

  val tableItem = "app.wiwin_rec_brand_cate_all_merge_attr_without_cate2"
  val tableUser = "app.wiwin_rec_feature_user_attr"

  /** II. tables train*/
  //1.sample
  //val tableSample = "app.wiwin_rec_user_brand_cate_train_sample_d_without_cate2"
  val tableSampleActAll = "app.wiwin_rec_user_brand_cate_sample_act_merge_d"
  val tableUsersJoinedSamp = "app.wiwin_rec_feature_train_sample_user_attr"

  //2. user catboost
  //catboost_collected: user,collected,len
  //collected: sex<replace>1<in>0.2,0.3,0.1....<between>age<replace>15<in>0.03,0.02,...
  //val tableTrainUserCatBoostCollected = "galaxy.wiwin_rec_train_user_catboost_collected"


  /** III. tables predict*/


  /** IV. table catboost used in class CatBoostV5 for generating catboost values*/
  val tableCatBoostV5 = "tmp.wiwin_rec_user_feature_pv_follow_cart_search_catboost_v5" //map for train and predict
  val tableCatBoostV5BC = "tmp.wiwin_rec_user_feature_pv_follow_cart_search_catboost_v5_bc" //remove cate2 and cate3, broadcast

  //separator
  val sepIn = "<in>"
  val sepBetween = "<between>"
  val sepReplace = "<replace>"


  /**
    * get date before diff days
    * @return Date String
    */
  def getDaysBefore(orgDt: String, diff: Int): String = {
    var dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd")
    var cal: Calendar = Calendar.getInstance()
    cal.setTime(dateFormat.parse(orgDt))
    cal.add(Calendar.DATE, -diff)
    var yesterday = dateFormat.format(cal.getTime())
    "'" + yesterday + "'"
  }
  def getDateBeforeNDays(n: Int): String ={
    val cal: Calendar = Calendar.getInstance()
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd")
    cal.add(Calendar.DATE, -n)
    val dt = dateFormat.format(cal.getTime())
    dt
  }

  def getTrainDateArr(orgDt: String, dateDiff: Int, nDiff:Int): Array[String] = {
    var diff = 0
    val buff = mutable.ArrayBuffer.empty[String]

    while (diff < dateDiff) {
      buff.append(getDateMinusDays(orgDt,diff).toString)
      diff = diff + nDiff
    }
    buff.toArray
  }
  def getDatePlusDays(date: String,
                      n: Int): Date = {
    val dt = DateTime.parse(date)
    val millis = dt.plusDays(n).getMillis
    new Date(millis)
  }

  def getDateMinusDays(date: String,
                       n: Int): Date = {
    val dt = DateTime.parse(date)
    val millis = dt.minusDays(n).getMillis
    new Date(millis)
  }


  /**
    * calculate the ratio : colNume/colDeno
    * if colDeno=0 return colNume
    * @return Column
    */
  def ratioColumn(colNume: Column,
            colDeno: Column,
            colNew: String): Column = {
    import org.apache.spark.sql.functions._
    val f = udf { (nume: Double,deno: Double) =>
      if(deno==0){
        nume
      }else{
        nume/deno
      }
    }
    f(colNume.cast(DoubleType),colDeno.cast(DoubleType)).as(colNew)
  }

  /**
    * convert string of doubles into Vector
    * String(0.2,0.3,0.5) => Vector[0.2,0.3,0.5]
    * @return Column
    */
  def stringDoubles2Vector(colName:String, n:Int): Column ={

    val f = udf { (str: String) =>
      val arr = str.split(",")
      var vec = Vectors.zeros(n)

      if(arr.length == n){
        vec = Vectors.dense(arr.map(s=>s.toDouble)).compressed
      }
      vec
    }
    val attr = new AttributeGroup(colName+"_vectored", n)
    val meta = attr.toMetadata()
    f(col(colName).cast(StringType)).as(colName+"_vectored", meta)
  }

  /**
    * convert a category value to a Vector(onehot or stringindex) based on values
    * @param values: all values of the feature
    * @return
    */
  def createCatMappingCol(colName: String,
                          newColName: String,
                          enableOneHot: Boolean,
                          values: Array[String]): Column = {
    val n = values.length
    val map = values.zipWithIndex.toMap

    if (enableOneHot) {
      val f = udf { (value: String) =>
        val i = map.getOrElse(value, -1)
        if (i >= 0) {
          Vectors.sparse(n, Array(i), Array(1.0))
        } else {
          Vectors.sparse(n, Array.emptyIntArray, Array.emptyDoubleArray)
        }
      }
      val attr = new AttributeGroup(newColName, n)
      val meta = attr.toMetadata()
      f(col(colName).cast(StringType)).as(newColName, meta)

    } else {

      val f = udf { (value: String) =>
        map.getOrElse(value, -1).toDouble + 1.0
      }
      f(col(colName).cast(StringType)).as(newColName)
    }
  }

  /**
    * convert values to a Vector based on values
    * @param delimiter : delimiter connected multiple values
    * @param values : all values of the feature
    * @return Column
    * for example: input=a#b#c, delimiter=#, values=(a,b,c,d,e,f) then return Vector[1,1,1,0,0,0]
    */
  def createSetMappingCol(colName: String,
                          newColName: String,
                          delimiter: String,
                          values: Array[String]): Column = {
    val n = values.length
    val map = values.zipWithIndex.toMap

    val f = udf { (value: String) =>
      val arr = Array.ofDim[Double](n)
      if(value!=null && value.length()>0){
        value.split(delimiter).foreach { item =>
          val i = map.get(item)
          if (i.isDefined) {
            arr(i.get) = 1.0
          }
        }
      }

      Vectors.dense(arr).compressed
    }
    val attr = new AttributeGroup(newColName, n)
    val meta = attr.toMetadata()
    f(col(colName).cast(StringType)).as(newColName, meta)
  }


  /**
    * resolve configure file from resources
   */
  def resolveConfig(configFile: String): JsonNode = {
    val fileUrl: URL = this.getClass.getClassLoader.getResource(configFile)
    println(fileUrl)
    val mapper = new ObjectMapper()
    val tree: JsonNode = mapper.readTree(Source.fromURL(fileUrl).mkString)

    tree
  }
  /**
    * resolve configure file from local file system
    */
  def resolveConfigOuter(configFile: String): JsonNode = {
    val mapper = new ObjectMapper()
    val tree: JsonNode = mapper.readTree(Source.fromFile(configFile).mkString)
    tree
  }
  /**
    * resolve configure file from HDFS
    */
  def resolveConfigHDFS(spark:SparkSession, filePath: String): JsonNode = {
    //import com.fasterxml.jackson.databind.ObjectMapper
    import org.apache.hadoop.conf.Configuration
    import org.apache.hadoop.fs.{FileSystem, Path}

    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    val mapper = new ObjectMapper

    val root = mapper.readTree(fs.open(new Path(filePath)))

    root
  }

  /**
    * if the hdfs file exist
    */
  def checkHDFSFileExist(spark:SparkSession,modelPath: String): Boolean = {

    val HDFSFileSytem = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    HDFSFileSytem.exists(new Path(modelPath))
  }
  /**
    * delete the hdfs file
    */
  def deleteHDFSFile(spark:SparkSession,filePath:String,flag:Boolean = false): Unit ={
    import org.apache.hadoop.fs._

    val hdfs : FileSystem = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    hdfs.delete(new Path(filePath), flag)
  }


//  def checkHDFSDirExist(spark:SparkSession,dir: String): Boolean = {
//    import org.apache.hadoop.fs.FileSystem
//    import org.apache.hadoop.fs.Path
//    val HDFSFileSytem: FileSystem = FileSystem.get(spark.sparkContext.hadoopConfiguration)
//    val p = new Path(dir)
//    HDFSFileSytem.exists(p) && HDFSFileSytem.getFileStatus(p).isDirectory
//  }
}
