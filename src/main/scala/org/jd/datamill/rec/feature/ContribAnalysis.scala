package org.jd.datamill.rec.feature
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.Metadata
import scala.collection.mutable
import org.jd.datamill.rec.Utils

/**

  nohup spark-submit --class org.jd.datamill.rec.feature.ContribAnalysis \
--master yarn \
--deploy-mode client \
--num-executors 256 \
--executor-memory 4G \
--executor-cores 4 \
--driver-memory 8G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.memory.useLegacyMode=true \
--conf spark.default.parallelism=1024 \
--conf spark.storage.memoryFraction=0.2 \
--conf spark.shuffle.memoryFraction=0.8 \
--driver-java-options "-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps" \
--conf spark.network.timeout=1200s \
--conf spark.sql.shuffle.partitions=2048 \
--conf spark.sql.broadcastTimeout=1200 \
--jars /home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-spark-0.72.jar,/home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-0.72.jar \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_test_20181226_3dt_no_user_attr_1_5 \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/xgb_model_no_user_1_5/bestmodel_depth8_tree200 \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/xgb_model_no_user_1_5/importance_no_user_1_5 \
> logs/contribAnalysis_no_user_20190108_1_5.log 2>&1 &

  spark.scheduler.listenerbus.eventqueue.size 100000
  */

object ContribAnalysis {

  val nPartitionsLarge = 10240
  val nPartitionsMid = 2048

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("ContribAnalysis")
      .enableHiveSupport()
      .getOrCreate()

    val pathTest = args(0)
    val pathModel = args(1)
    val pathImportance = args(2)

    val dataTest = spark.read.parquet(pathTest)//.randomSplit(Array(0.2,0.8))(0)
    println(s"zwx data read done   dataTest.count : ${dataTest.count()}")

    /**get name of all features */
    val meta: org.apache.spark.sql.types.Metadata = dataTest
      .schema(dataTest.schema.fieldIndex("features"))
      .metadata
    val attrs: Metadata = meta.getMetadata("ml_attr").getMetadata("attrs")
    //index,name: 0,cpp_level
    val featIdxNames: Array[(Int, String)] = attrs.getMetadataArray("numeric") //binary??
      .map(meta => (meta.getLong("idx").toInt,meta.getString("name")))

    val xgbModel: XGBoostModel = XGBoost.loadModelFromHadoopFile(s"${pathModel}")(spark.sparkContext)

    /**importance of weight */
    //name,weight: f0,12
    val featureImportance: mutable.Map[String, Integer] = xgbModel.booster.getFeatureScore()
    //index,weight
    val fiIdxWeight: Map[Int, Int] = featureImportance.map{case (name,weight) =>(name.substring(1).toInt,weight.toInt)}.toMap
    //weight,name
    val fiWeightName: Array[(String, Int)] =  featIdxNames
      .map{case (idx,name) =>(fiIdxWeight.getOrElse(idx,0),name)}
      .sortBy(_._1)
      .map{case (weight,name) =>(name,weight)}
    println(s"zwx importance of weight  ${fiWeightName.mkString(";")}")
    spark.sparkContext
      .parallelize(fiWeightName.map{case (name,weight)=>name+","+weight}, 1)
      .saveAsTextFile(pathImportance)

    /**contributions of tree shap */
//    val myBooster: Broadcast[Booster] = spark.sparkContext.broadcast(xgbModel.booster)
//    val rddLabelPoint: RDD[LabeledPoint] = dataTest.rdd.map { row =>
//      val label = row.getInt(0)
//      val features = row.getAs[org.apache.spark.ml.linalg.Vector](1)
//      new LabeledPoint(label.toFloat,features.toSparse.indices,features.toSparse.values.map(d => d.toFloat))
//    }.repartition(nPartitionsLarge)
//    rddLabelPoint.persist(StorageLevel.MEMORY_AND_DISK)
//    println(s"zwx data rddLabelPoint.count : ${rddLabelPoint.count()}")
//
//    val contribAll: RDD[Array[Float]] = rddLabelPoint.mapPartitions { testData =>
//
//      //The feature contributions and bias.
//      val contrib: Array[Array[Float]] = myBooster.value.predictContrib(new DMatrix(testData))
//
//      contrib.iterator
//    }.repartition(nPartitionsMid)
//    contribAll.persist(StorageLevel.MEMORY_AND_DISK)
//    println(s"zwx data contribAll.count : ${contribAll.count()}")
//
//
//    val contribSum: Array[Float] = contribAll.reduce { (acc, vec) =>
//      acc zip vec map {
//        //global impact: sum of abs(shap values)
//        case (a, b) => Utils.abs(a) + Utils.abs(b)
//      }
//    }
//    println(s"zwx contribSum : ${contribSum.mkString(";")}")
//
//    //val (values, indices) = contribSum.zipWithIndex.sortBy(_._1).unzip
//    //val contribSorted: Array[(Float, Int)] = contribSum.zipWithIndex.sortBy(_._1)
//    //println(s"zwx contribSorted  ${contribSorted.mkString(";")}")
//
//    val contribSorted: Array[(String, Float)] = contribSum.zipWithIndex
//      .map{case (contrib,idx) =>(featIdxNames.toMap.getOrElse(idx,"bias"),contrib)}.sortBy(_._2)
//    println(s"zwx contribSorted  ${contribSorted.mkString(";")}")
//
//    spark.sparkContext.parallelize(contribSorted.map{case (name,weight)=>name+","+weight}, 1)
//      .saveAsTextFile(pathContribShap)

  }
}
