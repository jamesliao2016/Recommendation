package org.jd.datamill.rec.predict

import com.fasterxml.jackson.databind.JsonNode
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.jd.datamill.rec.Utils

/**
create external table if not exists tmp.wiwin_rec_user_brand_cate_result(
    user_log_acct string,
    brand_code string,
    item_third_cate_cd string,
    score double )
PARTITIONED BY (  dt string)
LOCATION
  'hdfs://ns1018/user/jd_ad/ads_polaris/tmp.db/wiwin_rec_user_brand_cate_result';

*/
/**

  export SPARK_HOME=/data0/spark/spark-2.1.0-bin-hadoop2.7.1-online-JDGPU-v1.2.0-201810091011-jdk1.7
  export  PYTHONPATH="${SPARK_HOME}/python/lib/py4j-0.10.6-src.zip:$PYTHONPATH"

  nohup spark-submit --class org.jd.wiwin.rec.sort.XgboostPre \
--master yarn --deploy-mode client \
--executor-memory 4G \
--num-executors 256 \
--executor-cores 4 \
--driver-memory 4G \
--conf spark.memory.useLegacyMode=true \
--conf spark.sql.shuffle.partitions=1024 \
--conf spark.default.parallelism=1024 \
--conf spark.shuffle.memoryFraction=0.3 \
--conf spark.network.timeout=1200s \
--jars /home/bi_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-spark-0.72.jar,/home/bi_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-0.72.jar \
jars/rec4-2.0.jar \
hdfs://ns3/user/mart_sz/bi_polaris/zhangwenxiang6/predict_brand_cate/wd_v4/feature_predict_20181219 \
hdfs://ns3/user/mart_sz/bi_polaris/zhangwenxiang6/predict_brand_cate/wd_v4/model_xgb_20181213/bestmodel_depth8_tree200 \
2018-12-19 \
> logs/rec_predict_20181219.log 2>&1 &

  */

object XgboostPre {
  def predict(spark: SparkSession, predTree:JsonNode, featurePath: String, modelPath: String): Unit = {
    import spark.implicits._

    val tableOut = predTree.get("outputTable").asText()
    val dt = predTree.get("dt").asText()

    val featureDF = spark.read.parquet(featurePath)

    val xgbClassificationModel = XGBoost.loadModelFromHadoopFile(modelPath)(spark.sparkContext)

    val predictDF = xgbClassificationModel.transform(featureDF)

    //split probability
    //val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
    //val toArrUdf = udf(toArr)
    val Element2: Any =>Double = _.asInstanceOf[DenseVector](1)
    val toElement2 = udf(Element2)

    val predictResult = predictDF
      .select("user_log_acct","main_brand_code","item_third_cate_cd","probabilities")
      .withColumn("score",toElement2(col("probabilities")))
      .select("user_log_acct","main_brand_code","item_third_cate_cd","score")

//    val predictDFWithProbSplit = predictDFTmp
//      .select("user_log_acct","main_brand_code","item_third_cate_cd","prediction","probability_arr")
//      .withColumn("probPos",col(""))
//      .rdd.map{
//      line =>
//
//        val user = line.getString(0)
//        val brand = line.getString(1)
//        val cate = line.getString(2)
//
//        val prediction = line.getDouble(3)
//
//        val probability = line.getSeq(4).toArray[Double]
//        val probNeg = probability(0)
//        val probPos = probability(1)
//
//        (user,brand,cate, prediction, probNeg,probPos)
//    }.toDF("user_log_acct","main_brand_code","item_third_cate_cd","prediction","probNeg","probPos")
//    predictDFWithProbSplit.show()


    predictResult.createOrReplaceTempView("result")

    spark.sqlContext.setConf("mapred.compress.map.output","false")
    spark.sqlContext.setConf("hive.merge.mapfiles","true")
    spark.sqlContext.setConf("hive.merge.mapredfiles","true")
    spark.sqlContext.setConf("hive.merge.size.per.task","256000000")
    spark.sqlContext.setConf("hive.merge.smallfiles.avgsize","200000000")
    spark.sqlContext.setConf("hive.exec.dynamic.partition","true")
    spark.sqlContext.setConf("hive.exec.dynamic.partition.mode","nonstrict")

    spark.sql(s"insert overwrite table ${tableOut} partition(dt) " +
      s" select *, '${dt}' from result")

  }
}

