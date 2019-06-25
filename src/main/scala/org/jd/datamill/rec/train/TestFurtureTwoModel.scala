package org.jd.datamill.rec.train

import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
/**

export SPARK_HOME=/data0/spark/spark-2.1.0-bin-hadoop2.7.1-online-JDGPU-v1.2.0-201810091011-jdk1.7
  export  PYTHONPATH="${SPARK_HOME}/python/lib/py4j-0.10.6-src.zip:$PYTHONPATH"

  nohup spark-submit --class org.jd.datamill.rec.train.TestFurtureTwoModel \
--master yarn \
--deploy-mode client \
--num-executors 400 \
--executor-memory 16G \
--executor-cores 4 \
--driver-memory 4G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.memory.useLegacyMode=true \
--conf spark.default.parallelism=8000 \
--conf spark.storage.memoryFraction=0.2 \
--conf spark.shuffle.memoryFraction=0.8 \
--driver-java-options "-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps" \
--conf spark.network.timeout=1200s \
--conf spark.sql.shuffle.partitions=8000 \
--conf spark.sql.broadcastTimeout=1200 \
  --conf spark.executor.memoryoverhead=4096 \
--conf spark.yarn.executor.memoryoverhead=4096 \
--queue root.bdp_jmart_ad.jd_ad_dev \
--jars /home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-spark-0.72.jar,/home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-0.72.jar \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_train_20190107_1dt_no_user_sample5_all \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_test_20190107_1dt_no_user_sample5_all \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/xgb_model_no_user_ratio_1_2/bestmodel_depth8_tree200 \
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/xgb_model_no_user_sample2_1_2/bestmodel_depth8_tree200 \
> logs/rec_testFurture_predict_2model_pre_data_sample5_all.log 2>&1 &


  */
object TestFurtureTwoModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("TestFurture")
      .enableHiveSupport()
      .getOrCreate()

    val pathData = Array(args(0),args(1))
    val pathModel1 = args(2)
    val pathModel2 = args(3)

    validate(spark,pathModel1,pathModel2,pathData)
  }

  def validate(spark:SparkSession,modelPath1:String,modelPath2:String,dataPath:Array[String]): Unit ={
    val data = spark.read.parquet(dataPath:_*).drop("sample_type").distinct()

    val xgbModel1: XGBoostModel = XGBoost.loadModelFromHadoopFile(s"${modelPath1}")(spark.sparkContext)
    val predictDF1 = xgbModel1.transform(data)

    val xgbModel2: XGBoostModel = XGBoost.loadModelFromHadoopFile(s"${modelPath2}")(spark.sparkContext)
    val predictDF2 = xgbModel2.transform(data)


    //split probability
    val Element2: Any =>Double = _.asInstanceOf[DenseVector](1)
    val toElement2 = udf(Element2)

    val predict1 = predictDF1
      .select("user_log_acct","main_brand_code","item_third_cate_cd","probabilities","label")
      .withColumn("probPos1",toElement2(col("probabilities")))
      .select("user_log_acct","main_brand_code","item_third_cate_cd","probPos1","label")

    val predict2 = predictDF2
      .select("user_log_acct","main_brand_code","item_third_cate_cd","probabilities","label")
      .withColumn("probPos2",toElement2(col("probabilities")))
      .select("user_log_acct","main_brand_code","item_third_cate_cd","probPos2","label")

    val predictAll = predict1.join(predict2,Array("user_log_acct","main_brand_code","item_third_cate_cd","label"))
      .withColumn("probPosMean",(col("probPos1")+col("probPos2"))/2)

    println(s"predict1.count : ${predict1.count()}    predictAll.count : ${predictAll.count}")
    import spark.implicits._
    import org.apache.spark.sql.types._
    val predictAndLabel = predictAll.select($"probPosMean".cast(DoubleType), $"label".cast(DoubleType))
      .map { case Row(prediction: Double, label: Double) =>
        (prediction, label)
      }.rdd

    val metrics = new BinaryClassificationMetrics(predictAndLabel)
    val auROC = metrics.areaUnderROC
    val auPR = metrics.areaUnderPR()

    println(s"result matrix   " + " auc: " + s"$auROC"  +"     auPR: " + s"$auPR")

  }
}
