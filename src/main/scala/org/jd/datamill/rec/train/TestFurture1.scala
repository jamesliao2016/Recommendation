package org.jd.datamill.rec.train

import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

/**


  nohup spark-submit --class org.jd.datamill.rec.train.TestFurture1 \
--master yarn \
--deploy-mode client \
--num-executors 256 \
--executor-memory 4G \
--executor-cores 4 \
--driver-memory 4G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.memory.useLegacyMode=true \
--conf spark.default.parallelism=4096 \
--conf spark.storage.memoryFraction=0.2 \
--conf spark.shuffle.memoryFraction=0.8 \
--driver-java-options "-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps" \
--conf spark.network.timeout=1200s \
--conf spark.sql.shuffle.partitions=4096 \
--conf spark.sql.broadcastTimeout=1200 \
  --conf spark.executor.memoryoverhead=4096 \
--conf spark.yarn.executor.memoryoverhead=4096 \
--queue root.bdp_jmart_ad.jd_ad_dev \
--jars /home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-spark-0.72.jar,/home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-0.72.jar \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_train_20190107_1dt_several_brandcate \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_test_20190107_1dt_several_brandcate \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/xgb_model_sample345/bestmodel_depth8_tree200 \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/xgb_model_sample2_1_10/bestmodel_depth8_tree200 \
7820 \
"('1_2','2')" \
> logs/rec_testFurture1_model2_1_10_several_bc_7820_1.log 2>&1 &


  */
object TestFurture1 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("TestFurture")
      .enableHiveSupport()
      .getOrCreate()

    val pathData = Array(args(0),args(1))
    val pathModel1 = args(2)
    val pathModel2 = args(3)

    val brand = args(4)
    val sampleTypes = args(5).stripSuffix("\"").stripPrefix("\"")

    println(s"zwx ${sampleTypes}")
    validate(spark,pathModel1,pathModel2,pathData,brand,sampleTypes)
  }

  def validate(spark:SparkSession,
               modelPath1:String,modelPath2:String,
               dataPath:Array[String],
               brand:String,
               sampleTypes:String): Unit ={
    val dataOrigin = spark.read.parquet(dataPath:_*)

    val data = if(brand=="no"){
      dataOrigin
        .filter(s"sample_type in ${sampleTypes}")
        .drop("sample_type").distinct()
    }else{
      dataOrigin
        .filter(s"main_brand_code='${brand}'")
        .filter(s"sample_type in ${sampleTypes}")
        .drop("sample_type").distinct()
    }


    val xgbModel1: XGBoostModel = XGBoost.loadModelFromHadoopFile(s"${modelPath1}")(spark.sparkContext)
    val predictDF1 = xgbModel1.transform(data)

    val xgbModel2: XGBoostModel = XGBoost.loadModelFromHadoopFile(s"${modelPath2}")(spark.sparkContext)
    val predictDF2 = xgbModel2.transform(data)

    import spark.implicits._
    import org.apache.spark.sql.types._
    //split probability
    val Element2: Any =>Double = _.asInstanceOf[DenseVector](1)
    val toElement2 = udf(Element2)

    val predict1 = predictDF1
      .select("user_log_acct","main_brand_code","item_third_cate_cd","probabilities","label")
      .withColumn("probPos1",toElement2(col("probabilities")))
      .select("user_log_acct","main_brand_code","item_third_cate_cd","probPos1","label")
    val predictAndLabel1: RDD[(Double, Double)] = predict1.select($"probPos1".cast(DoubleType), $"label".cast(DoubleType))
      .map { case Row(prediction: Double, label: Double) =>
        (prediction, label)
      }.rdd
    matrixRes(predictAndLabel1,"model1")

    val predict2 = predictDF2
      .select("user_log_acct","main_brand_code","item_third_cate_cd","probabilities","label")
      .withColumn("probPos2",toElement2(col("probabilities")))
      .select("user_log_acct","main_brand_code","item_third_cate_cd","probPos2","label")
    val predictAndLabel2: RDD[(Double, Double)] = predict2.select($"probPos2".cast(DoubleType), $"label".cast(DoubleType))
      .map { case Row(prediction: Double, label: Double) =>
        (prediction, label)
      }.rdd
    matrixRes(predictAndLabel2,"model2")

    val predictMean = predict1.join(predict2,Array("user_log_acct","main_brand_code","item_third_cate_cd","label"))
      .withColumn("probPosMean",(col("probPos1")+col("probPos2"))/2)

    println(s"predict1.count : ${predict1.count()}    predictAll.count : ${predictMean.count}")

    val predictAndLabelMean: RDD[(Double, Double)] = predictMean.select($"probPosMean".cast(DoubleType), $"label".cast(DoubleType))
      .map { case Row(prediction: Double, label: Double) =>
        (prediction, label)
      }.rdd
    matrixRes(predictAndLabelMean,"2Mean")
  }

  def matrixRes(predictAndLabel:RDD[(Double, Double)],flag:String): Unit ={
    val metrics = new BinaryClassificationMetrics(predictAndLabel)
    val auROC = metrics.areaUnderROC
    val auPR = metrics.areaUnderPR()

    println(s"${flag} result matrix   " + " auc: " + s"$auROC"  +"     auPR: " + s"$auPR")

  }
}
