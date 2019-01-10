package org.jd.datamill.rec.train

import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

/**

  export SPARK_HOME=/data0/spark/spark-2.1.0-bin-hadoop2.7.1-online-JDGPU-v1.2.0-201810091011-jdk1.7
  export  PYTHONPATH="${SPARK_HOME}/python/lib/py4j-0.10.6-src.zip:$PYTHONPATH"

  nohup spark-submit --class org.jd.datamill.rec.train.TestFurtureReal \
--master yarn \
--deploy-mode client \
--num-executors 500 \
--executor-memory 16G \
--executor-cores 4 \
--driver-memory 8G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.memory.useLegacyMode=true \
--conf spark.default.parallelism=8000 \
--conf spark.storage.memoryFraction=0.2 \
--conf spark.shuffle.memoryFraction=0.8 \
--driver-java-options "-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps" \
--conf spark.network.timeout=1200s \
--conf spark.sql.shuffle.partitions=8000 \
--conf spark.sql.broadcastTimeout=1200 \
--queue root.bdp_jmart_ad.jd_ad_dev \
--jars /home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-spark-0.72.jar,/home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-0.72.jar \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/feature_predict_20181231_with_label_distinct \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/xgb_model1_1_10/bestmodel_depth8_tree200 \
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/predict_result_test \
> logs/rec_TestFurtureReal_test_predict_model_p1_n10_distinct.log 2>&1 &

29dt test most no act sample 2: hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/xgb_model_1/bestmodel_depth8_tree200
  */

object TestFurtureReal {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("TestFurture")
      .enableHiveSupport()
      .getOrCreate()

    val pathData = Array(args(0))
    val pathModel = args(1)
    val pathRes = args(2)

    validate(spark,pathModel,pathData,pathRes)
  }

  def validate(spark:SparkSession,modelPath:String,dataPath:Array[String],resPath: String): Unit ={
    val data = spark.read.parquet(dataPath:_*)

    val xgbModel: XGBoostModel = XGBoost.loadModelFromHadoopFile(s"${modelPath}")(spark.sparkContext)
    val predictDF = xgbModel.transform(data)

    //predictDF.write.mode("overwrite").parquet(resPath)
    predictDF.show()
    predictDF.groupBy("label","prediction").count().show()

    //split probability
    val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
    val toArrUdf = udf(toArr)
    val predictDFTmp = predictDF.withColumn("probability_arr",toArrUdf(col("probabilities")))

    import spark.implicits._
    val predictDFWithProbSplit = predictDFTmp.repartition(10240)
      .select("label","prediction","probability_arr")
      .rdd.map{
      line =>

        val label = line.getInt(0)
        val prediction = line.getDouble(1)

        val probability = line.getSeq(2).toArray[Double]
        val probNeg = probability(0)
        val probPos = probability(1)
        //val mkStr = probability.mkString(",")

        (label, prediction, probNeg,probPos)
    }.toDF("label","prediction","probNeg","probPos")
    predictDFWithProbSplit.show()

    import org.apache.spark.sql.types._
    val predictAndLabel = predictDFWithProbSplit.select($"probPos".cast(DoubleType), $"label".cast(DoubleType))
      .map { case Row(prediction: Double, label: Double) =>
        (prediction, label)
      }.rdd

    val metrics = new BinaryClassificationMetrics(predictAndLabel)
    val auROC = metrics.areaUnderROC
    val auPR = metrics.areaUnderPR()

    println(s"result matrix   " + " auc: " + s"$auROC"  +"     auPR: " + s"$auPR")

  }
}
