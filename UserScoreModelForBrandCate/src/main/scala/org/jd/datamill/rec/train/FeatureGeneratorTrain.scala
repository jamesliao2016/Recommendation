package org.jd.datamill.rec.train

import com.fasterxml.jackson.databind.JsonNode
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.jd.datamill.rec.Utils
import org.jd.datamill.rec.feature.{DataDaily, FeatureEngineerNoUserAttr}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**

  export SPARK_HOME=/data0/spark/spark-2.1.0-bin-hadoop2.7.1-online-JDGPU-v1.2.0-201810091011-jdk1.7
  export  PYTHONPATH="${SPARK_HOME}/python/lib/py4j-0.10.6-src.zip:$PYTHONPATH"

  nohup spark-submit --class org.jd.datamill.rec.train.FeatureGeneratorTrain \
--master yarn --deploy-mode client \
--executor-memory 16G \
--num-executors 256 \
--executor-cores 4 \
--driver-memory 8G \
  --conf spark.dynamicAllocation.enabled=false \
--conf spark.memory.useLegacyMode=true \
--conf spark.sql.shuffle.partitions=8000 \
--conf spark.default.parallelism=8000 \
--conf spark.shuffle.memoryFraction=0.2 \
  --conf spark.shuffle.memoryFraction=0.8 \
--conf spark.network.timeout=1200s \
  --conf spark.sql.codegen.wholeStage=false \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
2018-12-26 7 \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_train_20181226_3dt_no_user_attr_ratio \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_test_20181226_3dt_no_user_attr_ratio \
no \
> logs/userScoreModel_feature_train_20190109_26_3dt_no_user_attr_ratio.log 2>&1 &

以下是未来某天(30号)的数据，用于test效果：pos:0.4亿，neg:3.4亿，1:8.5
注意：实际是catboost :
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_train_20181230_1dt \
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_test_20181230_1dt \

无catboost
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_train_catboost_20181230_1dt \
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_test_catboost_20181230_1dt \

no user
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_train_20181230_1dt_no_user_attr \
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_test_20181230_1dt_no_user_attr \
no user even ratio
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_train_20181230_1dt_no_user_attr_ratio \
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_test_20181230_1dt_no_user_attr_ratio \

  --queue bdp_jmart_ad.bdp_jmart_ad_docker2 \
  */

object FeatureGeneratorTrain {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("FeatureGeneratorTrain")
      .enableHiveSupport()
      .getOrCreate()

    val configFile = "featureConfig.json"
    val tree = Utils.resolveConfig(configFile)
    val iter = tree.fieldNames()
    while (iter.hasNext) {
      println(iter.next())
    }
    val wd = tree.get("wd").asText()
    println(s"zwx  wd  :  ${wd}")

    val endDate = args(0)
    val dateDiff = args(1).toInt
    val trainPath = args(2)
    val testPath = args(3)
    val flagDaily = args(4)

    generate(spark, endDate, dateDiff, tree, trainPath,testPath,flagDaily)

  }

  def generate(spark: SparkSession,
               endDate: String, dateDiff: Int,
               tree: JsonNode,
               trainDFPath: String,testDFPath: String,
               flagDaily:String): Unit = {

    val trainDateArr = Utils.getTrainDateArr(endDate, dateDiff, 3)

    val wd = tree.get("wd").asText()
    val fileNameDailyArr = mutable.ArrayBuffer.empty[String]
    for(dateStr <- trainDateArr){
      val pathDataDaily = wd+"train_daily_feature_all_"+dateStr
      //run daily or not
      if(flagDaily == "yes"){
        DataDaily.dataJoin(spark,dateStr,pathDataDaily)
      }
      println(s" loop each date : ${dateStr}")
      fileNameDailyArr.append(pathDataDaily)
    }

    val orgTrainDf = spark.read.parquet(fileNameDailyArr.toArray:_*)
      //distinct data because of sample_type
      .drop("sample_type").distinct()
      .na.fill(0).na.fill("0")

    orgTrainDf.groupBy("y_value").count().show()

    val transDF = FeatureEngineerNoUserAttr.generateFeatures(spark, orgTrainDf, tree, "train")
      .na.fill(0).na.fill("0")
    transDF.persist(StorageLevel.MEMORY_AND_DISK)
    println(s"transDF.count : ${transDF.count()}")

    val allCol = transDF.columns
    println(s"feature columns : ${allCol.length}    ${allCol.mkString(",")}")
    println(s"feature columns :  ${allCol.zipWithIndex.mkString(";")}")

    val embeddingCols = ArrayBuffer[String]()
    allCol.foreach { col =>
      if (col != "y_value" & col != "sample_type" & col != "user_log_acct" & col != "main_brand_code" & col != "item_third_cate_cd") {
        embeddingCols.append(col)
      }
    }

    /** assemble */
    val assembler = new VectorAssembler()
      .setInputCols(embeddingCols.toArray)
      .setOutputCol("features")

    val outputAssemble = assembler.transform(transDF)
      .select("y_value", "user_log_acct", "main_brand_code", "item_third_cate_cd", "features")
      .withColumnRenamed("y_value", "label")
    outputAssemble.persist(StorageLevel.MEMORY_AND_DISK)

    println(s"sample  positive : ${outputAssemble.filter("label=1").count()}  " +
      s"and negative : ${outputAssemble.filter("label=0").count()}")

    //split to train and test
    val Array(x_train, x_test) = outputAssemble.randomSplit(Array(0.7, 0.3), seed = 1673419217)

    println(s"data size   x_train: ${x_train.count()}     x_test: ${x_test.count()}")

    x_train.write.mode("overwrite").parquet(trainDFPath)
    x_test.write.mode("overwrite").parquet(testDFPath)

  }


}
