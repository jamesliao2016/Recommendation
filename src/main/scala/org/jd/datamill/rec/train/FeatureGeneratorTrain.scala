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
--executor-memory 8G \
--num-executors 500 \
--executor-cores 4 \
--driver-memory 4G \
  --conf spark.dynamicAllocation.enabled=false \
--conf spark.memory.useLegacyMode=true \
--conf spark.sql.shuffle.partitions=4096 \
--conf spark.default.parallelism=4096 \
--conf spark.shuffle.memoryFraction=0.2 \
  --conf spark.shuffle.memoryFraction=0.8 \
--conf spark.network.timeout=1200s \
  --conf spark.sql.codegen.wholeStage=false \
  --queue bdp_jmart_ad.bdp_jmart_ad_docker2 \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
2018-12-29 12 \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_train_20181229_4dt \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_test_20181229_4dt \
no \
"('1','1_2','1_3','1_4','1_5','2','3','4','5')" \
> logs/userScoreModel_feature_train_20181229_4dt.log 2>&1 &


no user all sample_type
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_train_20190107_1dt_all_type \
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_test_20190107_1dt_all_type \

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
    val sampleTypes = args(5).stripSuffix("\"").stripPrefix("\"")

    generate(spark, endDate, dateDiff, tree, trainPath,testPath,flagDaily,sampleTypes)

  }

  def generate(spark: SparkSession,
               endDate: String, dateDiff: Int,
               tree: JsonNode,
               trainDFPath: String,testDFPath: String,
               flagDaily:String,
               sampleTypes:String): Unit = {

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

    val orgTrainDf = spark.read.parquet(fileNameDailyArr.toArray:_*)//.repartition(2048)
      .filter(s"sample_type in ${sampleTypes}")
      .randomSplit(Array(0.3,0.7),12345)(0)
      //distinct data because of sample_type
      .distinct()
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
      .select("sample_type","y_value", "user_log_acct", "main_brand_code", "item_third_cate_cd", "features")
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
