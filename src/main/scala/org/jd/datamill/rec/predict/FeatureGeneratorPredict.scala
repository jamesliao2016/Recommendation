package org.jd.datamill.rec.predict

import com.fasterxml.jackson.databind.JsonNode
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.jd.datamill.rec.Utils
import org.jd.datamill.rec.feature.{DataDailyPredict, FeatureEngineer, FeatureEngineerCatboost, FeatureEngineerNoUserAttr}

import scala.collection.mutable.ArrayBuffer

/**

  export SPARK_HOME=/data0/spark/spark-2.1.0-bin-hadoop2.7.1-online-JDGPU-v1.2.0-201810091011-jdk1.7
  export  PYTHONPATH="${SPARK_HOME}/python/lib/py4j-0.10.6-src.zip:$PYTHONPATH"

  nohup spark-submit --class org.jd.datamill.rec.predict.FeatureGeneratorPredict \
--master yarn --deploy-mode client \
--executor-memory 16G \
--num-executors 256 \
--executor-cores 4 \
--driver-memory 8G \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.sql.broadcastTimeout=1200 \
--conf spark.memory.useLegacyMode=true \
--conf spark.sql.shuffle.partitions=8000 \
--conf spark.default.parallelism=8000 \
--conf spark.storage.memoryFraction=0.2 \
--conf spark.shuffle.memoryFraction=0.8 \
--conf spark.network.timeout=1200s \
  --queue bdp_jmart_ad.bdp_jmart_ad_docker2 \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/predictConf.json \
2018-12-31 \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/feature_no_user_predict_20181231 \
> logs/userScoreModel_feature_predict_no_user_20181231.log 2>&1 &

--queue bdp_jmart_ad.bdp_jmart_ad_docker2 \
结果：
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/feature_predict_20181231
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/feature_catboost_predict_20181231
  */

object FeatureGeneratorPredict {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("FeatureGeneratorPredict")
      .enableHiveSupport()
      .getOrCreate()

    val confFeature = "featureConfig.json"
    val treeFeature = Utils.resolveConfig(confFeature)

    val predConfFilePath = args(0)
    val dt = args(1)
    val resDFPath = args(2)

    val treePred = Utils.resolveConfigHDFS(spark,predConfFilePath)

    generate(spark, dt, treeFeature,treePred,resDFPath)

  }
  def generate(spark: SparkSession,
               dt: String,
               treeFeature: JsonNode,treePred:JsonNode, predictDFPath: String
               ): Unit = {


    val wd = treePred.get("wd").asText()

    val pathDataDaily = wd+"feature_predict_"+dt
    val taskResArr = DataDailyPredict.dataJoin(spark,treePred,treeFeature,dt,pathDataDaily)
    for(task <- taskResArr){
      println(s"task status : ${task.toJson()}")
    }
    println(s" DataDailyPredict done: ${dt}")

    val orgPredictDf = spark.read.parquet(pathDataDaily)
      .na.fill(0).na.fill("0")

    val transDF = FeatureEngineerNoUserAttr.generateFeatures(spark, orgPredictDf, treeFeature, "predict")
      .na.fill(0).na.fill("0")

    val allCol = transDF.columns
    println(s"feature columns : ${allCol.length}    ${allCol.mkString(",")}")
    println(s"feature columns :  ${allCol.zipWithIndex.mkString(";")}")

    val embeddingCols = ArrayBuffer[String]()
    allCol.foreach { col =>
      if (col != "y_value" & col != "user_log_acct" & col != "main_brand_code" & col != "item_third_cate_cd") {
        embeddingCols.append(col)
      }
    }

    /** assemble */
    val assembler = new VectorAssembler()
      .setInputCols(embeddingCols.toArray)
      .setOutputCol("features")

    val output = assembler.transform(transDF)
      .select("user_log_acct", "main_brand_code", "item_third_cate_cd", "features")

    output.write.mode("overwrite").parquet(predictDFPath)
  }

}
