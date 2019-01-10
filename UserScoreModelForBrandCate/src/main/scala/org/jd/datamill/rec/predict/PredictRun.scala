package org.jd.datamill.rec.predict

import org.apache.spark.sql.SparkSession
import org.jd.datamill.rec.Utils


/**

  export SPARK_HOME=/data0/spark/spark-2.1.0-bin-hadoop2.7.1-online-JDGPU-v1.2.0-201810091011-jdk1.7
  export  PYTHONPATH="${SPARK_HOME}/python/lib/py4j-0.10.6-src.zip:$PYTHONPATH"

  nohup spark-submit --class org.jd.datamill.rec.predict.PredictRun \
--master yarn --deploy-mode client \
--executor-memory 16G \
--num-executors 256 \
--executor-cores 4 \
--driver-memory 8G \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.sql.broadcastTimeout=1200 \
--conf spark.memory.useLegacyMode=true \
--conf spark.sql.shuffle.partitions=4096 \
--conf spark.default.parallelism=4096 \
--conf spark.storage.memoryFraction=0.2 \
--conf spark.shuffle.memoryFraction=0.8 \
--conf spark.network.timeout=1200s \
--queue root.bdp_jmart_ad.jd_ad_dev \
  --jars /home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-spark-0.72.jar,/home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-0.72.jar \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/predictConf.json \
> logs/userScoreModel_predict_all_catboost_20190109_2.log 2>&1 &


--queue bdp_jmart_ad.bdp_jmart_ad_docker2 \
结果：
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/feature_predict_20181231
  hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/feature_catboost_predict_20181231
  */

object PredictRun {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("PredictRun")
      .enableHiveSupport()
      .getOrCreate()

    //conf inside : feature generator
    val confFeature = "featureConfig.json"
    val treeFeature = Utils.resolveConfig(confFeature)

    /** conf provided : predict conf */
    val predConfFilePath = args(0)
    val treePred = Utils.resolveConfigHDFS(spark,predConfFilePath)

    //date for table(act and attr)
    val dt = Utils.getDateBeforeNDays(2)

    //tmp file : feature file
    val featurePath = treePred.get("wd").asText()+s"feature_predict-${System.nanoTime}"

    // generate feature
    FeatureGeneratorPredict.generate(spark, dt, treeFeature,treePred,featurePath)


    val modelPath = treePred.get("modelPath").asText()+ treePred.get("modelVersion").asText()
    if(!Utils.checkHDFSFileExist(spark,modelPath)){
      println("ERROR : model path does not exist")
      sys.exit(1)
    }
    XgboostPre.predict(spark,treePred,featurePath,modelPath)

    //delete tmp file: feature
    //Utils.deleteHDFSFile(spark,featurePath)
  }


}
