package org.jd.datamill.rec.train

import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

/**

  nohup spark-submit --class org.jd.datamill.rec.train.Test \
--master yarn \
--deploy-mode client \
--num-executors 256 \
--executor-memory 8G \
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
--queue bdp_jmart_ad.bdp_jmart_ad_docker2 \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
> logs/rec_train_test_20190107.log 2>&1 &

  --queue bdp_jmart_ad.bdp_jmart_ad_docker2 \
  --conf spark.storage.memoryFraction=0.4 \

  */

object Test {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("TestTrain")
      .enableHiveSupport()
      .getOrCreate()

    val act = spark.sql("select user_log_acct, main_brand_code, item_third_cate_cd from app.wiwin_rec_user_brand_cate_act_merge " +
      "where dt='2018-12-31' " +
      "and (" +
      "(main_brand_code='8557' and item_third_cate_cd='655') or " +
      "(main_brand_code='8740' and item_third_cate_cd='672') or " +
      "(main_brand_code='19306' and item_third_cate_cd='12201') or " +
      "(main_brand_code='3659' and item_third_cate_cd='870') or " +
      "(main_brand_code='7820' and item_third_cate_cd='16756') " +
      ")")
    act.persist(StorageLevel.MEMORY_AND_DISK)
    act.count()

    val pathPred = "hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/feature_predict_20181231_with_label"
    val pred = spark.read.parquet(pathPred)

    val predAct = pred.join(act,Array("user_log_acct", "main_brand_code", "item_third_cate_cd"))

    //predAct.persist(StorageLevel.MEMORY_AND_DISK)
    predAct.count()

    predAct.write.mode("overwrite").parquet(pathPred+"_act_bc")

  }
}
