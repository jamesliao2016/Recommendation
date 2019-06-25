import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel


/**
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
2018-12-30 1 \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_train_20181230_1dt_no_user_attr_ratio \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/data_test_20181230_1dt_no_user_attr_ratio \
no \
> logs/userScoreModel_feature_train_20190109_30_1dt_no_user_attr_ratio.log 2>&1 &

  */
object Test {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Test")
      .enableHiveSupport()
      .getOrCreate()

    import org.apache.spark.sql.functions._

    val path = "hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_train_20181229_4dt_sample2_1_2_distinct"
    val df = spark.read.parquet(path).filter("label<=1")
    val pos = df.filter("label>0")
    val neg = df.filter("label<=0")

    val posReduced = pos.randomSplit(Array(0.2,0.8),12345)(0)
    //val negReduced = neg.randomSplit(Array(0.4,0.6),12345)(1)

    val res = posReduced.union(neg)

    res.groupBy("label").count.show()

    res.write.parquet(path+"_1_10")

  }
  def getRandom (dataset : DataFrame, n : Int) = {
    val count = dataset.count()
    val howManyTake = if (count > n) n else count
    dataset.sample(false,1.0*howManyTake/count).limit(n).toDF("user_log_acct")
  }
  def getRandomExpensive(dataset:DataFrame,n:Int): DataFrame ={
    import org.apache.spark.sql.functions.rand
    dataset.orderBy(rand()).limit(n)
  }
  def tmp(spark:SparkSession): Unit ={

    //155852
    val pathAndroidImp = "hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/tmp/android_300000_1234567.csv"
    val dataAndroidImp = spark.read.text(pathAndroidImp).toDF("user_log_acct")
    //155899
    val pathIosImp = "hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/tmp/ios_200000_0123456.csv"
    val dataIosImp = spark.read.text(pathIosImp).toDF("user_log_acct").except(dataAndroidImp)


    //173553
    val pathAndroidOther = "hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/tmp/android_other_300000_3456789.csv"
    val dataAndroidOther = spark.read.text(pathAndroidOther).toDF("user_log_acct")
    //50381
    val pathIosOther = "hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/tmp/ios_other_200000_2345678.csv"
    val dataIosOther = spark.read.text(pathIosOther).toDF("user_log_acct").except(dataAndroidOther)


    //sample
    val dataIosImpSampled = getRandomExpensive(dataIosImp,50381)

    val dataAndroidOtherSampled = getRandomExpensive(dataAndroidOther,155852)

    dataIosImpSampled.count()
    dataAndroidImp.count()

    dataIosOther.count()
    dataAndroidOtherSampled.count()


    dataIosImpSampled.write.mode("overwrite").parquet("hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/tmp/imp_ios")
    dataAndroidImp.write.mode("overwrite").parquet("hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/tmp/imp_android")

    dataIosOther.write.mode("overwrite").parquet("hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/tmp/other_ios")
    dataAndroidOtherSampled.write.mode("overwrite").parquet("hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/tmp/other_android")

  }
  def tt(spark:SparkSession): Unit ={

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

    predAct.persist(StorageLevel.MEMORY_AND_DISK)
    predAct.count()

    predAct.write.mode("overwrite").parquet(pathPred+"_act_bc")

  }
}
