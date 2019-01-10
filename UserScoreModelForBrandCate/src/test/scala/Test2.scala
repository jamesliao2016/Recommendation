import org.apache.spark.sql.SparkSession

object Test2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Test2")
      .enableHiveSupport()
      .getOrCreate()

    val path = "hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/feature_catboost_predict_20181231"
    val feature = spark.read.parquet(path)


    val ord = spark.sql("" +
      "select user_log_acct, main_brand_cd as main_brand_code, item_third_cate_cd, label from " +
      "(select user_log_acct, brand_cd as brand_code, item_third_cate_cd, 1 as label " +
      "from gdm.gdm_m04_ord_det_sum " +
      "where dt>='2019-01-01' and sale_ord_dt>='2019-01-01' and sale_ord_dt<='2019-01-03' " +
      "group by user_log_acct, brand_cd, item_third_cate_cd )ord " +
      "join " +
      "(select brand_cd, main_brand_cd from app.dim_dm_main_brand where dt='2018-12-31')main_brand " +
      "on ord.brand_code=main_brand.brand_cd ")

    val pred = feature.join(ord,Array("user_log_acct","main_brand_code","item_third_cate_cd"),"left_outer").na.fill(0)

    pred.write.mode("overwrite").parquet(path+"_with_label")

  }
}
