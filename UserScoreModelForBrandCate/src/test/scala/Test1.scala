import org.apache.spark.sql.SparkSession

object Test1 {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Test")
      .enableHiveSupport()
      .getOrCreate()

    val arr = Array("655","672","12201","870","16756")
    val path = "hdfs://ns1018/user/ads_polaris/zhangwenxiang6/user_score_model/wd_v1/"

    val bc = spark.sql("select * from tmp.zwx_tmp_user_brand_cate")

    import org.apache.spark.sql.functions._
    //var i=0
    for(cate3 <- arr){
      val tmp = bc.filter(col("item_third_cate_cd")===cate3).select("user_log_acct").distinct()
      tmp.write.mode("overwrite").parquet(path+"users_"+cate3)
      //i = i+1
    }

    import org.apache.hadoop.fs._
    import org.apache.hadoop.conf.Configuration

    val hdfs : FileSystem = FileSystem.get(new Configuration)
    hdfs.delete(new Path("hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/ttt"), false)

  }
}
