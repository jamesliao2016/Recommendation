package org.jd.datamill.rec.feature

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import scala.collection.mutable.ArrayBuffer

/**

  nohup spark-submit --class org.jd.datamill.rec.feature.Test \
--master yarn --deploy-mode client \
--executor-memory 8G \
--num-executors 256 \
--executor-cores 4 \
--driver-memory 4G \
--conf spark.memory.useLegacyMode=true \
--conf spark.sql.shuffle.partitions=4096 \
--conf spark.default.parallelism=4096 \
--conf spark.shuffle.memoryFraction=0.2 \
  --conf spark.shuffle.memoryFraction=0.8 \
--conf spark.network.timeout=1200s \
  --queue bdp_jmart_ad.bdp_jmart_ad_docker2 \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
> logs/rec_feature_test_20190104.log 2>&1 &

    --queue bdp_jmart_ad.bdp_jmart_ad_docker2 \

  */

object Test {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("TestFeature")
      .enableHiveSupport()
      .getOrCreate()

    val sample = spark.sql(
      s"select * from app.wiwin_rec_user_brand_cate_sample_act_merge_d " +
        s"where dt='2018-12-26' and sample_type='1' " +
        s"and user_log_acct is not null and user_log_acct!='' and user_log_acct!='-' and user_log_acct!='nobody' " +
        s"and main_brand_code is not null and item_third_cate_cd is not null " +
        s" " +
        s"union all " +
        s"select * from app.wiwin_rec_user_brand_cate_sample_act_merge_d " +
        s"where dt='2018-12-26' and sample_type!='1' " +
        s"and user_log_acct is not null and user_log_acct!='' and user_log_acct!='-' and user_log_acct!='nobody' " +
        s"and main_brand_code is not null and item_third_cate_cd is not null " +
        s"and rand() <= 0.1 ").drop("dt")

    val allCol = sample.columns

    val embeddingCols = ArrayBuffer[String]()
    allCol.foreach { col =>
      if (col != "y_value" & col != "sample_type" & col != "user_log_acct" & col != "main_brand_code" & col != "item_third_cate_cd") {
        embeddingCols.append(col)
      }
    }

    /** assemble */
    val assembler = new VectorAssembler().setInputCols(embeddingCols.toArray).setOutputCol("features")

    import spark.implicits._
    val outputAssemble = assembler.transform(sample)
    outputAssemble.persist(StorageLevel.MEMORY_AND_DISK)
    println(s" outputAssemble.count : ${outputAssemble.count()}")
    outputAssemble.groupBy("sample_type").count().show()

    import org.apache.spark.sql.functions._
    import org.apache.spark.ml.linalg.{Vector, Vectors}
    val lenUdf = udf((x: Vector) => x.numNonzeros)

    val filtered = outputAssemble.filter(lenUdf($"features") > 0)
    println(s" filtered.count : ${filtered.count()}")
    filtered.groupBy("sample_type").count().show()


  }
}
