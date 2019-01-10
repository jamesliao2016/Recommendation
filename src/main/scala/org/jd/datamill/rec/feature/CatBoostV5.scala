package org.jd.datamill.rec.feature

import com.fasterxml.jackson.databind.JsonNode
import org.apache.spark.sql.{Column, Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer
import org.jd.datamill.rec.Utils

/**

create external table tmp.wiwin_rec_user_feature_pv_follow_cart_search_catboost_v5(
  name_value string,
  ord_pv_usr_1_7 double,
  ord_pv_usr_8_30 double,
  ord_follow_usr_1_7 double,
  ord_follow_usr_8_30 double,
  ord_follow_usr_31_90 double,
  ord_follow_usr_91_730 double,
  ord_cart_usr_1_7 double,
  ord_cart_usr_8_30 double,
  ord_cart_usr_31_90 double,
  ord_cart_usr_91_730 double,
  ord_search_usr_1_7 double,
  ord_search_click_usr_1_7 double,
  percentage_users double
)partitioned by (dt string)
LOCATION
  'hdfs://ns1018/user/jd_ad/ads_polaris/tmp.db/wiwin_rec_user_feature_pv_follow_cart_search_catboost_v5';


create external table tmp.wiwin_rec_user_feature_pv_follow_cart_search_catboost_v5_bc(
  name_value string,
  ord_pv_usr_1_7 double,
  ord_pv_usr_8_30 double,
  ord_follow_usr_1_7 double,
  ord_follow_usr_8_30 double,
  ord_follow_usr_31_90 double,
  ord_follow_usr_91_730 double,
  ord_cart_usr_1_7 double,
  ord_cart_usr_8_30 double,
  ord_cart_usr_31_90 double,
  ord_cart_usr_91_730 double,
  ord_search_usr_1_7 double,
  ord_search_click_usr_1_7 double,
  percentage_users double
)partitioned by (dt string)
LOCATION
  'hdfs://ns1018/user/jd_ad/ads_polaris/tmp.db/wiwin_rec_user_feature_pv_follow_cart_search_catboost_v5_bc';


  */

/**
  export SPARK_HOME=/data0/spark/spark-2.3.0-bin-hadoop2.7.1-online-JD2.3.0.1-201806291504
  export  PYTHONPATH="${SPARK_HOME}/python/lib/py4j-0.10.6-src.zip:$PYTHONPATH"

  nohup spark-submit --class org.jd.datamill.rec.feature.CatBoostV5 \
--master yarn --deploy-mode client \
--executor-memory 16G \
--num-executors 256 \
--executor-cores 4 \
--driver-memory 4G \
--conf spark.memory.useLegacyMode=true \
--conf spark.sql.shuffle.partitions=10240 \
--conf spark.default.parallelism=10240 \
--conf spark.shuffle.memoryFraction=0.2 \
  --conf spark.shuffle.memoryFraction=0.8 \
--conf spark.network.timeout=1200s \
  --queue bdp_jmart_ad.bdp_jmart_ad_docker2 \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
  2019-01-01 \
> logs/rec_CatBoost5_20190103.log 2>&1 &

    --queue bdp_jmart_ad.bdp_jmart_ad_docker2 \

  */

/**
  * sum(order)/sum(pv)... :  value>0 then 1
  * follow reverse: follow/order
  * add one column: users/ all users
  */

object CatBoostV5 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Test")
      .enableHiveSupport()
      .getOrCreate()

    val configFile = "catboostConfig.json"
    val tree: JsonNode = Utils.resolveConfig(configFile)

    val dt = args(0).toString

    catBoosted(spark,dt,tree)
  }

  def catBoosted(spark:SparkSession,dt:String,tree:JsonNode): Unit ={


    val userFeature = spark.sql("select * from tmp.wiwin_rec_feature_catboost_user_attr_tmp_zwx " +
      s" where dt in ('2018-12-20','2018-12-30') " +
      s"and rand() <= 0.2 " +
      s"and user_log_acct is not null and user_log_acct!='' and user_log_acct!='_' and user_log_acct!='nobody' ")
      .na.fill(-100)
      .na.fill("-100")


    import org.apache.spark.sql.functions._
    val addUDF = udf{(ord1To3:Int, ord4To7:Int) => ord1To3+ord4To7}

    val data = userFeature
      .withColumn("pv_usr_1_7",addUDF(col("pv_usr_1_3"),col("pv_usr_4_7")))
      .withColumn("pv_usr_8_30",addUDF(col("pv_usr_8_15"),col("pv_usr_16_30")))
      .withColumn("sale_qtty_usr_91_730",addUDF(col("sale_qtty_usr_91_180"),col("sale_qtty_usr_181_730")))
      .withColumn("search_usr_1_7",addUDF(col("search_usr_1_3"),col("search_usr_4_7")))
      .withColumn("search_click_usr_1_7",addUDF(col("search_click_usr_1_3"),col("search_click_usr_4_7")))

    println(s"zwx data.count : ${data.count()}")

    val countUsers = data.select("user_log_acct").distinct().count().toDouble

    val catFeature = tree.get("cat_feature").iterator()
    var colsCatBoostedArr = ArrayBuffer[String]()
    //cols to catBoost
    while (catFeature.hasNext) {
      val colName = catFeature.next().toString.stripSuffix("\"").stripPrefix("\"")
      colsCatBoostedArr.append(colName)
    }
    val colsCatBoosted = colsCatBoostedArr.toArray

    val colsNameCalc =
      Array(
        "sale_qtty_usr_1_7",
        "sale_qtty_usr_8_30",
        "sale_qtty_usr_31_90",
        "sale_qtty_usr_91_730",
        "pv_usr_1_7",
        "pv_usr_8_30",
        "follow_usr_1_7",
        "follow_usr_8_30",
        "follow_usr_31_90",
        "follow_usr_91_730",
        "cart_usr_1_7",
        "cart_usr_8_30",
        "cart_usr_31_90",
        "cart_usr_91_730",
        "search_usr_1_7",
        "search_click_usr_1_7"
      )

    //replace value with colName+value
    val dataSelected = data.select(
      colsCatBoosted.map(colName => col(colName).cast(StringType)) ++
        colsNameCalc.map(name=>col(name)) ++
        Array(col("user_log_acct")):_*
    ).rdd.map{
      row =>
        val buffer = ArrayBuffer.empty[Any]
        // Add value to buffer
        var i = 0
        for(colName <- colsCatBoosted){
          buffer.append(colName+Utils.sepReplace+row.getAs[String](i))
          i = i+1
        }
        for(colName <- colsNameCalc){
          buffer.append(row.getAs[Int](i))
          i = i+1
        }
        buffer.append(row.getAs[String](i))
        // Build row
        Row.fromSeq(buffer)

    }
    val schema = StructType(
      colsCatBoosted.map(colName => StructField(colName, StringType, false))++
        colsNameCalc.map(colName => StructField(colName, IntegerType, false)) ++
        Array(StructField("user_log_acct", StringType, false))
    )

    val dataReplaced =  spark.sqlContext.createDataFrame(dataSelected, schema)
    dataReplaced.persist(StorageLevel.MEMORY_AND_DISK)
    dataReplaced.show()
    //one row to multiple rows
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.functions.explode
    import org.apache.spark.sql.functions.array
    val arrBufTmp = ArrayBuffer[Column]()
    arrBufTmp.append(col("user_log_acct"))
    arrBufTmp.appendAll(colsNameCalc.map(name => col(name)))
    arrBufTmp.append(explode(array(colsCatBoosted.map(colName => col(colName).cast(StringType)):_*)).as("key"))
    //ord_usr_1_3|ord_usr_4_7|ord_usr_1_7|pv_usr_1_3|pv_usr_4_7|pv_usr_1_7|follow_usr_1_3|follow_usr_4_7|
    //  follow_usr_1_7|cart_usr_1_3|cart_usr_4_7|cart_u      sr_1_7|search_usr_1_3|search_usr_4_7|search_usr_1_7|
    //  key|
    val dataExplode = dataReplaced.select(arrBufTmp.toArray:_*)
    dataExplode.persist(StorageLevel.MEMORY_AND_DISK)
    dataExplode.show()
    println(s"zwx dataExplode.count : ${dataExplode.count()}")
    //calculate
    val dataGrouped = dataExplode
      .groupBy("key")
      .agg(
        (sum(when(col("sale_qtty_usr_1_7")>0 , 1))/(sum(when(col("pv_usr_1_7")>0 , 1))+1)).as("ord_pv_usr_1_7"),
        (sum(when(col("sale_qtty_usr_8_30")>0 , 1))/(sum(when(col("pv_usr_8_30")>0 , 1))+1)).as("ord_pv_usr_8_30"),

        (sum(when(col("follow_usr_1_7")>0 , 1))/(sum(when(col("sale_qtty_usr_1_7")>0 , 1))+1)).as("ord_follow_usr_1_7"),
        (sum(when(col("follow_usr_8_30")>0 , 1))/(sum(when(col("sale_qtty_usr_8_30")>0 , 1))+1)).as("ord_follow_usr_8_30"),
        (sum(when(col("follow_usr_31_90")>0 , 1))/(sum(when(col("sale_qtty_usr_31_90")>0 , 1))+1)).as("ord_follow_usr_31_90"),
        (sum(when(col("follow_usr_91_730")>0 , 1))/(sum(when(col("sale_qtty_usr_91_730")>0 , 1))+1)).as("ord_follow_usr_91_730"),

        (sum(when(col("sale_qtty_usr_1_7")>0 , 1))/(sum(when(col("cart_usr_1_7")>0 , 1))+1)).as("ord_cart_usr_1_7"),
        (sum(when(col("sale_qtty_usr_8_30")>0 , 1))/(sum(when(col("cart_usr_8_30")>0 , 1))+1)).as("ord_cart_usr_8_30"),
        (sum(when(col("sale_qtty_usr_31_90")>0 , 1))/(sum(when(col("cart_usr_31_90")>0 , 1))+1)).as("ord_cart_usr_31_90"),
        (sum(when(col("sale_qtty_usr_91_730")>0 , 1))/(sum(when(col("cart_usr_91_730")>0 , 1))+1)).as("ord_cart_usr_91_730"),

        (sum(when(col("sale_qtty_usr_1_7")>0 , 1))/(sum(when(col("search_usr_1_7")>0 , 1))+1)).as("ord_search_usr_1_7"),
        (sum(when(col("sale_qtty_usr_1_7")>0 , 1))/(sum(when(col("search_click_usr_1_7")>0 , 1))+1)).as("ord_search_click_usr_1_7"),

        (countDistinct("user_log_acct")/countUsers).as("percentage_users")
      )
    //dataGrouped.show()
    dataGrouped.createOrReplaceTempView("dataCatBoosted")

    spark.sqlContext.setConf("mapred.compress.map.output","false")
    spark.sqlContext.setConf("hive.merge.mapfiles","true")
    spark.sqlContext.setConf("hive.merge.mapredfiles","true")
    spark.sqlContext.setConf("hive.merge.size.per.task","256000000")
    spark.sqlContext.setConf("hive.merge.smallfiles.avgsize","200000000")
    spark.sqlContext.setConf("hive.exec.dynamic.partition","true")
    spark.sqlContext.setConf("hive.exec.dynamic.partition.mode","nonstrict")

    spark.sql(s"insert overwrite table ${Utils.tableCatBoostV5} partition(dt) " +
      s" select *, '${dt}' from dataCatBoosted")

    /** make sure it is a small table */
    spark.sql(s"insert overwrite table ${Utils.tableCatBoostV5BC} partition(dt) " +
      s"select * from ${Utils.tableCatBoostV5} " +
      s"where dt='${dt}' " +
      s"and name_value not like '%cate2%' " +
      s"and name_value not like '%cate3%' " +
      s"and name_value not like '%top2%' " +
      s"and name_value not like '%top3%' " +
      s"and name_value not like '%brand%' ")
  }
}
