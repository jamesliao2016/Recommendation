package org.jd.datamill.rec.feature

import org.apache.spark.sql._
import org.apache.spark.storage.StorageLevel
import org.jd.datamill.rec.Utils

object DataDaily {
  val nRepartition = 10240
  /**
    *
    */
  //正负样本数据：user_log_acct, item_third_cate_cd, main_brand_code, y_value
  def getCandidate(spark:SparkSession, dt:String):DataFrame ={

    //train : pos<=0.5; neg<=0.05
    //valid : pos=all; neg <=0.5
    val sample = spark.sql(
      s"select * from ${Utils.tableSampleActAll} " +
        s"where dt='${dt}' and sample_type='1' " +
        s"and user_log_acct is not null and user_log_acct!='' and user_log_acct!='-' and user_log_acct!='nobody' " +
        s"and main_brand_code is not null and item_third_cate_cd is not null " +
        s" " +
        s"union all " +
        s"select * from ${Utils.tableSampleActAll} " +
        s"where dt='${dt}' and sample_type!='1' " +
        s"and user_log_acct is not null and user_log_acct!='' and user_log_acct!='-' and user_log_acct!='nobody' " +
        s"and main_brand_code is not null and item_third_cate_cd is not null " +
        s"and rand() <= 0.05 " +
        s"")
      //.drop("sample_type")
      .drop("dt")

    sample
  }


  def dataJoin(spark:SparkSession, dtCandidate:String,
               pathDataDaily:String
               ): Unit ={
    val dtFeature = dtCandidate
    val dtUser = Utils.getDateMinusDays(dtFeature, 1).toString

    println("zwx dataJoin in")

    //sample_act
    val sampleActAll = getCandidate(spark,dtCandidate)
    sampleActAll.persist(StorageLevel.MEMORY_AND_DISK)
    println(s" sampleActAll.count : ${sampleActAll.count()}")

    //item
    val item = spark.sql(s"select * from ${Utils.tableItem} where dt='${dtFeature}' " +
      s"and item_third_cate_cd is not null and main_brand_code is not null")
      .drop("item_second_cate_cd")
      .drop("dt")

    val userAttr = spark.sql(s"select * from ${Utils.tableUsersJoinedSamp} where dt='${dtUser}'")
      .drop("dt")

    //val userCatboost = spark.sql(s"select * from ${Utils.tableTrainUserCatBoostCollected} where dt='${dtUser}'")
    //  .drop("dt")

    //cols join
    val colsJoinItem = Array("item_third_cate_cd","main_brand_code")
    val colsJoinUser = Array("user_log_acct")

    val data = sampleActAll
      .join(item,colsJoinItem,"left_outer")
    //  .join(userCatboost,colsJoinUser,"left_outer")
      .join(userAttr,colsJoinUser,"left_outer")

    data.repartition(64).write.mode("overwrite").parquet(pathDataDaily)

    println(s"date  dtCandidate: ${dtCandidate}    data.count :  ${data.count()}")

  }

}
