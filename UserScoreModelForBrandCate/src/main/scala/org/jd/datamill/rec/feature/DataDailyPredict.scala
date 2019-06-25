package org.jd.datamill.rec.feature

import com.fasterxml.jackson.databind.JsonNode
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import org.jd.datamill.rec.Utils
import org.jd.datamill.rec.predict.TaskResult

import scala.collection.mutable.ArrayBuffer

object DataDailyPredict {

  val nRepartition = 10240

  /**
    * only the user attrs we need:
    * @param treeFeature
    * @return string
    */
  def sqlUserAttr(treeFeature:JsonNode): String ={
    val arr = ArrayBuffer[String]()

    val itOnehot = treeFeature.path("onehot_or_index_feature").iterator()
    while (itOnehot.hasNext) {
      val tmp = itOnehot.next().asText()
      arr.append(tmp)
    }
    val itOrg = treeFeature.path("org_user_feature").iterator()
    while (itOrg.hasNext) {
      val tmp = itOrg.next().asText()
      arr.append(tmp)
    }

    val itCatboost = treeFeature.path("bc_feature").iterator()
    while (itCatboost.hasNext) {
      val tmp = itCatboost.next().asText()
      arr.append(tmp)
    }

    var sql = " "
    for(tmp <- arr.distinct.toArray){
      sql = sql + tmp + ", "
    }

    sql = sql.trim().stripSuffix(",").stripPrefix(",")

    sql
  }

  /**
    * where brand and cate in sql based on predict conf
    * @param treePred
    * @return
    */
  def sqlWhereBrandCate(treePred:JsonNode): (String,String,String) ={
    //reduce data size base on main_brand_code and item_third_cate_cd
    var brandcateSql = ""
    var brandSql = ""
    var cateSql = ""
    val it = treePred.path("targetSets").iterator()
    while (it.hasNext) {
      val node = it.next()
      val brand = node.get("brand_code").asText()
      val cate3 = node.get("cate3").asText()
      brandcateSql = brandcateSql + " or (main_brand_code='"+brand+"' and item_third_cate_cd='"+cate3+"')"
      brandSql = brandSql + " or (main_brand_code='"+brand+"' )"
      cateSql = cateSql + " or ( item_third_cate_cd='"+cate3+"')"

    }
    brandcateSql = brandcateSql.trim().stripSuffix("or").stripPrefix("or")
    brandSql = brandSql.trim().stripSuffix("or").stripPrefix("or")
    cateSql = cateSql.trim().stripSuffix("or").stripPrefix("or")
    println(s"main_brand_cate and item_third_cate_cd sql : ${brandcateSql}  |   ${brandSql}   |  ${cateSql}")

    (brandcateSql,brandSql,cateSql)
  }
  def dataJoin(spark:SparkSession,
               treePred:JsonNode,
               treeFeature:JsonNode,
               dt:String,
               pathDataDaily:String
              ): Array[TaskResult] ={

    println("INFO dataJoin in")

    val tmpDir = treePred.get("wd").asText()

    val taskResArr = new ArrayBuffer[TaskResult]()

    //table user is one day late in the ETL
    val dtUser = Utils.getDateMinusDays(dt, 1).toString

    val sampleArr = ArrayBuffer[DataFrame]()

    //1.sample
    val it = treePred.path("targetSets").iterator()
    while (it.hasNext) {

      val node = it.next()
      val brand = node.get("brand_code").asText()
      val cate3 = node.get("cate3").asText()
      val path = node.get("path").asText()

      if(Utils.checkHDFSFileExist(spark,path)){
        //get sample from conf file
        val df = spark.read.parquet(path)
          .select("user_log_acct")
          .dropDuplicates()
          .na.drop()
          .withColumn("main_brand_code",lit(brand))
          .withColumn("item_third_cate_cd",lit(cate3))
        sampleArr.append(df)

        val taskRes = new TaskResult(node.get("id").asText(),brand,cate3,path,0,"")
        taskResArr.append(taskRes)
      }else{
        //error file path
        val taskRes = new TaskResult(node.get("id").asText(),brand,cate3,path,1,"userpin path does not exist")
        taskResArr.append(taskRes)
      }

    }
    if(sampleArr.size<=0){
      println("ERROR sampleArr.size=0 : all userpin file is invalid or no task to process")
      sys.exit(1)
    }
    val sample = sampleArr.toArray.reduce(_ union _)
    //sample.persist(StorageLevel.MEMORY_AND_DISK)
    //println(s"dataJoin sample.count : ${sample.count()}")

    //2.act
    val (brandcateSql,brandSql,cateSql) = sqlWhereBrandCate(treePred)
    val actUBC = spark.sql(s"select * " +
      s"from  ${Utils.tableActUBC} where dt='${dt}' " +
      s"and user_log_acct is not null and item_third_cate_cd is not null and main_brand_code is not null " +
      s"and ( ${brandcateSql} ) "+
      s"and user_log_acct is not null and user_log_acct !='' and user_log_acct !='_' and user_log_acct !='-' " +
      s"and user_log_acct !='nobody' and user_log_acct !='NULL' ")
      .drop("dt")
    val actUB = spark.sql(s"select * " +
      s"from  ${Utils.tableActUB} where dt='${dt}' " +
      s"and user_log_acct is not null and main_brand_code is not null " +
      s"and ( ${brandSql} ) "+
      s"and user_log_acct is not null and user_log_acct !='' and user_log_acct !='_' and user_log_acct !='-' " +
      s"and user_log_acct !='nobody' and user_log_acct !='NULL' ")
      .drop("dt")
    val actUC = spark.sql(s"select * " +
      s"from  ${Utils.tableActUC} where dt='${dt}' " +
      s"and user_log_acct is not null and item_third_cate_cd is not null " +
      s"and ( ${cateSql} ) "+
      s"and user_log_acct is not null and user_log_acct !='' and user_log_acct !='_' and user_log_acct !='-' " +
      s"and user_log_acct !='nobody' and user_log_acct !='NULL' ")
      .drop("dt")

    //3.item
    val item = spark.sql(s"select * from ${Utils.tableItem} where dt='${dt}' " +
      s"and ( ${brandcateSql} )" +
      s" and item_third_cate_cd is not null and main_brand_code is not null")
      .drop("dt")
      .drop("item_second_cate_cd")

    //5.user attr
    val userAttrSql = sqlUserAttr(treeFeature)
    val users = spark.sql(s"select user_log_acct, ${userAttrSql} " +
      s" from ${Utils.tableUser} where dt='${dtUser}' " +
      s"and user_log_acct is not null and user_log_acct !='' and user_log_acct !='_' and user_log_acct !='-' " +
      s"and user_log_acct !='nobody' and user_log_acct !='NULL' ")
      .drop("dt")
      .join(sample.select("user_log_acct").dropDuplicates(),Array("user_log_acct"))
    users.persist(StorageLevel.MEMORY_AND_DISK)
    println(s"dataJoin : users.count  ${users.count()}")

    val mergedMid = sample
      .join(broadcast(item),Array("item_third_cate_cd","main_brand_code"),"left_outer")
      .join(actUBC,Array("user_log_acct","item_third_cate_cd","main_brand_code"),"left_outer")
      .join(actUB,Array("user_log_acct","main_brand_code"),"left_outer")
      .join(actUC,Array("user_log_acct","item_third_cate_cd"),"left_outer")
    mergedMid.persist(StorageLevel.MEMORY_AND_DISK)
    println(s" data join mergedMid.count : ${mergedMid.count()}")

    val merged = mergedMid
      .join(users,Array("user_log_acct"),"left_outer")
      .drop("item_second_cate_cd")

    merged.write.mode("overwrite").parquet(pathDataDaily)
    println("data join done")

    mergedMid.unpersist()
    users.unpersist()

    taskResArr.toArray
  }

}
