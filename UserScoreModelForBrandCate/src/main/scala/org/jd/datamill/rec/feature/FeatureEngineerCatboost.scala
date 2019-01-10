package org.jd.datamill.rec.feature

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.linalg.{Vector, Vectors}
import com.fasterxml.jackson.databind.JsonNode
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DoubleType, StringType}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import scala.collection.mutable

import org.jd.datamill.rec.Utils

object FeatureEngineerCatboost {
  /**
    * generate map for user_attr -> catboost_vector
    * table catboost_v5 : sex<replace>1, 0.2,0.3,0.4,...,0.04
    * @return Broadcast[mutable.Map[String, Map[String, Vector)))
    *        one from the map could be : sex->Map(1->Vector(0.2,0.3,0.4...), 2->Vector(0.1,0.2...), ...)
    */
    def getBCMap(spark:SparkSession,tree:JsonNode): Broadcast[mutable.Map[String, Map[String, Vector]]] ={


      val bcFeatureDF = spark.sql("select * from " +
        "(select b.* from " +
        s"(SELECT MAX(dt) as latest_dt FROM ${Utils.tableCatBoostV5BC} " +
        "WHERE dt IS NOT NULL) a " +
        "join " +
        s"${Utils.tableCatBoostV5BC} b " +
        "ON a.latest_dt=b.dt " +
        "WHERE b.dt IS NOT NULL " +
        ")mid " +
        s"where name_value is not null "
      )
        .drop("dt")
        .na.fill(0)

      import spark.implicits._
      //cols catBoosted: "0.2,0.4,0.8..." to vector,
      val dataCatBoosted = bcFeatureDF.rdd.map{
        row =>

          val nameValue = row.getAs[String](0).split(Utils.sepReplace)

          val buffer = ArrayBuffer.empty[Double]

          val len = row.length
          //13 catboost values
          for(i <- 1 until len){
            buffer.append(row.getDouble(i))
          }

          (nameValue(0),nameValue(1),Vectors.dense(buffer.toArray).compressed)
      }.toDF("attrName","attrValue","catboostVec")

      import org.apache.spark.ml.linalg.Vector

      val finalMap = mutable.Map.empty[String,Map[String,Vector]]

      val bcFeaturesIter = tree.get("bc_feature").iterator()
      var lenBCFeature = 0
      while(bcFeaturesIter.hasNext){
        val cur = bcFeaturesIter.next().toString.stripSuffix("\"").stripPrefix("\"")
        println(cur)
        lenBCFeature += 1

        val featureMap: Map[String, Vector] = dataCatBoosted
          .filter(s"attrName='${cur}'")
          .collect()
          .map(r=>(r.getString(1),r.getAs[Vector](2)))
          .map(seq => (seq._1,seq._2)).toMap

        finalMap.put(cur,featureMap)
      }

      //sex->Map(1->0.2, 2->0.4), ...
      val bcMap: Broadcast[mutable.Map[String, Map[String, Vector]]] = spark.sparkContext.broadcast(finalMap)

      bcMap
    }

  /**
    * user attr to vector(catboost values) based on BCMap (Above)
    * @param bcMap
    * @param nVec : length of catboost values in case null -> Vector(zeros(n))
    * @return
    */
  def categoryMapToCatboost(colName:String,
                            newColName:String,
                            bcMap:Broadcast[mutable.Map[String, Map[String, Vector]]],
                            nVec:Int): Column ={
    val oneFeatureMap: Map[String, Vector] = bcMap.value.getOrElse(colName,Map.empty)

    if(oneFeatureMap.isEmpty){
      println(s"error : catboost map empty for ${colName}")
      return lit(Vectors.zeros(nVec))
    }

    val f = udf { (attrValue: String) =>
      oneFeatureMap.getOrElse(attrValue,Vectors.zeros(nVec))
    }

    f(col(colName).cast(StringType)).as(newColName)
  }

  //final step : generate features
  def generateFeatures(spark: SparkSession, df: DataFrame, tree: JsonNode, jobType: String): DataFrame = {

    //final cols after process
    val colsAll = ArrayBuffer[Column]()

    /** 0. user brand cate ...*/
    if (jobType == "train") {
      colsAll.append(col("y_value"))
      colsAll.append(col("user_log_acct"))
      colsAll.append(col("main_brand_code"))
      colsAll.append(col("item_third_cate_cd"))
    } else if (jobType == "predict") {
      colsAll.append(col("user_log_acct"))
      colsAll.append(col("main_brand_code"))
      colsAll.append(col("item_third_cate_cd"))
    } else {
      println("Error feature generator type!")
      sys.exit(1)
    }

    /** 1. continue value, so put in directly */
    val orgFeature = tree.get("org_feature").iterator()
    //cols no need to process, put into directly
    while (orgFeature.hasNext) {
      val cur = orgFeature.next().asText()
      println(cur)
      colsAll.append(col(cur).cast(DoubleType))
    }
    val orgUserFeature = tree.get("org_user_feature").iterator()
    //cols no need to process, put into directly
    while (orgUserFeature.hasNext) {
      val cur = orgUserFeature.next().asText()
      println(cur)
      colsAll.append(col(cur).cast(DoubleType))
    }
    /** 2. ratio features */
    val colsRatioFeatures = calcRatioFeatures(tree,"ratio_feature")
    colsAll.appendAll(colsRatioFeatures)

    /** 3. user_cat_attr: onehot or indexed or setMapping */
    //only deal the user_cat_attr in the conf file
    val checkUserAttrArr = ArrayBuffer[String]()
    val userAttrFeature = tree.get("onehot_or_index_feature").iterator()
    while (userAttrFeature.hasNext) {
      val cur = userAttrFeature.next().asText()
      println(cur)
      checkUserAttrArr.append(cur)
    }
    val userCatDealedArr = userCatAttrFeatures(checkUserAttrArr.toArray)
    colsAll.appendAll(userCatDealedArr)

     /** 4. catboost 13 dimensions : bcMap */
    val bcMap = getBCMap(spark,tree)
    println(s"bcMap   size: ${bcMap.value.size}     ${bcMap.value}")
    val catFeature = tree.get("bc_feature").iterator()
    //cols catBoosted: "0.2,0.4,0.8..." to vector,
    while (catFeature.hasNext) {
      val colName = catFeature.next().asText()
      println(s"bc_feature : ${colName}")
      if(bcMap.value.contains(colName)){
        val colCatboosted = categoryMapToCatboost(colName,colName+"_catboost_vec",bcMap,13)
        colsAll.append(colCatboosted)
        println("bcMap contains "+colName)
      }
    }

    val resDF = df
      .select(colsAll.toArray: _*)
    resDF
  }

  /**
    * onehot or stringindex the features of user attr
    * @param checkFeatureArr
    * @return
    */
  def userCatAttrFeatures(checkFeatureArr:Array[String]): Array[Column] ={

    val colsArr = new ArrayBuffer[Column]()

    /** index or onehot determining by true or false */
    val ulevelCol = Utils.createCatMappingCol("user_active_level", "user_active_level_onehot", true,
      Array("注册会员", "企业会员", "铜牌会员", "银牌会员", "金牌会员", "钻石会员", "易迅会员", "VIP会员"))
    if (checkFeatureArr.contains("user_active_level")) {
      println("youyouyou")
      colsArr.append(ulevelCol)
    }

    val sexCol = Utils.createCatMappingCol("gender", "gender_onehot", true,
      Array("男", "女"))
    if (checkFeatureArr.contains("gender")) {
      colsArr.append(sexCol)
    }

    val ageCol = Utils.createCatMappingCol("age", "age_onehot", false,
      Array("15岁以下", "16-25岁", "26-35岁", "36-45岁", "46-55岁", "56岁以上"))
    if (checkFeatureArr.contains("age")) {
      colsArr.append(ageCol)
    }

    val marriageCol = Utils.createCatMappingCol("marriage_status", "marriage_status_onehot", true,
      Array("未婚", "已婚"))
    if (checkFeatureArr.contains("marriage_status")) {
      colsArr.append(marriageCol)
    }


    val educationCol = Utils.createCatMappingCol("user_education", "user_education_onehot", false,
      Array("1", "2", "3", "4"))
    if (checkFeatureArr.contains("user_education")) {
      colsArr.append(educationCol)
    }

    val professionCol = Utils.createCatMappingCol("user_profession", "user_profession_onehot", true,
      Array("a", "b", "c", "d", "e", "f", "g", "h"))
    if (checkFeatureArr.contains("user_profession")) {
      colsArr.append(professionCol)
    }

    val haschildCol = Utils.createCatMappingCol("has_child", "has_child_onehot", false,
      Array("有小孩"))
    if (checkFeatureArr.contains("has_child")) {
      colsArr.append(haschildCol)
    }


    val childageCol = Utils.createCatMappingCol("cpp_seni_childage", "cpp_seni_childage_onehot", true,
      Array("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"))
    if (checkFeatureArr.contains("cpp_seni_childage")) {
      colsArr.append(childageCol)
    }

    val hascarCol = Utils.createCatMappingCol("cpt_glob_cars", "cpt_glob_cars_onehot", false,
      Array("是"))
    if (checkFeatureArr.contains("cpt_glob_cars")) {
      colsArr.append(hascarCol)
    }


    val provinceCol = Utils.createCatMappingCol("cpp_base_regprovince", "cpp_base_regprovince_onehot", true,
      Array("广东", "云南", "内蒙古", "湖北", "新疆", "海南", "西藏", "陕西", "天津", "广西",
        "河南", "贵州", "江苏", "宁夏", "青海", "福建", "黑龙江", "辽宁", "重庆", "安徽",
        "山东", "湖南", "上海", "山西", "甘肃", "北京", "河北", "浙江", "江西", "四川", "吉林"))
    if (checkFeatureArr.contains("cpp_base_regprovince")) {
      colsArr.append(provinceCol)
    }

    val paytypeCol = Utils.createSetMappingCol("pay_mode_preference", "pay_mode_preference_onehot", "#",
      Array("00000a", "00000b", "00000c",
        "1", "10", "11", "12", "14",
        "2", "3", "5", "6", "68", "7", "70", "8"))
    if (checkFeatureArr.contains("pay_mode_preference")) {
      colsArr.append(paytypeCol)
    }


    val clientCol = Utils.createSetMappingCol("csf_sale_client", "csf_sale_client_onehot", "#",
      Array("app&m", "pc", "shouQ", "weixin", "其他"))
    if (checkFeatureArr.contains("csf_sale_client")) {
      colsArr.append(clientCol)
    }

    val firstordtmCol = Utils.createCatMappingCol("csf_saletm_first_ord_tm", "csf_saletm_first_ord_tm_onehot", false,
      Array("一个月", "两个月","三个月", "六个月", "半年及以上"))
    if (checkFeatureArr.contains("csf_saletm_first_ord_tm")) {
      colsArr.append(firstordtmCol)
    }

    val lastordtmCol = Utils.createCatMappingCol("last_ord_time", "last_ord_time_onehot", false,
      Array("一个月", "两个月", "三个月", "六个月", "半年及以上"))
    if (checkFeatureArr.contains("last_ord_time")) {
      colsArr.append(lastordtmCol)
    }

    val lastlogintmCol = Utils.createCatMappingCol("csf_saletm_last_login_tm", "csf_saletm_last_login_tm_onehot", false,
      Array("一个月", "两个月", "三个月", "六个月", "半年及以上"))
    if (checkFeatureArr.contains("csf_saletm_last_login_tm")) {
      colsArr.append(lastlogintmCol)
    }

    val mombabyCol = Utils.createCatMappingCol("muying_medal_level", "muying_medal_level_onehot", false,
      Array("V0", "V1", "V2", "V3", "V4"))
    if (checkFeatureArr.contains("muying_medal_level")) {
      colsArr.append(mombabyCol)
    }

    val beautyCol = Utils.createCatMappingCol("beauty_medal_level", "beauty_medal_level_onehot", false,
      Array("V0", "V1", "V2", "V3", "V4"))
    if (checkFeatureArr.contains("beauty_medal_level")) {
      colsArr.append(beautyCol)
    }

    val wineCol = Utils.createCatMappingCol("csf_medal_wine", "csf_medal_wine_onehot", false,
      Array("V0", "V1", "V2", "V3", "V4"))
    if (checkFeatureArr.contains("csf_medal_wine")) {
      colsArr.append(wineCol)
    }


    val purchpowerCol = Utils.createCatMappingCol("purchasing_power", "purchasing_power_onehot", false,
      Array("1", "2", "3", "4", "5"))
    if (checkFeatureArr.contains("purchasing_power")) {
      colsArr.append(purchpowerCol)
    }

    val lifecycleCol = Utils.createCatMappingCol("cgp_cycl_lifecycle", "cgp_cycl_lifecycle_onehot", false,
      Array("1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "11", "12", "13", "14", "15", "16"))
    if (checkFeatureArr.contains("cgp_cycl_lifecycle")) {
      colsArr.append(lifecycleCol)
    }


    val groupCol = Utils.createCatMappingCol("cvl_rfm_all_group", "cvl_rfm_all_group_onehot", true,
      Array("rfm1_重要价值客户", "rfm2_重要发展客户", "rfm3_重要保持客户", "rfm4_重要挽留客户",
        "rfm5_一般价值客户", "rfm6_一般发展客户", "rfm7_一般保持客户", "rfm8_一般挽留客户"))
    if (checkFeatureArr.contains("cvl_rfm_all_group")) {
      colsArr.append(groupCol)
    }

    val valuegrpCol = Utils.createCatMappingCol("cvl_glob_valuegrp", "cvl_glob_valuegrp_onehot", false,
      Array("价值低", "价值中", "价值高", "非常高"))
    if (checkFeatureArr.contains("cvl_glob_valuegrp")) {
      colsArr.append(valuegrpCol)
    }


    val loyaltyCol = Utils.createCatMappingCol("cvl_glob_loyalty", "cvl_glob_loyalty_onehot", true,
      Array("近期-偶然型", "近期-普通型", "近期-投机型", "中度-忠诚型",
        "远期-偶然型", "远期-普通型", "远期-投机型"))
    if (checkFeatureArr.contains("cvl_glob_loyalty")) {
      colsArr.append(loyaltyCol)
    }

    val promotionCol = Utils.createCatMappingCol("promotions_sensitive_level", "promotions_sensitive_level_onehot", false,
      Array("L1-1", "L1-2", "L1-3", "L1-4", "L1-5",
        "L2-1", "L2-2", "L2-3", "L2-4", "L2-5",
        "L3-1", "L3-2", "L3-3", "L3-4", "L3-5",
        "L4-1", "L4-2", "L4-3", "L4-4", "L4-5"))
    if (checkFeatureArr.contains("promotions_sensitive_level")) {
      colsArr.append(promotionCol)
    }

    val commentCol = Utils.createCatMappingCol("comment_sensitive_level", "comment_sensitive_level_onehot", false,
      Array("L1-1", "L1-2", "L1-3", "L1-4", "L1-5",
        "L2-1", "L2-2", "L2-3", "L2-4", "L2-5",
        "L3-1", "L3-2", "L3-3", "L3-4", "L3-5",
        "L4-1", "L4-2", "L4-3", "L4-4", "L4-5"))
    if (checkFeatureArr.contains("comment_sensitive_level")) {
      colsArr.append(commentCol)
    }

    //add pop features
    val cvlRfmPopSuperCol = Utils.createCatMappingCol("cvl_rfm_pop_super", "cvl_rfm_pop_super_onehot", true,
      Array("rfm1_重要价值客户", "rfm2_重要发展客户", "rfm3_重要保持客户", "rfm4_重要挽留客户",
        "rfm5_一般价值客户", "rfm6_一般发展客户", "rfm7_一般保持客户", "rfm8_一般挽留客户"))
    if (checkFeatureArr.contains("cvl_rfm_pop_super")) {
      colsArr.append(cvlRfmPopSuperCol)
    }

    val cvlRfmPopCol = Utils.createCatMappingCol("cvl_rfm_pop", "cvl_rfm_pop_onehot", true,
      Array("rfm1_重要价值客户", "rfm2_重要发展客户", "rfm3_重要保持客户", "rfm4_重要挽留客户",
        "rfm5_一般价值客户", "rfm6_一般发展客户", "rfm7_一般保持客户", "rfm8_一般挽留客户"))
    if (checkFeatureArr.contains("cvl_rfm_pop")) {
      colsArr.append(cvlRfmPopCol)
    }

    val csfSaleRebuyLastlyCol = Utils.createCatMappingCol("csf_sale_rebuy_lasty", "csf_sale_rebuy_lasty_onehot", true,
      Array("复购", "去年之前无购买", "去年无购买", "首次购"))
    if (checkFeatureArr.contains("csf_sale_rebuy_lasty")) {
      colsArr.append(csfSaleRebuyLastlyCol)
    }

    val csfSaleRebuyCol = Utils.createCatMappingCol("csf_sale_rebuy", "csf_sale_rebuy_onehot", true,
      Array("复购", "首次购", "今年无购买"))
    if (checkFeatureArr.contains("csf_sale_rebuy")) {
      colsArr.append(csfSaleRebuyCol)
    }

    colsArr.toArray

  }
  //calculate ratio features: numerators/denominators
  def calcRatioFeatures(tree:JsonNode,name:String):Array[Column] = {

    val arrBuf = new ArrayBuffer[Column]

    val it = tree.path(name).iterator()
    while (it.hasNext) {
      val node = it.next()
      //fenzi
      val numerators = node.path("numerators").iterator()
      var colNume: Column = (lit(0)).as("zero")
      var colNumeName = "numerator"
      while (numerators.hasNext) {
        val numerator = numerators.next().asText()
        colNumeName = colNumeName+"_"+numerator
        val f = udf { (add1: Double,add2: Double) =>
          add1 + add2
        }
        colNume = f(colNume.cast(DoubleType),col(numerator).cast(DoubleType))
      }
      //colNume = colNume.as(colNumeName)
      //fenmu
      val denominators = node.path("denominators").iterator()
      var colDeno = (lit(0)).as("zero")
      var colDenoName = "denominator"
      while (denominators.hasNext) {
        val denominator = denominators.next().asText()
        colDenoName = colDenoName+"_"+denominator
        val f = udf { (add1: Double,add2: Double) =>
          add1 + add2
        }
        colDeno = f(colDeno.cast(DoubleType),col(denominator).cast(DoubleType))
      }
      //colDeno = colDeno.as(colDenoName)

      val newName = colNumeName+"_devide_"+colDenoName
      //devide
      arrBuf.append(Utils.ratioColumn(colNume,colDeno,newName))
    }

    arrBuf.toArray
  }

}
