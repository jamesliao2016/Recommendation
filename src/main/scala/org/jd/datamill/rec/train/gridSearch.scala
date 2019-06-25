package org.jd.datamill.rec.train

import java.net.URL

import scala.io.Source
import scala.collection.mutable.{ArrayBuffer, Map}
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import ml.dmlc.xgboost4j.scala.spark.XGBoostModel
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.storage.StorageLevel


/**

  export SPARK_HOME=/data0/spark/spark-2.1.0-bin-hadoop2.7.1-online-JDGPU-v1.2.0-201810091011-jdk1.7
  export  PYTHONPATH="${SPARK_HOME}/python/lib/py4j-0.10.6-src.zip:$PYTHONPATH"

  nohup spark-submit --class org.jd.datamill.rec.train.gridSearch \
--master yarn \
--deploy-mode client \
--num-executors 256 \
--executor-memory 8G \
--executor-cores 8 \
--driver-memory 4G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.memory.useLegacyMode=true \
--conf spark.default.parallelism=2048 \
--conf spark.storage.memoryFraction=0.2 \
--conf spark.shuffle.memoryFraction=0.8 \
--driver-java-options "-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps" \
--conf spark.network.timeout=1200s \
--conf spark.sql.shuffle.partitions=2048 \
--conf spark.sql.broadcastTimeout=1200 \
    --conf spark.task.cpus=8 \
--queue root.bdp_jmart_ad.jd_ad_dev \
--jars /home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-spark-0.72.jar,/home/ads_polaris/zhangwenxiang6/tools/jars_xgb_072/xgboost4j-0.72.jar \
jars/UserScoreModelForBrandCate-1.0-SNAPSHOT.jar \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_train_20181229_4dt_reduced \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/data_test_20181229_4dt \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/xgb_model_all_reduced/ \
hdfs://ns1018/user/jd_ad/ads_polaris/zhangwenxiang6/user_score_model/wd_v2/xgb_model_all_reduced/bestmodel_depth8_tree200 \
> logs/rec_gridSearch_all_reduced_20190220.log 2>&1 &

  --conf spark.task.cpus=4 \

  --conf spark.executor.memoryoverhead=4096 \
--conf spark.yarn.executor.memoryoverhead=4096 \
  --queue bdp_jmart_ad.bdp_jmart_ad_docker2 \
  --conf spark.storage.memoryFraction=0.4 \

  */


object gridSearch {
  val configFile = "xgboostParam.json"

  val numCoalesce = 256

  def resolveConfig(configFile: String): JsonNode = {

    val fileUrl: URL = this.getClass.getClassLoader.getResource(configFile)

    val mapper = new ObjectMapper()
    val tree: JsonNode = mapper.readTree(Source.fromURL(fileUrl).mkString)

    tree
  }

  def paramGridList(): (ArrayBuffer[String], ArrayBuffer[ArrayBuffer[Any]]) = {
    val intBoost = Array("num_workers", "seed","num_class")
    val doubleBoost = Array("eta","gamma", "alpha", "lambda", "min_child_weight", "max_delta_step","subsample", "colsample_bytree")
    val strArray = Array("boosterType", "tree_method","objective")
    val intArray = Array("max_depth", "max_leaf_nodes", "num_round")

    //"num_class": [
    //2
    //]

    //"objective": [
    //"binary:logistic"
    //]

    val tree = resolveConfig(configFile)
    //val tree = resolveConfig("/Users/zhangwenxiang1/work/workspaces/IdeaProjects/rec_brand_cate_xgboost0.72_dataframe/src/main/resources/xgboostParam.json")
    val iterTree = tree.fieldNames()

    //check if all params in json are valid
    var isValid = 1
    var vectorLen = 0
    var existParams = new ArrayBuffer[String]()
    while (iterTree.hasNext && isValid == 1) {
      isValid = 0
      vectorLen += 1
      val matchStr = iterTree.next()
      existParams.append(matchStr)
      val param = tree.get(matchStr)
      if (!intBoost.find({
        x: String => x == matchStr
      }).isEmpty) {
        isValid = 1
      } else if (!doubleBoost.find({
        x: String => x == matchStr
      }).isEmpty) {
        isValid = 1
      } else if (!strArray.find({
        x: String => x == matchStr
      }).isEmpty) {
        isValid = 1
      } else if (!intArray.find({
        x: String => x == matchStr
      }).isEmpty) {
        isValid = 1
      } else {
        throw new RuntimeException("The params of xgboost4j spark is not valid.")
      }
    }

    //construct grid search array
    val iter = tree.fieldNames()
    val paramList = new ArrayBuffer[ArrayBuffer[Any]]()
    paramList.append(new ArrayBuffer[Any]())
    var countIndex = 0
    for (param <- existParams) {
      val copyTmp = paramList.clone()

      val paramArr = tree.get(param)
      for (i <- 0 until paramArr.size()) {
        if (i == 0) {
          for (j <- 0 until paramList.length) {
            paramList(j).append(paramArr.get(0).toString)
          }
        } else {
          for (j <- 0 until copyTmp.length) {
            paramList.append(copyTmp(j).clone())
          }
          for (j <- copyTmp.length*i until copyTmp.length * (i + 1)) {
            paramList(j)(countIndex) = paramArr.get(i).toString
          }
        }
      }
      countIndex += 1
    }

    (existParams, paramList)
  }


  def getBestModel(spark:SparkSession, modelAuc: Map[String, Double],
                   modelPath: String, bestModelPath: String,
                   testPath:String): Unit = {
    import ml.dmlc.xgboost4j.scala.spark.XGBoost
    import spark.implicits._

    println("Best Model")
    val x_test = spark.read.parquet(testPath)

    var bestAuc = 0.0
    var bestKey = ""
    var i = 0
    modelAuc.foreach {
      row =>

        val saveModelPath = row._1
        //import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel

        val xgbModel: XGBoostModel = XGBoost.loadModelFromHadoopFile(s"$modelPath/$saveModelPath")(spark.sparkContext)
        val predictDF = xgbModel.transform(x_test)

        predictDF.show()
        predictDF.groupBy("label","prediction").count().show()

        //split probability
        val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
        val toArrUdf = udf(toArr)
        val predictDFTmp = predictDF.withColumn("probability_arr",toArrUdf(col("probabilities")))

        val predictDFWithProbSplit = predictDFTmp
          .select("label","prediction","probability_arr")
          .rdd.map{
          line =>

            val label = line.getInt(0)
            val prediction = line.getDouble(1)

            val probability = line.getSeq(2).toArray[Double]
            val probNeg = probability(0)
            val probPos = probability(1)
            //val mkStr = probability.mkString(",")

            (label, prediction, probNeg,probPos)
        }.toDF("label","prediction","probNeg","probPos")
        predictDFWithProbSplit.show()

        import org.apache.spark.sql.types._
        val predictAndLabel = predictDFWithProbSplit.select($"probPos".cast(DoubleType), $"label".cast(DoubleType))
          .map { case Row(prediction: Double, label: Double) =>
            (prediction, label)
          }.rdd

        val metrics = new BinaryClassificationMetrics(predictAndLabel)
        val auROC = metrics.areaUnderROC
        val auPR = metrics.areaUnderPR()

        println(s"$saveModelPath" + " auc: " + s"$auROC"  +"     auPR: " + s"$auPR")

        if (auROC > bestAuc) {
          bestAuc = auROC
          bestKey = row._1
        }
        i += 1
    }
    if (bestKey != "") {
      println(s"gridSearch  getBestModel  bestKey : ${bestKey}")
      bestKey
      val bestModel = s"$modelPath" + "/" + s"$bestKey"
      import scala.sys.process._

      val copyModel = s"hdfs dfs -cp $bestModel $bestModelPath" !

    } else {
      sys.exit(1)
    }
  }

  def checkModelExist(modelPath: String, key: String): Boolean = {
    val conf = new Configuration()
    val HDFSFileSytem = FileSystem.get(conf)

    HDFSFileSytem.exists(new Path(modelPath.stripSuffix("/") + "/" + key.stripSuffix(",")))
  }


  def runModelGrid(spark:SparkSession, existParams: ArrayBuffer[String], paramList: ArrayBuffer[ArrayBuffer[Any]],
                   trainPath: String,testPath: String, modelPath: String): Map[String, Double] = {

    var modelAuc: Map[String, Double] = Map()

    //val x_train1 = spark.read.parquet(trainPath).repartition(2048)
    //println(s"x_train1.count  : ${x_train1.count()}")
    //x_train1.show(30)

    val x_train = spark.read.parquet(trainPath)
      .drop("sample_type")
      .distinct()
      //.randomSplit(Array(0.7, 0.3), seed = 1673419217)(0)
      //.repartition(1024)
      .coalesce(numCoalesce)
   // println(s"x_train.count  : ${x_train.count()}")
    x_train.persist(StorageLevel.MEMORY_AND_DISK)
    println(s"x_train.count  : ${x_train.count()}")
    //x_train.show()

    paramList.foreach { cur =>
      var i = 0
      var line = ""
      while (i < existParams.length) {
        line += existParams(i) + ":" + cur(i) + ","
        i += 1
      }

      if (!checkModelExist(modelPath, line.stripSuffix(",")
        //.replace("=","_")
        .replace(":","_")
        .replace(",","_"))) {

        val obj = new XgboostSort()

        obj.train(spark, x_train,testPath, line.stripSuffix(","), modelPath, modelAuc)
        //spark.close()
      }
    }
    println("gird over!")

    modelAuc
  }


  def main(args: Array[String]): Unit = {
    val trainPath = args(0)
    val testPath = args(1)
    val modelPath = args(2).stripSuffix("/")
    val bestModelPath = args(3)

    val spark = SparkSession
      .builder()
      .appName("gridSearch")
      .enableHiveSupport()
      .getOrCreate()

    val (paramParams, paramParamsArr) = paramGridList()
    println(paramParams)
    println(paramParamsArr)

    val modelAuc = runModelGrid(spark,paramParams, paramParamsArr, trainPath,testPath, modelPath)
    getBestModel(spark, modelAuc, modelPath, bestModelPath, testPath)

  }
}

