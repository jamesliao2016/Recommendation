package org.jd.datamill.rec.predict

case class TaskResult(taskId: String,
                      brand: String,
                      cate3: String,
                      userPath: String,
                      errorCode: Int,
                      errorMessage: String){

  def toJson():String =
    s"""{"taskId":$taskId,"brand":"$brand","cate3":"$cate3","userPath":"$userPath",
       |"errorCode":$errorCode,"errorMessage":"$errorMessage"}""".stripMargin
}

