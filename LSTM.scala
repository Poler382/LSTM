import breeze.linalg._
import math._
object tester{
  
  def load()={
    var Xs=List(Array[Double](29))
    val f = scala.io.Source.fromFile("/home/share/text8").getLines.toList.head.take(2000).toArray
    val str = f
    for(i<-0 until str.size){
      val X=conv(str(i))
      Xs=X::Xs
    }
    Xs.reverse
  }
  def conv(c:Char)={
    val x=DenseVector.zeros[Double](29)
    x("abcdefghijklmnopqrstuvwxyz ".indexOf(c))=1d
    x.toArray
  }

  def learning(l:Double,num:Int,back:Int)={
    println("学習率:"+l+" 履歴長:"+back+" 学習回数:"+num)
    val ds=load()

    val ml = new ML()
    val lstm = new lstm(29,29,29)
    val af = new Affine (29,29)
    val sig = new Sigmoid()
    val layer = List(lstm,af,sig)
    
    for(n<- 0 to num){
      var count=0
      var err = 0d
      var output=""
      var target=""
      var Lst = new Stack[Array[Double]]()

      for(i<-0 until 200){
        for(j <- 0 to 20){
          
          val y = ml.forwards(layer,ds(i+j))
          Lst.push(y)
        }
        var ylast = Lst.head
        err += -(ylast.map(math.log).zip( ds(i+20)).map{case (a,b) => a*b}).sum

        for(j <- 20 to 0 by -1){
          val d = ml.backwards(layer,(Lst.pop -ds(i+j)))
        }


        output+="abcdefghijklmnopqrstuvwxyz []"(argmax(ylast))
        target+="abcdefghijklmnopqrstuvwxyz []"(argmax(ds(i+20)))

        if(argmax(ylast)==argmax(ds(i+20))){
          count+=1
        }

        ml.updates(layer)
       
      }

      println(n+":   ,"+count/num.toDouble*100.0+"%,"+err/200.0)
      if(n==num){
        println("----学習終了----")
        println(count/num.toDouble*100.0+"%")
        println(err/num.toDouble)
        println(output)
        println(target)
      }
    }

  }
}
//Tが可変の変数をうけとれるように
class Stack[T](){
  var x=List[T]()
  def push(a:T)={
    x = a::x
  }
  def pop()={
    var t = x.head
    x = x.tail
    t
  }
  def head()={
    x.head
  }

  def p(i:Int)={
    x(i)
  }
}


class lstm (val In : Int,val M:Int,val hsize :Int = 100)extends Layer{

  var h_hat_L  = List(new Affine(In+hsize,M),new Tanh())
  var it_L     = List(new Affine(In+hsize,M),new Sigmoid())
  var ft_L     = List(new Affine(In+hsize,M),new Sigmoid())
  var ot_L     = List(new Affine(In+hsize,M),new Sigmoid())
  var cList    = new Stack[Array[Double]]()
  var hList    = new Stack[Array[Double]]()
  var dList    = new Stack[Array[Double]]()
  var oList    = new Stack[Array[Double]]()
  var fList    = new Stack[Array[Double]]()
  var tanhList = new Stack[Array[Double]]()
  var itList   = new Stack[Array[Double]]()
  var h_hatList= new Stack[Array[Double]]()

  val t = new Tanh()
  hList.push(new Array[Double](29))
 
  def forward(xs:Array[Double])={
    
    var h_hat = learning.forwards(h_hat_L,xs++hList.head)
    var it    = learning.forwards(it_L,xs++hList.head)
    var ft    = learning.forwards(ft_L,xs++hList.head)
    var ot    = learning.forwards(ot_L,xs++hList.head)
   
    val c1 = it.zip(h_hat).map{case (a,b) => a*b}
    val c2 = ft.zip(cList.head).map{case (a,b) => a*b}
    val c  =c1.zip(c2).map{case (a,b) => a+b}

    cList.push(c)
    tanhList.push(t.forward(c))
    itList.push(it)
    h_hatList.push(h_hat)
    oList.push(ot)
    fList.push(ft)

    val h_t = ot.zip(tanhList.head).map{case (a,b) => a*b}

    hList.push(h_t)
    h_t
  }
  
  var rList   = new Stack[Array[Double]]()
  def backward(d:Array[Double])={
    val ds = d.zip(rList.head).map{case (a,b) => a+b}

    val b_ot=learning.backwards(ot_L,ds.zip(tanhList.pop()).map{case (a,b)=>a*b })
    val b_tanh = t.backward(ds.zip(oList.pop()).map{case (a,b) => a*b})

    val bc = cList.pop().zip(b_tanh).map{case (a,b) => a+b}
    val bf = learning.backwards(ft_L ,cList.head.zip(bc).map{case (a,b) => a*b})

    val m = cList.push( fList.head.zip(bc).map{case (a,b) => a*b})

    val bh_hat = learning.backwards(h_hat_L,bc.zip(itList.pop).map{case (a,b)=>a*b})

    val bi = learning.backwards(it_L,bc.zip(h_hatList.head).map{case (a,b) => a*b})
   
    var dxh =  List[Double]()

    for(i <- 0 until bi.size){
      val temp = bi(i)+bh_hat(i)+b_ot(i)+bf(i)
      dxh ::= temp
    }

    dxh = dxh.reverse

    val preh = dxh.drop(hsize)
    
    rList.push(preh.toArray)

    d
  }

  
  def update(){
    learning.updates(h_hat_L)
    learning.updates(it_L)
    learning.updates(ft_L)
    learning.updates(ot_L)
  }

  def reset(){
    learning.resets(h_hat_L)
    learning.resets(it_L)
    learning.resets(ft_L)
    learning.resets(ot_L)

  }




}
