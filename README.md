## bilstm+crf实现的命名实体识别

bilstm+crf实现的命名实体识别，开箱即用



bisltm+crf的实现是在参考pytorch的官方教程的基础上，全部换成了矩阵并行操作



需要下载sogou预训练词向量，地址：http://www.sogou.com/labs/resource/cs.php

将下载的预训练词向量放入ResumeNER/data文件夹下面



训练完后进行测试：python extract.py --text "王强是高级工程师，毕业于XXX大学"



evaulate还有些不完善，后面有时间会改.....
