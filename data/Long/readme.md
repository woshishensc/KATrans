---
Topic:
    - 科技互联网
,    - 工业制造

Ext:
    - .zip

DatasetUsage:
    - 8282067

FolderName:
    - /home/mw/input/qvyu8549/
---

# 数据简介
“中国软件杯”大学生软件设计大赛——龙源风电赛道数据集，本赛题数据集由全球最大风电运营企业龙源电力提供，采集自真实风力发电数据。 预选赛训练数据和区域赛训练数据分别为不同10个风电场近一年的运行数据共30万余条，每15分钟采集一次，包括风速、风向、温度、湿度、气压和真实功率等。

# 读取方法
```python
import pandas as pd

data = pd.read_csv(file_name)
```

# 比赛链接
https://aistudio.baidu.com/aistudio/competition/detail/887/0/introduction


## **引用格式**
```
@misc{qvyu8549,
    title = { 2023大学生软件杯-龙源风电赛道数据 },
    author = { 小王同学呼啦啦 },
    howpublished = { \url{https://www.heywhale.com/mw/dataset/64cbbd6ff98f2683eaa84acc} },
    year = { 2023 },
}
```

对数据进行了修改，将ROUND(A.Power,0)复制到第二列，作为真实风功率数据。
检查发现原数据中16/18的真实风功率为空，故删掉