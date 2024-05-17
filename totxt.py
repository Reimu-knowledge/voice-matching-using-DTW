import pandas as pd

# 从文本文件中读取内容到数据帧
df = pd.read_csv('./fbank/fbank.txt', delimiter=' ')


# 将数据帧写入Excel文件
df.to_excel('./fbank/fbank.xlsx', index=False)