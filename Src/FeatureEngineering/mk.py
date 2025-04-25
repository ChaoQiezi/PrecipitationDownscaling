# @Author  : ChaoQiezi
# @Time    : 2025/4/23 上午11:55
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: mk

"""
This script is used to 
"""

import matplotlib.pyplot as plt
import pandas as pd
import pymannkendall as mk

# 读取指定Sheet，假设数据结构为宽格式（城市为行，年份为列）
file_path = r"E:\MyTEMP\MK\综合得分data.xls"
out_path = r"E:\MyTEMP\MK\MK_result.xlsx"
df = pd.read_excel(file_path, sheet_name='data', index_col='地级行政区')


# 定义MK趋势分类函数
def classify_mk_trend(scores):
    if len(scores) < 2:
        return "数据不足"
    try:
        result = mk.original_test(scores)
        p = result.p
        trend = result.trend
        slope = result.slope

        if trend == 'decreasing':
            if p <= 0.01:
                return '极显著减少'
            elif p <= 0.05:
                return '显著减少'
            else:
                return '不显著减少'
        elif trend == 'increasing':
            if p <= 0.01:
                return '极显著增加'
            elif p <= 0.05:
                return '显著增加'
            else:
                return '不显著增加'
        else:
            return '不显著增加' if slope > 0 else '不显著减少'
    except:
        return "分析失败"


# 对每个城市应用MK检验并分类
results = {}
for city in df.index:
    scores = df.loc[city, 'CompScore_8062'].dropna().values
    results[city] = classify_mk_trend(scores)

# 转换为DataFrame并保存结果
result_df = pd.DataFrame.from_dict(results, orient='index', columns=['趋势类别'])
result_df.to_excel(out_path)

print("分析完成！结果已保存至 MK趋势分析结果.xlsx")