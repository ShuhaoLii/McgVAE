import pandas as pd

# 给定的列数序列
col_sequence = [6, 5, 5, 6, 5, 6, 5, 5]

speed = pd.read_csv('./Datasets/PeMS_speed_43.csv',index_col=0)

# 初始化一个空的 DataFrame 用于存储结果，索引与原始 DataFrame 对齐
result_df_custom_cols = pd.DataFrame(index=speed.index)

# 当前开始处理的列的索引
start_col_index = 0

# 遍历给定的列数序列，计算每个区段的平均值
for i, cols in enumerate(col_sequence):
    # 计算指定列数的平均值
    avg_col = speed.iloc[:, start_col_index:start_col_index+cols].mean(axis=1)
    # 将计算得到的平均值作为新的一列添加到结果 DataFrame 中
    result_df_custom_cols[f'Avg_{i+1}'] = avg_col
    # 更新下一个区段开始处理的列的索引
    start_col_index += cols

# 验证修正后的结果 DataFrame 的形状是否为 (8059, 8)
result_shape_custom_cols = result_df_custom_cols.shape

# 输出修正后的结果 DataFrame 的前几行以及形状，用于验证
result_df_custom_cols.head(), result_shape_custom_cols

result_df_custom_cols.to_csv('./Datasets/PeMS_road_speed_43.csv')