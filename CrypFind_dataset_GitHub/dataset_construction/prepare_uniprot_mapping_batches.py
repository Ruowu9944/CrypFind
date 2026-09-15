import pandas as pd
import os

# 1. 加载你那 150 万行的清单
df = pd.read_csv('global_high_quality_targets.csv')
# 2. 提取不重复的 UniProt ID
unique_ids = df['uniprotAccession'].unique().tolist()
print(f"总计不重复 ID 数量: {len(unique_ids)}")

# 3. 每 10 万个 ID 分成一个小文件，存入 ids_to_map 文件夹
os.makedirs('ids_to_map', exist_ok=True)
chunk_size = 100000
for i in range(0, len(unique_ids), chunk_size):
    sub_list = unique_ids[i : i + chunk_size]
    with open(f'ids_to_map/batch_{i//chunk_size}.txt', 'w') as f:
        f.write('\n'.join(sub_list))

print(f"分批完成，请去 ids_to_map 文件夹查看 batch_*.txt 文件。")