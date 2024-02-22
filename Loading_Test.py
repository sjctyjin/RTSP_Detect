import pandas as pd
import datetime
data = pd.read_csv('Static/product_info.csv', header=None,
                 names=['Product', 'Product_ID', 'ZoneCount', 'Position', 'Setup_time'])

proddata = {
        'Product': "測試",
        'Product_ID': 123,
        'ZoneCount': 2,  # 將 ndarray 轉換為列表
        'Position': [""],  # 將 ndarray 轉換為列表
        'Setup_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

print(data)
print(len(data))

new_df = pd.DataFrame(proddata)

new_df.to_csv('Static/product_info.csv', mode='a', header=False, index=False)
data = pd.read_csv('Static/product_info.csv', header=None,
                 names=['Product', 'Product_ID', 'ZoneCount', 'Position', 'Setup_time'])

print("刷新值 : ", data)
