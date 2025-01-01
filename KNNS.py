import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import seaborn as sns

# 1. 資料讀取與初步處理
def load_data(dataset_name):
    if dataset_name == 'penguin':
        penguin = sns.load_dataset("penguins")
        data = pd.DataFrame(penguin)
        data.dropna(how="any",inplace=True)  # 處理缺失值
        #KNNS不能分析字串，因此這裡將島嶼資料替換為數字
        data["island"] = data["island"].map(
            {"Torgersen": "0", "Biscoe": "1", "Dream": "2"}
        )
        #鰭狀肢長度、喙深度、島嶼、體重
        #當兩個特徵時，我們選擇 鰭狀肢長度、喙深度
        #當三個特徵時，我們選擇 鰭狀肢長度、喙深度、島嶼
        features = ['flipper_length_mm', 'bill_depth_mm', 'island', 'body_mass_g']
        label = 'species'
    elif dataset_name == 'iris':
        iris = sns.load_dataset("iris")
        data = pd.DataFrame(iris)
        data.dropna(how="any",inplace=True)  # 處理缺失值
        #花瓣長度、花瓣寬度、萼片寬度、萼片長度
        features = ['petal_length', 'petal_width', 'sepal_width', 'sepal_length']
        label = 'species'
    else:
        raise ValueError("Unsupported dataset. Use 'penguin' or 'iris'.")
    return data, features, label

# 2. 前處理函數
def preprocess_data(data, features, method):
    scaler = None
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    if scaler:
        data[features] = scaler.fit_transform(data[features])
    return data

# 3. KNN 分析函數
def knn_analysis(data, features, label, n, k, preprocess=None, repeat=100):
    avg_acc_list = []
    std_acc_list = []
    
    for _ in range(repeat):  # 重複實驗
        # 隨機分割訓練集與測試集
        train_data, test_data = train_test_split(data, test_size=0.5, stratify=data[label])
        
        # 我們取總資料集的50%作為訓練母集
        # 從中隨機取出 n 個樣本
        # 若 n 為 5 我們會從各個label(penguin/iris品種)中各隨機取五個值進入訓練子集
        # 若 n 為 10 我們會從各個label (penguin/iris品種)中隨機取得十個值進入訓練子集 
        # 意即我們訓練集的大小會是 n 的倍數

        train_data = train_data.groupby(label).sample(n=n)
         
        
        # 前處理
        if preprocess:
            train_data = preprocess_data(train_data, features, preprocess)
            test_data = preprocess_data(test_data, features, preprocess)
        
        # 建模與測試
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_data[features], train_data[label])
        predictions = knn.predict(test_data[features])
        acc = accuracy_score(test_data[label], predictions)
        avg_acc_list.append(acc)
    
    # 計算平均辨識率和標準差
    avg_acc = np.mean(avg_acc_list)
    std_acc = np.std(avg_acc_list)
    return avg_acc, std_acc
# 4. 繪圖函數
def plot_results(results, n_values, k_values, feature_amount, preprocess_methods):
    for amount in feature_amount:
        for n in n_values:
            plt.figure(figsize=(10, 6))
            for method, acc_data in results[amount].items():
                avg_accs = [acc_data[n][k][0] for k in k_values]  # 平均辨識率
                plt.plot(k_values, avg_accs, label=f'Preprocess: {method}')
            plt.title(f'KNN Results (n={n}) (feature amount={amount})')
            plt.xlabel('K value')
            plt.xticks(k_values)
            plt.ylabel('Average Accuracy')
            plt.legend()
            plt.grid()
            plt.savefig(f"./result/{dataset_name} - features{amount} - n{n}.png")
            plt.show()

# 5. 主程序
if __name__ == "__main__":
    global dataset_name
    dataset_name = 'iris'  # 選擇數據集 'penguin' 或 'iris'
    data, features, label = load_data(dataset_name)
    
    print("processing")
    
    # 設定實驗參數
    n_values = [5, 10, 15, 20]
    k_values = [1, 3, 5, 7, 9]
    feature_amount = [2, 3, len(features)]
    preprocess_methods = ['none', 'minmax', 'zscore', 'robust']
    repeat = 100  # 重複次數
    
    # 保存結果
    results = {amount: {method: {} for method in preprocess_methods} for amount in feature_amount}
    
    for amount in feature_amount:
        selected_features = features[:amount]  # 根據特徵數量選擇特徵
        for preprocess in preprocess_methods:
            for n in n_values:
                results[amount][preprocess][n] = {}
                for k in k_values:
                    print(f"processing data.. features={amount}, n={n}, k={k}, preprocess={preprocess}")
                    if preprocess == 'none':
                        avg_acc, std_acc = knn_analysis(data, selected_features, label, n, k, preprocess=None, repeat=repeat)
                    else:
                        avg_acc, std_acc = knn_analysis(data, selected_features, label, n, k, preprocess=preprocess, repeat=repeat)
                    results[amount][preprocess][n][k] = (avg_acc, std_acc)
    
    # 繪製結果
    plot_results(results, n_values, k_values, feature_amount, preprocess_methods)
    
    print("end")