import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from datetime import datetime
import os
import joblib

def load_xlsx_data(filepath):
    print(f"加载数据: {filepath}")
    df = pd.read_excel(filepath)
    print("原始数据前5行:")
    print(df.head())
    print(f"\n原始数据形状: {df.shape}")
    print("\n原始数据类型:")
    print(df.dtypes)
    return df

def clean_and_transform(df):
    print("\n开始数据清洗和转换...")
    # 重命名列方便处理
    column_mapping = {}
    if '车源号' in df.columns:
        column_mapping['车源号'] = 'car_id'
    if '品牌车型' in df.columns:
        column_mapping['品牌车型'] = 'brand_model'
    if '价格（万）' in df.columns:
        column_mapping['价格（万）'] = 'price'
    if '上牌时间' in df.columns:
        column_mapping['上牌时间'] = 'register_date'
    if '表显里程（万公里）' in df.columns:
        column_mapping['表显里程（万公里）'] = 'mileage'
    if '变速箱' in df.columns:
        column_mapping['变速箱'] = 'transmission'
    if '排量' in df.columns:
        column_mapping['排量'] = 'displacement'
    if '油耗' in df.columns:
        column_mapping['油耗'] = 'fuel_consumption'
    
    df = df.rename(columns=column_mapping)

    # 保留所有可能有用的特征列
    columns_to_drop = []
    potential_useful_columns = ['序号', 'price', 'mileage', 'brand_model', 'register_date', 
                               'transmission', 'displacement', 'fuel_consumption']
    
    for col in df.columns:
        if col not in potential_useful_columns:
            columns_to_drop.append(col)
    
    df = df.drop(columns=columns_to_drop)
    print(f"保留的列: {df.columns.tolist()}")

    # 删除价格或里程为空的样本
    df = df.dropna(subset=['price', 'mileage'])
    print(f"删除缺失价格或里程后的数据形状: {df.shape}")

    # 处理可能存在的非数值型里程数据
    def extract_numeric_mileage(mile_str):
        if isinstance(mile_str, (int, float)):
            return mile_str
        try:
            # 尝试提取数值，去除如"百公里内"等文本
            numeric_part = ''.join(c for c in str(mile_str) if c.isdigit() or c == '.')
            return float(numeric_part) if numeric_part else np.nan
        except:
            return np.nan

    df['mileage'] = df['mileage'].apply(extract_numeric_mileage)
    
    # 处理车龄信息
    if 'register_date' in df.columns:
        def calculate_age(date_str):
            try:
                if pd.isna(date_str):
                    return np.nan
                # 处理不同格式的日期
                if isinstance(date_str, str):
                    if '年' in date_str:
                        year = int(date_str.split('年')[0])
                    else:
                        # 尝试其他格式
                        reg_time = pd.to_datetime(date_str, errors='coerce')
                        if pd.isna(reg_time):
                            return np.nan
                        year = reg_time.year
                elif isinstance(date_str, pd.Timestamp):
                    year = date_str.year
                else:
                    return np.nan
                
                current_year = datetime.now().year
                return current_year - year
            except:
                return np.nan

        df['car_age'] = df['register_date'].apply(calculate_age)
        df.drop(['register_date'], axis=1, inplace=True)
    
    # 处理品牌信息
    if 'brand_model' in df.columns:
        def extract_brand(brand_model):
            if pd.isna(brand_model):
                return "未知"
            brand = str(brand_model).split('-')[0] if '-' in str(brand_model) else str(brand_model)
            # 提取第一个单词作为品牌
            import re
            match = re.search(r'^[\u4e00-\u9fa5a-zA-Z]+', brand)
            if match:
                return match.group()
            return brand
            
        df['brand'] = df['brand_model'].apply(extract_brand)
        
        # 提取车型（如果有）
        def extract_model(brand_model_str):
            if pd.isna(brand_model_str):
                return "未知"
            parts = str(brand_model_str).split('-')
            if len(parts) > 1:
                return parts[1]
            return "未知"
            
        df['model'] = df['brand_model'].apply(extract_model)
        df.drop(['brand_model'], axis=1, inplace=True)
        
        # 编码品牌
        le_brand = LabelEncoder()
        df['brand'] = le_brand.fit_transform(df['brand'])
        
        # 编码车型
        le_model = LabelEncoder()
        df['model'] = le_model.fit_transform(df['model'])
        
        # 保存编码器，便于后续使用
        os.makedirs('models', exist_ok=True)
        joblib.dump(le_brand, 'models/le_brand.pkl')
        joblib.dump(le_model, 'models/le_model.pkl')
    
    # 处理变速箱类型
    if 'transmission' in df.columns:
        def process_transmission(trans_str):
            if pd.isna(trans_str):
                return "未知"
            trans_str = str(trans_str).lower()
            if '自动' in trans_str or 'at' in trans_str:
                return "自动"
            elif '手动' in trans_str or 'mt' in trans_str:
                return "手动"
            elif '双离合' in trans_str or 'dct' in trans_str:
                return "双离合"
            elif '无级变速' in trans_str or 'cvt' in trans_str:
                return "无级变速"
            else:
                return "其他"
        
        df['transmission'] = df['transmission'].apply(process_transmission)
        le_trans = LabelEncoder()
        df['transmission'] = le_trans.fit_transform(df['transmission'])
        joblib.dump(le_trans, 'models/le_transmission.pkl')
    
    # 处理排量
    if 'displacement' in df.columns:
        def extract_displacement(disp_str):
            if pd.isna(disp_str):
                return np.nan
            try:
                # 提取数字部分
                import re
                match = re.search(r'(\d+\.?\d*)', str(disp_str))
                if match:
                    return float(match.group(1))
                return np.nan
            except:
                return np.nan
        
        df['displacement'] = df['displacement'].apply(extract_displacement)
    
    # 处理油耗
    if 'fuel_consumption' in df.columns:
        def extract_fuel_consumption(fuel_str):
            if pd.isna(fuel_str):
                return np.nan
            try:
                # 提取数字部分
                import re
                match = re.search(r'(\d+\.?\d*)', str(fuel_str))
                if match:
                    return float(match.group(1))
                return np.nan
            except:
                return np.nan
        
        df['fuel_consumption'] = df['fuel_consumption'].apply(extract_fuel_consumption)
    
    # 确保所有列都是数值型
    for col in df.columns:
        if col != '序号' and df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                print(f"警告: 列 '{col}' 无法转换为数值型，将被删除")
                df.drop(col, axis=1, inplace=True)
    
    # 为每列单独填充缺失值
    for col in df.columns:
        if col != '序号' and df[col].dtype in [np.float64, np.int64]:
            # 使用中位数填充缺失值
            median_value = df[col].median()
            if not pd.isna(median_value):  # 确保中位数不是NaN
                df[col] = df[col].fillna(median_value)
            else:
                # 如果中位数是NaN（可能所有值都是NaN），则用0填充
                df[col] = df[col].fillna(0)
    
    # 创建交互特征
    if 'car_age' in df.columns and 'mileage' in df.columns:
        # 年均里程
        df['annual_mileage'] = df['mileage'] / np.maximum(1, df['car_age'])
    
    # 删除异常值（可选）
    if 'price' in df.columns:
        q1 = df['price'].quantile(0.01)
        q3 = df['price'].quantile(0.99)
        df = df[(df['price'] >= q1) & (df['price'] <= q3)]
        print(f"删除价格异常值后的数据形状: {df.shape}")
    
    print("\n处理后数据前5行:")
    print(df.head())
    print(f"\n处理后数据形状: {df.shape}")
    print("\n处理后数据类型:")
    print(df.dtypes)
    
    # 保存处理后的数据
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/cleaned_data.csv', index=False)
    print("处理后的数据已保存到 'data/processed/cleaned_data.csv'")
    
    return df

def create_features(X_train, X_test):
    """创建额外的特征"""
    print("\n创建额外特征...")
    
    # 标准化数值特征
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    num_features = [f for f in num_features if f != '序号']
    
    if num_features:
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
        X_test_scaled[num_features] = scaler.transform(X_test[num_features])
        
        # 保存scaler
        joblib.dump(scaler, 'models/scaler.pkl')
        
        # 创建多项式特征
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        poly_features_train = poly.fit_transform(X_train_scaled[num_features])
        poly_features_test = poly.transform(X_test_scaled[num_features])
        
        # 保存多项式转换器
        joblib.dump(poly, 'models/poly_features.pkl')
        
        # 创建多项式特征的列名
        poly_features_names = []
        for i, feat1 in enumerate(num_features):
            for j in range(i, len(num_features)):
                feat2 = num_features[j]
                poly_features_names.append(f"{feat1}_{feat2}")
        
        # 确保生成的特征名长度与实际特征数量匹配
        poly_features_names = poly_features_names[:poly_features_train.shape[1]]
        
        # 添加多项式特征到数据集
        poly_train_df = pd.DataFrame(poly_features_train, columns=poly_features_names)
        poly_test_df = pd.DataFrame(poly_features_test, columns=poly_features_names)
        
        # 重置索引以确保连接顺序正确
        X_train_scaled = X_train_scaled.reset_index(drop=True)
        X_test_scaled = X_test_scaled.reset_index(drop=True)
        poly_train_df = poly_train_df.reset_index(drop=True)
        poly_test_df = poly_test_df.reset_index(drop=True)
        
        X_train_final = pd.concat([X_train_scaled, poly_train_df], axis=1)
        X_test_final = pd.concat([X_test_scaled, poly_test_df], axis=1)
        
        print(f"添加多项式特征后训练集形状: {X_train_final.shape}")
        print(f"添加多项式特征后测试集形状: {X_test_final.shape}")
        
        return X_train_final, X_test_final
    else:
        print("未找到数值特征，跳过创建额外特征")
        return X_train, X_test

def split_data(df):
    print("\n拆分训练集和测试集...")
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def preprocess_pipeline_xlsx(filepath):
    df = load_xlsx_data(filepath)
    df = clean_and_transform(df)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 创建额外特征
    X_train_featured, X_test_featured = create_features(X_train, X_test)
    
    print("\nX_train前5行:")
    print(X_train_featured.head())
    print("\nX_train数据类型:")
    print(X_train_featured.dtypes)
    
    # 保存特征工程后的数据集
    os.makedirs('data/featured', exist_ok=True)
    X_train_featured.to_csv('data/featured/X_train.csv', index=False)
    X_test_featured.to_csv('data/featured/X_test.csv', index=False)
    pd.Series(y_train).to_csv('data/featured/y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/featured/y_test.csv', index=False)
    print("特征工程后的数据已保存到 'data/featured/' 目录")
    
    return X_train_featured, X_test_featured, y_train, y_test