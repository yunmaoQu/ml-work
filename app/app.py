import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

# 加载模型
model = joblib.load("models/XGBoost_(调优)_model.pkl")

# 品牌映射（你可以根据实际训练数据补充）
brand_mapping = {
    '大众': 0,
    '丰田': 1,
    '本田': 2,
    '宝马': 3,
    '奥迪': 4,
    '奔驰': 5,
    # ... 添加更多品牌
}

st.set_page_config(page_title="二手车价格预测", layout="centered")
st.title("🚗 二手车价格预测系统")
st.markdown("请输入以下信息，系统将预测车辆售价（单位：万元）")

# --- 用户输入 ---
brand_name = st.selectbox("品牌", list(brand_mapping.keys()))
year = st.slider("上牌年份", 2005, datetime.datetime.now().year, 2018)
mileage = st.number_input("表显里程（万公里）", min_value=0.0, max_value=50.0, value=5.0, step=0.1)

# 添加可选高级选项
with st.expander("高级选项（可选）"):
    transmission = st.selectbox("变速箱类型", ["自动", "手动", "双离合", "无级变速", "其他"], index=0)
    transmission_mapping = {"自动": 0, "手动": 1, "双离合": 2, "无级变速": 3, "其他": 4}
    transmission_encoded = transmission_mapping[transmission]
    
    displacement = st.number_input("排量(L)", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
    fuel_consumption = st.number_input("百公里油耗(L)", min_value=4.0, max_value=15.0, value=7.0, step=0.1)
    model_value = st.selectbox("车型", ["普通型", "豪华型", "运动型", "商务型", "其他"], index=0)
    model_mapping = {"普通型": 0, "豪华型": 1, "运动型": 2, "商务型": 3, "其他": 4}
    model_encoded = model_mapping[model_value]

# --- 特征处理 ---
car_age = datetime.datetime.now().year - year
brand_encoded = brand_mapping[brand_name]

# 计算年均里程
annual_mileage = mileage / max(1, car_age)  # 避免除以零

# --- 预测 ---
if st.button("🚀 立即预测"):
    try:
        # 创建特征字典
        features = {
            '序号': [1],
            'mileage': [mileage],
            'transmission': [transmission_encoded],
            'displacement': [displacement],
            'fuel_consumption': [fuel_consumption],
            'car_age': [car_age],
            'brand': [brand_encoded],
            'model': [model_encoded],
            'annual_mileage': [annual_mileage]
        }
        
        # 添加交互特征
        for feat1 in ['mileage', 'transmission', 'displacement', 'fuel_consumption', 'car_age', 'brand', 'model', 'annual_mileage']:
            for feat2 in ['mileage', 'transmission', 'displacement', 'fuel_consumption', 'car_age', 'brand', 'model', 'annual_mileage']:
                if feat1 <= feat2:  # 只添加唯一的组合
                    feat_name = f"{feat1}_{feat2}"
                    features[feat_name] = [features[feat1][0] * features[feat2][0]]
        
        # 创建DataFrame
        input_df = pd.DataFrame(features)
        
        # 确保列的顺序与模型训练时相同
        expected_columns = ['序号', 'mileage', 'transmission', 'displacement', 'fuel_consumption', 
                           'car_age', 'brand', 'model', 'annual_mileage', 
                           'mileage_mileage', 'mileage_transmission', 'mileage_displacement', 
                           'mileage_fuel_consumption', 'mileage_car_age', 'mileage_brand', 
                           'mileage_model', 'mileage_annual_mileage', 'transmission_transmission', 
                           'transmission_displacement', 'transmission_fuel_consumption', 
                           'transmission_car_age', 'transmission_brand', 'transmission_model', 
                           'transmission_annual_mileage', 'displacement_displacement', 
                           'displacement_fuel_consumption', 'displacement_car_age', 
                           'displacement_brand', 'displacement_model', 'displacement_annual_mileage', 
                           'fuel_consumption_fuel_consumption', 'fuel_consumption_car_age', 
                           'fuel_consumption_brand', 'fuel_consumption_model', 
                           'fuel_consumption_annual_mileage', 'car_age_car_age', 'car_age_brand', 
                           'car_age_model', 'car_age_annual_mileage', 'brand_brand', 'brand_model', 
                           'brand_annual_mileage', 'model_model', 'model_annual_mileage', 
                           'annual_mileage_annual_mileage']
        
        # 检查是否缺少列并添加
        for col in expected_columns:
            if col not in input_df.columns:
                st.warning(f"缺少特征列: {col}，将使用默认值0")
                input_df[col] = 0
        
        # 确保列顺序一致
        input_df = input_df[expected_columns]
        
        # 预测
        prediction = model.predict(input_df)[0]
        st.success(f"✅ 预测价格为：¥ {prediction:.2f} 万元")
        
        # 显示主要影响因素
        st.subheader("价格主要影响因素:")
        st.info(f"• 车龄: {car_age}年（减少价值约{car_age*1.2:.1f}万元）")
        st.info(f"• 品牌等级: {brand_name}（品牌影响约±{abs(brand_encoded-2)*1.5:.1f}万元）")
        st.info(f"• 里程: {mileage}万公里（减少价值约{mileage*0.8:.1f}万元）")
        
    except Exception as e:
        st.error(f"预测发生错误: {str(e)}")
        st.info("请检查特征名称和顺序是否匹配")
        st.code(str(input_df.columns.tolist()))