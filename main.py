from utils.preprocess import preprocess_pipeline_xlsx
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

def evaluate_model(name, y_true, y_pred):
    """评估模型性能，计算各种指标"""
    # 计算RMSE (Root Mean Squared Error)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 计算MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))) * 100
    
    print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    return {
        'name': name,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape
    }

def save_evaluation_results(results, file_path='results/model_evaluation.json'):
    """保存模型评估结果为JSON文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 添加时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_with_timestamp = {
        'timestamp': timestamp,
        'models': results
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_timestamp, f, ensure_ascii=False, indent=4)
    
    print(f"评估结果已保存到 {file_path}")

def plot_results(results, file_path='results/model_comparison.png'):
    """绘制模型比较图表"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    model_names = [result['name'] for result in results]
    rmse_values = [result['rmse'] for result in results]
    r2_values = [result['r2'] for result in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 第一个Y轴：RMSE
    bars1 = ax1.bar(x - width/2, rmse_values, width, label='RMSE', color='#2878B5')
    ax1.set_ylabel('RMSE（越低越好）', color='#2878B5')
    ax1.tick_params(axis='y', labelcolor='#2878B5')
    
    # 第二个Y轴：R²
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, r2_values, width, label='R²', color='#9AC9DB')
    ax2.set_ylabel('R²（越高越好）', color='#9AC9DB')
    ax2.tick_params(axis='y', labelcolor='#9AC9DB')
    
    # X轴设置
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    
    # 标题和图例
    plt.title('二手车价格预测模型对比')
    
    # 在每个柱子上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # 创建联合图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(file_path)
    print(f"模型对比图已保存到 {file_path}")

def tune_hyperparameters(X_train, y_train):
    """对XGBoost模型进行超参数调优"""
    print("\n开始XGBoost超参数调优...")
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # 创建基础模型
    xgb_model = xgb.XGBRegressor(random_state=42)
    
    # 创建网格搜索
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,  # 交叉验证折数
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,  # 使用所有CPU
        verbose=1
    )
    
    # 执行网格搜索
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"超参数调优完成，耗时 {end_time - start_time:.2f} 秒")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳RMSE: {-grid_search.best_score_:.4f}")
    
    # 保存最佳参数
    best_params = grid_search.best_params_
    with open('models/xgboost_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    return best_params

def main():
    print("=== 二手车价格预测模型训练与评估 ===")
    start_time = time.time()
    
    # 数据预处理
    X_train, X_test, y_train, y_test = preprocess_pipeline_xlsx('data/二手车数据.xlsx')
    
    # 创建模型字典
    models = {
        '线性回归': LinearRegression(),
        'Ridge回归': Ridge(alpha=1.0),
        'Lasso回归': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost (默认)': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }
    
    # 是否进行超参数调优
    do_hyperparameter_tuning = True
    
    if do_hyperparameter_tuning:
        try:
            # 超参数调优（可选，耗时较长）
            best_params = tune_hyperparameters(X_train, y_train)
            
            # 使用最优参数创建XGBoost模型
            models['XGBoost (调优)'] = xgb.XGBRegressor(**best_params, random_state=42)
        except Exception as e:
            print(f"超参数调优过程中出错: {e}")
            print("跳过调优，继续使用默认参数")
    
    # 训练和评估模型
    results = []
    
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        model_start_time = time.time()
        model.fit(X_train, y_train)
        model_train_time = time.time() - model_start_time
        
        print(f"{name} 模型训练完成，耗时 {model_train_time:.2f} 秒")
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        result = evaluate_model(name, y_test, y_pred)
        result['train_time'] = model_train_time
        results.append(result)
        
        # 保存模型
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{name.replace(' ', '_')}_model.pkl")
    
    # 选择最优模型（以RMSE最小为标准）
    best_model_result = min(results, key=lambda x: x['rmse'])
    best_model_name = best_model_result['name']
    
    print(f"\n✅ 最优模型为：{best_model_name}")
    print(f"RMSE: {best_model_result['rmse']:.2f}, R²: {best_model_result['r2']:.2f}")
    
    # 保存评估结果
    save_evaluation_results(results)
    
    # 绘制对比图
    plot_results(results)
    
    # 重命名最优模型
    os.rename(
        f"models/{best_model_name.replace(' ', '_')}_model.pkl", 
        "models/best_model.pkl"
    )
    
    print(f"\n✅ 最优模型已保存为 models/best_model.pkl")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()