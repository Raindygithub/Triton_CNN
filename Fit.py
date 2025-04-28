import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial_regression_fit(data, degree=3, plot=True):
    """
    使用多项式回归拟合数据
    :param data: 包含数值的张量列表，如 [torch.tensor(0.1250), ...]
    :param degree: 多项式的阶数，默认为3
    :param plot: 是否绘制拟合曲线，默认为True
    :return: 多项式回归模型
    """
    # 确保 data 中的每个元素都是 torch.Tensor 类型
    y = np.array([item.item() if isinstance(item, torch.Tensor) else item for item in data])
    x = np.arange(1, len(data) + 1).reshape(-1, 1)  # 生成自增的 x 值

    # 生成多项式特征
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)

    # 使用线性回归拟合多项式特征
    model = LinearRegression()
    model.fit(x_poly, y)

    # 如果需要绘制拟合曲线
    if plot:
        # 生成用于绘图的 x 值
        x_fit = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        x_fit_poly = poly_features.transform(x_fit)
        y_fit = model.predict(x_fit_poly)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label='Original Data', color='blue')
        plt.plot(x_fit, y_fit, label=f'Polynomial Fit (Degree {degree})', color='red', linestyle='--')
        plt.xlabel('Steps/Epochs')
        plt.ylabel('Value')
        plt.title('Polynomial Regression Fit')
        plt.legend()
        plt.grid(True)
        plt.show()

    return model
