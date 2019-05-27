from collections import Counter
from functools import partial
from linear_algebra import dot, vector_add
from stats import median, standard_deviation
from probability import normal_cdf
from gradient_descent import minimize_stochastic
from simple_linear_regression import total_sum_of_squares
import math, random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import pandas
import numpy as np


def predict(x_i, beta):
    return dot(x_i, beta)


def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)


def squared_error(x_i, y_i, beta):
    return error(x_i, y_i, beta) ** 2


def squared_error_gradient(x_i, y_i, beta):
    """the gradient corresponding to the ith squared error term"""
    return [-2 * x_ij * error(x_i, y_i, beta)
            for x_ij in x_i]


def estimate_beta(x, y):
    beta_initial = [random.random() for x_i in x[0]]
    return minimize_stochastic(squared_error,
                               squared_error_gradient,
                               x, y,
                               beta_initial,
                               0.001)


def multiple_r_squared(x, y, beta):
    sum_of_squared_errors = sum(error(x_i, y_i, beta) ** 2
                                for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)


def bootstrap_sample(data):
    """randomly samples len(data) elements with replacement"""
    return [random.choice(data) for _ in data]


def bootstrap_statistic(data, stats_fn, num_samples):
    """evaluates stats_fn on num_samples bootstrap samples from data"""
    return [stats_fn(bootstrap_sample(data))
            for _ in range(num_samples)]


def estimate_sample_beta(sample):
    x_sample, y_sample = list(zip(*sample))  # magic unzipping trick
    return estimate_beta(x_sample, y_sample)


def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)


#
# REGULARIZED REGRESSION
#

# alpha is a *hyperparameter* controlling how harsh the penalty is
# sometimes it's called "lambda" but that already means something in Python
def ridge_penalty(beta, alpha):
    return alpha * dot(beta[1:], beta[1:])


def squared_error_ridge(x_i, y_i, beta, alpha):
    """estimate error plus ridge penalty on beta"""
    return error(x_i, y_i, beta) ** 2 + ridge_penalty(beta, alpha)


def ridge_penalty_gradient(beta, alpha):
    """gradient of just the ridge penalty"""
    return [0] + [2 * alpha * beta_j for beta_j in beta[1:]]


def squared_error_ridge_gradient(x_i, y_i, beta, alpha):
    """the gradient corresponding to the ith squared error term
    including the ridge penalty"""
    return vector_add(squared_error_gradient(x_i, y_i, beta),
                      ridge_penalty_gradient(beta, alpha))


def estimate_beta_ridge(x, y, alpha):
    """use gradient descent to fit a ridge regression
    with penalty alpha"""
    beta_initial = [random.random() for x_i in x[0]]
    return minimize_stochastic(partial(squared_error_ridge, alpha=alpha),
                               partial(squared_error_ridge_gradient,
                                       alpha=alpha),
                               x, y,
                               beta_initial,
                               0.001)


def lasso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])


# -------------------------------------------------------------------------------------------------------------------------


def cv_diff_value(df, start_date, term, nameposition):  # 종가 일간 변화량
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 3) or (i + j > 230 - 2):
            break
        df.loc[i + j + nameposition, "cv_diff_value"] = df.values[i + j + nameposition][6] - df.values[i + j + nameposition + 1][6]


def cv_diff_rate(df, start_date, term, nameposition):  # 종가 일간 변화율
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 3) or (i + j > 230 - 2):
            break
        if float(df.values[i + j + nameposition + 1][6]) == float(0):
            df.loc[i + j + nameposition, "cv_diff_rate"] = 0
        else:
            df.loc[i + j + nameposition, "cv_diff_rate"] = abs(df.values[i + j + nameposition][6] / df.values[i + j + nameposition + 1][6] - 1) * 100

def cv_diff_rate_rate(df, start_date, term, nameposition):  # 종가 일간 변화율의 변화율
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 3) or (i + j > 230 - 2):
            break
        if df.values[i + j + nameposition + 1][10] == None:
            df.loc[i + j + nameposition, "cv_diff_rate_rate"] = 0
        if float(df.values[i + j + nameposition + 1][10]) == float(0):
            df.loc[i + j + nameposition, "cv_diff_rate_rate"] = 0
        else:
            df.loc[i + j + nameposition, "cv_diff_rate_rate"] = abs(df.values[i + j + nameposition][10] / df.values[i + j + nameposition + 1][10] - 1) * 100



def cv_ma3_value(df, start_date, term, nameposition):  # 종가의 3일 이동평균
    for i in range(int(term) + 1):
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 5) or (i + j > 230 - 4):
            break
        df.loc[i + j + nameposition, "cv_ma3_value"] = (df.values[i + j + nameposition][6] +
                                                        df.values[i + j + nameposition + 1][6] +
                                                        df.values[i + j + nameposition + 2][6])/3


def cv_ma3_rate(df, start_date, term, nameposition):  # 종가의 3일 이동평균의 일간 변화율
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 3) or (i + j > 230 - 2):
            break
        if df.values[i + j + nameposition + 1][11] == None:
            df.loc[i + j + nameposition, "cv_ma3_rate"] = 0
        if float(df.values[i + j + nameposition + 1][11]) == float(0):
            df.loc[i + j + nameposition, "cv_ma3_rate"] = 0
        else:
            df.loc[i + j + nameposition, "cv_ma3_rate"] = abs(df.values[i + j + nameposition][11] / df.values[i + j + nameposition + 1][11] - 1) * 100


def cv3d_diff_rate(df, start_date, term, nameposition):  # 3일간의 종가 상승률을 2번째 날의 값으로 설정
    if start_date == "20171222":
        start_date = int(start_date) - 1
    start_date = int(start_date) + 1  # 2번째 날이어야하니까 하나를 올려 계산
    while 1: # 주말 검출시 하루씩 밀어 계산
        if int(start_date) in df.basic_date.values:
            break
        else:
            start_date = int(start_date) + 1
    for i in range(int(term) + 1):
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    if start_date == 20171222:
                        j -= 1
                    break
        if (i + j + nameposition > 476490 - 5) or (i + j > 230 - 4):
            break
        if i + j + nameposition == -1:
            continue
        if float(df.values[i + j + nameposition + 5][6]) == float(0):
            df.loc[i + j + nameposition, "cv3d_diff_rate"] = 0
        else:
            df.loc[i + j + nameposition + 1, "cv3d_diff_rate"] = abs(df.values[i + j + nameposition][6] / df.values[i + j + nameposition + 3][6] - 1) * 100


def ud_3d(df, start_date, term, nameposition):
    if start_date == "20171222":
        start_date = int(start_date) - 1
    start_date = int(start_date) + 1  # 2번째 날이어야하니까 하나를 올려 계산
    while 1: # 주말 검출시 하루씩 밀어 계산
        if int(start_date) in df.basic_date.values:
            break
        else:
            start_date = int(start_date) + 1
    for i in range(int(term) + 1):  # 주식 위치 찾기
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    if start_date == 20171222:
                        j -= 1
                    break
        if (i + j + nameposition > 476490 - 5) or (i + j > 230 - 4):
            break
        if i + j + nameposition == -1:
            continue
        if ((df.loc[i + j + nameposition, "cv_diff_value"] > 0) and
                (df.loc[i + j + nameposition + 1, "cv_diff_value"] > 0) and
                (df.loc[i + j + nameposition + 2, "cv_diff_value"] > 0)):  # 5일 연속 종가 상승할때 2번째 날의 값 1
            df.loc[i + j + nameposition + 1, "ud_3d"] = 1
        elif ((df.loc[i + j + nameposition, "cv_diff_value"] < 0) and
              (df.loc[i + j + nameposition + 1, "cv_diff_value"] < 0) and
              (df.loc[i + j + nameposition + 2, "cv_diff_value"] < 0)):  # 5일 연속 종가 하락할때 2번째 날의 값 -1
            df.loc[i + j + nameposition + 1, "ud_3d"] = -1
        else:  # 2번째 날의 값 0
            df.loc[i + j + nameposition + 1, "ud_3d"] = 0


def  vv_diff_value(df, start_date, term, nameposition):  # 거래량 일간 변화량
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 3) or (i + j > 230 - 2):
            break
        df.loc[i + j + nameposition, "vv_diff_value"] = df.values[i + j + nameposition][7] - df.values[i + j + nameposition + 1][7]


def vv_diff_rate(df, start_date, term, nameposition):  # 거래량 일간 변화율
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 3) or (i + j > 230 - 2):
            break
        if float(df.values[i + j + nameposition + 1][7]) == float(0):
            df.loc[i + j + nameposition, "vv_diff_rate"] = 0
        else:
            df.loc[i + j + nameposition, "vv_diff_rate"] = abs(df.values[i + j + nameposition][7] / df.values[i + j + nameposition + 1][7] - 1) * 100


def vv_ma3_value(df, start_date, term, nameposition):  # 거래량의 3일 이동평균
    for i in range(int(term) + 1):
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 5) or (i + j > 230 - 3):
            break
        df.loc[i + j + nameposition, "vv_ma3_value"] = (df.values[i + j + nameposition][7] +
                                                        df.values[i + j + nameposition + 1][7] +
                                                        df.values[i + j + nameposition + 2][7]) / 3


def vv_ma3_rate(df, start_date, term, nameposition):  # 거래량의 3일 이동평균의 변화율
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 5) or (i + j > 230 - 4):
            break
        if float(df.values[i + j + nameposition + 1][17]) == float(0):
            df.loc[i + j + nameposition, "vv_ma3_rate"] = 0
        else:
            df.loc[i + j + nameposition, "vv_ma3_rate"] = abs(df.values[i + j + nameposition][17] / df.values[i + j + nameposition + 1][17] - 1) * 100


if __name__ == "__main__":
    # 트레이닝-------------------------------------------------소스코드
    df = pandas.read_csv('stock_history.csv', encoding='CP949')  # basic_date, stockname, stock_code, open_value, high_value, low_value, close_value, volume_value

    for i in range(8, 14): # 쓸데없는값 제거
        del df["Unnamed: " + str(i)]

    nameposition = 0  # 선택한 주식의 시작위치

    while 1:
        start_date = input("시작 일을 입력하시오(ex . 20171222) : ")  # 학습 데이터의 시작일자 입력
        if int(start_date) in df.basic_date.values:
            break
        else:
            print("해당 일자의 주식 정보가 없습니다. 확인후 이용해 주시기 바랍니다.")

    term = input("학습 기간을 입력하시오(주말 제외) : ")  # 학습데이터의 학습 기간을 입력

    companyname = input("원하는 주식 명을 입력하세요 : ")  # 원하는 주식명 입력

    for k in range(len(df)):
        if df.loc[k,"stockname"] == str(companyname):
            nameposition = k
            break

    df["bias"] = 1
    cv_diff_value(df, start_date, term, nameposition)
    cv_diff_rate(df, start_date, term, nameposition)
    cv_ma3_value(df, start_date, term, nameposition)
    cv_ma3_rate(df, start_date, term, nameposition)
    cv3d_diff_rate(df, start_date, term, nameposition)
    ud_3d(df, start_date, term, nameposition)
    vv_diff_value(df, start_date, term, nameposition)
    vv_diff_rate(df, start_date, term, nameposition)
    vv_ma3_value(df, start_date, term, nameposition)
    vv_ma3_rate(df, start_date, term, nameposition)
    cv_diff_rate_rate(df, start_date, term, nameposition)

    df.to_csv('stock_history_added.csv', encoding='CP949')

    df = df.dropna(axis=0)

    dfx = df[["bias", "cv_diff_rate", "cv_ma3_rate", "ud_3d", "cv_diff_rate_rate"]]  # 독립변수
    dfy = df[["cv3d_diff_rate"]]  # 종속변수

    dfx = dfx.values
    dfy = dfy.values

    dfy = np.ravel(dfy, order='C')  # 1차원 리스트로 변환

    # 테스트---------------------------------------------------소스코드

    df = pandas.read_csv('stock_history_added.csv', encoding='CP949')  # basic_date, stockname, stock_code, open_value, high_value, low_value, close_value, volume_value
    del df["Unnamed: " + str(0)] # 앞에서 인덱스값 저장된것을 지워준다.

    while 1:
        start_date = input("테스트 시작 일을 입력하시오(ex . 20171222) : ")  # 테스트 시작일 입력
        if int(start_date) in df.basic_date.values:
            break
        else:
            print("해당 일자의 주식 정보가 없습니다. 확인후 이용해 주시기 바랍니다.")

    term = input("테스트 기간을 입력하시오(주말 제외) : ")  # 테스트 기간 입력

    companyname = input("원하는 주식 명을 입력하세요 : ")  # 원하는 주식명 입력

    for k in range(len(df)):  # 원하는 주식의 위치를 저장
        if df.loc[k, "stockname"] == str(companyname):
            nameposition = k
            break

    df["bias"] = 1

    cv_diff_value(df, start_date, term, nameposition)
    cv_diff_rate(df, start_date, term, nameposition)
    cv_ma3_value(df, start_date, term, nameposition)
    cv_ma3_rate(df, start_date, term, nameposition)
    cv3d_diff_rate(df, start_date, term, nameposition)
    ud_3d(df, start_date, term, nameposition)
    vv_diff_value(df, start_date, term, nameposition)
    vv_diff_rate(df, start_date, term, nameposition)
    vv_ma3_value(df, start_date, term, nameposition)
    vv_ma3_rate(df, start_date, term, nameposition)
    cv_diff_rate_rate(df, start_date, term, nameposition)

    df.to_csv('stock_history_added.csv', encoding='CP949')

    df = df.dropna(axis=0)

    dftx = df[["bias", "cv_diff_rate", "cv_ma3_rate", "ud_3d", "cv_diff_rate_rate"]]  # 독립변수
    dfty = df[["cv3d_diff_rate"]]  # 종속변수

    dftx = dftx.values
    dfty = dfty.values

    dfty = np.ravel(dfty, order='C')

    random.seed(0)

    myreg = LinearRegression(False).fit(dfx, dfy)  # 알파를 베타의 첫항목으로 계산
    print("beta of LR : ", myreg.coef_)

    print("training data : ", multiple_r_squared(dfx, dfy, myreg.coef_))
    print("test data : ", multiple_r_squared(dftx, dfty, myreg.coef_))

    # 엑셀파일에 저장하자.
    df = pandas.read_csv('stock_history_added.csv', encoding='CP949')  # basic_date, stockname, stock_code, open_value, high_value, low_value, close_value, volume_value
    del df["Unnamed: " + str(0)]  # 앞에서 인덱스값 저장된것을 지워준다.

    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):  # 처음에 시작일자를 j로 설정
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        df.loc[i + j + nameposition, "predict"] = myreg.predict(dftx[i].reshape(1, 5))
    df.to_csv('stock_history_added.csv', encoding='CP949')

    for alpha in [0.0, 0.01, 0.1, 1, 10]:
        ridge_reg = Ridge(alpha, fit_intercept=False, solver="auto")
        ridge_reg.fit(dfx, dfy)
        beta = ridge_reg.coef_
        print("alpha", alpha)
        print("beta", beta)
        print("training : ", multiple_r_squared(dfx, dfy, beta))
        print("test : ", multiple_r_squared(dftx, dfty, beta))