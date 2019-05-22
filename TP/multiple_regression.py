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
            for j in range(len(df)):
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 3) or (i + j > 230 - 2):
            break
        if int(df.values[i + j + nameposition + 1][6]) == int(0):
            df.loc[i + j + nameposition, "cv_diff_rate"] = 0
        else:
            df.loc[i + j + nameposition, "cv_diff_rate"] = abs(df.values[i + j + nameposition][6] / df.values[i + j + nameposition + 1][6] - 1) * 100


def cv_ma5_value(df, start_date, term, nameposition):  # 종가의 5일 이동평균
    for i in range(int(term) + 1):
        if i == 0:
            for j in range(len(df)):
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 7) or (i + j > 230 - 5):
            break
        df.loc[i + j + nameposition, "cv_ma5_value"] = (df.values[i + j + nameposition][6] +
                                                        df.values[i + j + nameposition + 1][6] +
                                                        df.values[i + j + nameposition + 2][6] +
                                                        df.values[i + j + nameposition + 3][6] +
                                                        df.values[i + j + nameposition + 4][6])/5


def cv_ma5_rate(df, start_date, term, nameposition):  # # 종가의 5일 이동평균의 일간 변화율
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 7) or (i + j > 230 - 6):
            break
        if int(df.values[i + j + nameposition + 1][11]) == int(0):
            df.loc[i + j + nameposition, "cv_ma5_rate"] = 0
        else:
            df.loc[i + j + nameposition, "cv_ma5_rate"] = abs(df.values[i + j + nameposition][11] / df.values[i + j + nameposition + 1][11] - 1) * 100


def cv5d_diff_rate(df, start_date, term, nameposition):  # N일간의 종가 상승률을 (N-1)번째 날의 값으로 설정
    if start_date == "20171222":
        start_date = int(start_date) - 1
    start_date = int(start_date) + 1  # n-1번째 날이어야하니까 하나를 올려 계산
    while 1: # 주말 검출시 하루씩 밀어 계산
        if int(start_date) in df.basic_date.values:
            break
        else:
            start_date = int(start_date) + 1
    for i in range(int(term) + 1):
        if i == 0:
            for j in range(len(df)):
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    if start_date == 20171222:
                        j -= 1
                    break
        if (i + j + nameposition > 476490 - 7) or (i + j > 230 - 6):
            break
        if i + j + nameposition == -1:
            continue
        if int(df.values[i + j + nameposition + 5][6]) == int(0):
            df.loc[i + j + nameposition, "cv5d_diff_rate"] = 0
        else:
            df.loc[i + j + nameposition + 1, "cv5d_diff_rate"] = abs(df.values[i + j + nameposition][6] / df.values[i + j + nameposition + 5][6] - 1) * 100


def ud_5d(df, start_date, term, nameposition):
    if start_date == "20171222":
        start_date = int(start_date) - 1
    start_date = int(start_date) + 1  # n-1번째 날이어야하니까 하나를 올려 계산
    while 1: # 주말 검출시 하루씩 밀어 계산
        if int(start_date) in df.basic_date.values:
            break
        else:
            start_date = int(start_date) + 1
    for i in range(int(term) + 1):
        if i == 0:
            for j in range(len(df)):
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    if start_date == 20171222:
                        j -= 1
                    break
        if (i + j + nameposition > 476490 - 7) or (i + j > 230 - 6):
            break
        if i + j + nameposition == -1:
            continue
        if ((df.loc[i + j + nameposition, "cv_diff_value"] > 0) and
                (df.loc[i + j + nameposition + 1, "cv_diff_value"] > 0) and
                (df.loc[i + j + nameposition + 2, "cv_diff_value"] > 0) and
                (df.loc[i + j + nameposition + 3, "cv_diff_value"] > 0) and
                (df.loc[i + j + nameposition + 4, "cv_diff_value"] > 0)):  # 5일 연속 종가 상승할때 i+1번째 날의 값 1
            df.loc[i + j + nameposition + 1, "ud_5d"] = 1
        elif ((df.loc[i + j + nameposition, "cv_diff_value"] < 0) and
              (df.loc[i + j + nameposition + 1, "cv_diff_value"] < 0) and
              (df.loc[i + j + nameposition + 2, "cv_diff_value"] < 0) and
              (df.loc[i + j + nameposition + 3, "cv_diff_value"] < 0) and
              (df.loc[i + j + nameposition + 4, "cv_diff_value"] < 0)):  # 5일 연속 종가 하락할때 i+1번째 날의 값 -1
            df.loc[i + j + nameposition + 1, "ud_5d"] = -1
        else:  # i+1번째 날의 값 0
            df.loc[i + j + nameposition + 1, "ud_5d"] = 0

def  vv_diff_value(df, start_date, term, nameposition):  #거래량 일간 변화량
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 3) or (i + j > 230 - 2):
            break
        df.loc[i + j + nameposition, "vv_diff_value"] = df.values[i + j + nameposition][7] - df.values[i + j + nameposition + 1][7]

def vv_diff_rate(df, start_date, term, nameposition):  # 거래량 일간 변화율
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 3) or (i + j > 230 - 2):
            break
        if int(df.values[i + j + nameposition + 1][7]) == int(0):
            df.loc[i + j + nameposition, "vv_diff_rate"] = 0
        else:
            df.loc[i + j + nameposition, "vv_diff_rate"] = abs(df.values[i + j + nameposition][7] / df.values[i + j + nameposition + 1][7] - 1) * 100

def vv_maN_value(df, start_date, term, nameposition):  # 거래량의 5일 이동평균
    for i in range(int(term) + 1):
        if i == 0:
            for j in range(len(df)):
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 7) or (i + j > 230 - 5):
            break
        df.loc[i + j + nameposition, "vv_maN_value"] = (df.values[i + j + nameposition][7] +
                                                        df.values[i + j + nameposition + 1][7] +
                                                        df.values[i + j + nameposition + 2][7] +
                                                        df.values[i + j + nameposition + 3][7] +
                                                        df.values[i + j + nameposition + 4][7]) / 5

def vv_maN_rate(df, start_date, term, nameposition):
    for i in range(int(term)):
        if i == 0:
            for j in range(len(df)):
                if str(df.loc[j + nameposition, "basic_date"]) == str(start_date):
                    break
        if (i + j + nameposition > 476490 - 7) or (i + j > 230 - 6):
            break
        if int(df.values[i + j + nameposition + 1][17]) == int(0):
            df.loc[i + j + nameposition, "vv_maN_rate"] = 0
        else:
            df.loc[i + j + nameposition, "vv_maN_rate"] = abs(df.values[i + j + nameposition][17] / df.values[i + j + nameposition + 1][17] - 1) * 100

if __name__ == "__main__":

    df = pandas.read_csv('stock_history.csv', encoding='CP949')  # basic_date, stockname, stock_code, open_value, high_value, low_value, close_value, volume_value
    for i in range(8, 14):
        del df["Unnamed: " + str(i)]
    nameposition = 0

    while 1:
        start_date = input("시작 일을 입력하시오(ex . 20171222) : ")
        if int(start_date) in df.basic_date.values:
            break
        else:
            print("해당 일자의 주식 정보가 없습니다. 확인후 이용해 주시기 바랍니다.")

    term = input("학습 기간을 입력하시오(주말 제외) : ")
    companyname = input("회사를 입력하세요 : ")

    for k in range(len(df)):
        if df.loc[k,"stockname"] == str(companyname):
            nameposition = k
            break

    df["bias"] = 1
    cv_diff_value(df, start_date, term, nameposition)
    cv_diff_rate(df, start_date, term, nameposition)
    cv_ma5_value(df, start_date, term, nameposition)
    cv_ma5_rate(df, start_date, term, nameposition)
    cv5d_diff_rate(df, start_date, term, nameposition)
    ud_5d(df, start_date, term, nameposition)
    vv_diff_value(df, start_date, term, nameposition)
    vv_diff_rate(df, start_date, term, nameposition)
    vv_maN_value(df, start_date, term, nameposition)
    vv_maN_rate(df, start_date, term, nameposition)

    df.to_csv('stock_history_added.csv', encoding='CP949')

    df = df.dropna(axis=0)

    dfx = df[["bias", "cv_diff_value", "cv_diff_rate", "cv_ma5_value", "cv_ma5_rate" ,"ud_5d"]]
    dfy = df[["cv5d_diff_rate"]]

    dfx = dfx.values
    dfy = dfy.values

    # 15.3 모델학습하기
    random.seed(0)
    # beta = estimate_beta(dfx, dfy) # [30.63, 0.972, -1.868, 0.911]
    # print("beta", beta)
    # scikit-learn을 쓴다면: fit_intercept=False : 알파를 베타의 첫항목으로 계산
    myreg = LinearRegression(False).fit(dfx, dfy)
    print("beta of LR:", myreg.coef_)

    # 15.5 적합성(Goodness of fit)
    print("r-squared", multiple_r_squared(dfx, dfy, myreg.coef_[0]))
    print()

    """

    # 15.6 bootstrap
    print("digression: the bootstrap")
    # 101 points all very close to 100
    close_to_100 = [99.5 + random.random() for _ in range(101)]

    # 101 points, 50 of them near 0, 50 of them near 200, 1 of them near 100
    far_from_100 = ([99.5 + random.random()] +
                    [random.random() for _ in range(50)] +
                    [200 + random.random() for _ in range(50)])


    print("bootstrap_statistic(close_to_100, median, 10):")
    print(bootstrap_statistic(close_to_100, median, 10))
    print("bootstrap_statistic(far_from_100, median, 10):")
    print(bootstrap_statistic(far_from_100, median, 10))
    print()

    # 15.7 계수의 표준오차 (skip)
    random.seed(0) # so that you get the same results as me
    bootstrap_betas = bootstrap_statistic(list(zip(x, daily_minutes_good)),
                                          estimate_sample_beta,
                                          100)

    bootstrap_standard_errors = [
        standard_deviation([beta[i] for beta in bootstrap_betas])
        for i in range(4)]

    print("bootstrap standard errors", bootstrap_standard_errors)
    print()

    print("p_value(30.63, 1.174)", p_value(30.63, 1.174))
    print("p_value(0.972, 0.079)", p_value(0.972, 0.079))
    print("p_value(-1.868, 0.131)", p_value(-1.868, 0.131))
    print("p_value(0.911, 0.990)", p_value(0.911, 0.990))
    print()


    # 15.8 Regularization
    print("regularization")
    # (1) 교재방법
    print("(1) by Textbook")
    random.seed(0)
    for alpha in [0.0, 0.01, 0.1, 1, 10, 20, 50]:
        beta = estimate_beta_ridge(x, daily_minutes_good, alpha=alpha)
        print("alpha", alpha)
        print("beta", beta)
        # print("dot(beta[1:],beta[1:])", dot(beta[1:], beta[1:]))
        print("r-squared", multiple_r_squared(x, daily_minutes_good, beta))
        print()
    # (2) scikit-liearn 사용방법
    print("(2) by scikit-learn")
    for alpha in [0.0, 0.01, 0.1, 1, 10, 20, 50]:
        ridge_reg = Ridge(alpha, fit_intercept=False, solver="cholesky")
        ridge_reg.fit(x, daily_minutes_good)
        beta = ridge_reg.coef_
        print("alpha", alpha)
        print("beta", beta)
        print("r-squared", multiple_r_squared(x, daily_minutes_good, beta))
    """