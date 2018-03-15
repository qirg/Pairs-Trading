import numpy as np
import quandl
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import math
from scipy.optimize import minimize

'''
Roughly follows 'Pairs Trading - Quantitative Methods and Analysis'
'''

'''
TODO:
- Johansen Cointegration
- Hurst Exponent
- Extended Pairs Trading (Multi-Asset Cointegration - Johansen)
- Kalman Filter
- Slippage Model (Fee Structure and Liquidity Model)
- Automate the process of finding an optimal delta (bootstrap sampling / cross-validation)
- Automate cointegrated pair finding
- Maximize profit and trade success rate
'''

'''
NOTES:
- It typically doesn't matter whether we use log or regular stock prices for model analysis
'''


quandl.ApiConfig.api_key = 'your key'
quandl.ApiConfig.api_version = '2015-04-09'


pair = [('ADBE', 'MSFT'), ('INTC', 'MSFT')]

adbe = quandl.get('WIKI/ADBE', start_date='2007-01-01', end_date='2017-01-01')
msft = quandl.get('WIKI/MSFT', start_date='2007-01-01', end_date='2017-01-01')

adbe_close = np.array(adbe['Close'])
msft_close = np.array(msft['Close'])

adbe_log_close = np.log(adbe_close)
msft_log_close = np.log(msft_close)



'''
from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(X, Y)
print(lm.coef_, lm.intercept_)
'''



## ------------------------------ ##
result = ts.coint(adbe_log_close, msft_log_close)
print('Cointegration Test:', result)
print('Is there cointegration:', (True if result[1] < 0.05 else False)) # 95% confidence level

'''
Testing Tradability
'''

# Have to fix gamma calculation so that it matches the regression method
Y = adbe_log_close
X = sm.add_constant(msft_log_close)
model = sm.OLS(Y, X).fit()
print('New stuff...', model.params, model.tvalues)
print('Gamma', model.params[1])
print(model.summary())
# Multifactor Approach ----------------------
# A - adbe, B - msft
gamma = (np.cov(adbe_log_close, msft_log_close)/np.var(msft_log_close))[0, 1]
print('Proper Gamma:', gamma)
gamma_prime = (np.cov(adbe_log_close, msft_log_close)/np.var(adbe_log_close))[0, 1]
print('Proper Gamma Prime:', gamma_prime)
# Choose the bigger out of gamma and gamma price - reduces precision error (page 108)
gamma = max(gamma, gamma_prime)

resid = adbe_log_close - gamma * msft_log_close
result = ts.adfuller(resid, maxlag=1, regression='c')
print(result)

mu = resid.mean()
delta = 0.1

up = mu + delta
down = mu - delta


# long 1 share of A (adbe) and short gamma shares of B (msft)

# Regression approach ----------------------
# slope is gamma, intercept is mu (equilibrium)
print('Slope:', model.params[1]) # cointegration coefficient
print('Intercept:', model.params[0]) # premium

print(resid.mean())


def optimization(capital, k):

    shares_msft = math.floor(capital / msft_close[k]) # min
    shares_adbe = math.floor(capital / adbe_close[k]) # max

    def fun(x):
        return (x[0] * adbe_close[k] - gamma * x[1] * msft_close[k])

    def con1(x):
        return x[0] * adbe_close[k] - gamma * x[1] * msft_close[k] # >= 0

    def con2(x):
        return capital - (x[0] * adbe_close[k] + x[1] * msft_close[k])

    cons = ({'type': 'ineq', 'fun': con1},
            {'type': 'eq', 'fun': con2})
    bnds = [(0, shares_adbe), (0, shares_msft)] # should be (0, shares)

    optimal = minimize(fun, [100, 100], method='SLSQP', bounds=bnds, constraints=cons)['x']
    print('optimal parameters:', optimal, optimal[0], long, gamma, optimal[1], short_price)

    # need integer values
    optimal[0] = math.floor(optimal[0]) # ADBE
    optimal[1] = math.floor(optimal[1]) # MSFT
    return optimal

## ----- STRATEGY STARTS HERE ----- ##

'''
Long if spread is below mu - delta
Short if spread is above mu + delta
'''

capital = 10000
init_capital = capital
open = False
long = False
short = False
position = []
revenue = []
profit = []
# Assume trading fees are $5 each way
# Comment out the reinvestment part below if you do not wish to reinvest
for i in range(len(resid)):
    if not open and resid[i] <= down: # long the spread
        long_price = adbe_close[i]
        short_price = msft_close[i]
        params = optimization(capital, i)
        position.append({'share_adbe': params[0],
                         'share_msft': params[1],
                         'price_adbe': long_price,
                         'price_msft': short_price})
        open = True
        long = True
    elif open and long and resid[i] > mu:
        old_pos = position[-1]['share_adbe'] * position[-1]['price_adbe'] - \
                  gamma * position[-1]['share_msft'] * position[-1]['price_msft']
        new_pos = position[-1]['share_adbe'] * adbe_close[i] - gamma * position[-1]['share_msft'] * msft_close[i]
        revenue.append(new_pos - old_pos)
        profit.append(new_pos - old_pos - 10)
        #capital += profit[-1] # reinvestment
        open = False
        long = False

    if not open and resid[i] >= up: # short the spread
        long_price = adbe_close[i]
        short_price = msft_close[i]
        params = optimization(capital, i)
        position.append({'share_adbe': params[0],
                         'share_msft': params[1],
                         'price_adbe': long_price,
                         'price_msft': short_price})
        open = True
        short = True
    elif open and short and resid[i] < mu:
        old_pos = position[-1]['share_adbe'] * position[-1]['price_adbe'] - \
                  gamma * position[-1]['share_msft'] * position[-1]['price_msft']
        new_pos = position[-1]['share_adbe'] * adbe_close[i] - gamma * position[-1]['share_msft'] * msft_close[i]
        revenue.append(old_pos - new_pos)
        profit.append(old_pos - new_pos - 10)
        #capital += profit[-1] # reinvestment
        open = False
        short = False

print('Revenue:', revenue)
print('Total Revenue:', np.sum(revenue))
print('Profit:', profit)
print('Total Profit:', np.sum(profit))
print('Trading Success Rate:', str(len([*filter(lambda x: x > 0, profit)]) / len(profit) * 100) + '%')
roi = (np.sum(profit))/init_capital
print('ROI:', str(roi * 100) + '%')
print('Long:', long, '| Short:', short) # indicates if there is an open position currently

def max_drawdown(vec):
    maximums = np.maximum.accumulate(vec)
    drawdowns = 1 - vec / maximums
    return np.max(drawdowns)

profit = np.array(profit)
print('Max Drawdown:', max_drawdown(profit))












