import numpy as np

def posterior(X, y, m0, S0, sigma2):
    SN = np.linalg.inv(np.linalg.inv(S0)+(1/sigma2)*X.T.dot(X))
    mN = SN.dot(np.linalg.inv(S0).dot(m0)+(1/sigma2)*X.T.dot(y))
    return mN, SN

def sequential_posterior(X, y, m0, S0, sigma2):
    # return (mN, SN, means)
    means = []
    mN,SN = m0.copy(),S0.copy()
    for i in range(len(X)):
        mN,SN = posterior(X[i:i+1],y[i:i+1],mN,SN,sigma2)
        means.append(mN.copy())
    return mN,SN,np.array(means)


def predictive(X_test, mN, SN, sigma2):
    # return (mean, var)
    mean = mN.dot(X_test.T)
    var = sigma2 + np.diag(X_test.dot(SN.dot(X_test.T)))
    return mean, var
