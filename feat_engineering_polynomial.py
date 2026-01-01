import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2) # reduce display precision on numpy arrays

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features 
X = x**2      #<-- added engineered feature


X = X.reshape(-1, 1)  #X should be a 2-D Matrix
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)
print(f"These model_w and model_b values are the optimal values for the model y = w * x^2 + b")

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value" , color="blue"); plt.xlabel("x"); plt.ylabel("y"); plt.legend();
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# now to try y = w0x0 + w1x1^2 + w2x2^3 + b

x = np.arange(0, 20, 1)
y = x**2

# enginering the features
X = np.c_[x, x**2, x**3]   #<-- added engineered feature, 
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-7)
print(f"These model_w and model_b values are the optimal values for the model y = w0*x + w1*x^2 + w2*x^3 + b")
plt.scatter(x, y , marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, np.dot(X, model_w) + model_b, label="Predicted Value" , color="blue"); plt.xlabel("x"); plt.ylabel("y"); plt.legend();
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
"""
 i got These model_w and model_b values are the optimal values for the model y = w0x0 + w1x1^2 + w2x2^3 + b
 this means the feature x1^2 is the most important feature for the model, followed by x2^3 and then x0.
"""


# An alternate View, 
"""
Above, polynomial features were chosen based on how well they matched the target data. Another way to think about this is to note that we are still using linear regression once we have created new features. Given that, the best features will be linear relative to the target. This is best understood with an example.
"""

x = np.arange(0, 20, 1)
y = x **2
# feature engineering
X = np.c_[x, x**2, x**3]
X_features = ['x','x^2','x^3']
fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True) # this is a 1x3 plot, with the y axis shared between the plots
for i in range (len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()

# this is a 1x3 plot, with the y axis shared between the plots
# the x axis is the feature, and the y axis is the target
# the plots are x, x^2, and x^3
# the plots are shared on the y axis
# the plots are labeled with the feature name
# the plots are titled with the feature name
# the plots are shown

"""
THe x^2 feature is the best feature for the model, followed by x^3 and then x.
It is linear relative to the target.
The 3 feastures have different slopes/scales, so we need to scale the features.
"""

# create target data, with features engineered
x = np.arange(0, 20, 1)
y = np.c_[x, x**2, x**3]

print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}") 
print(X)

# normalize the features by column using z-score normalization
X = zscore_normalize_features(X)
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")


# now try again with alfa = 1e-1
x = np.arange(0, 20, 1)
y = x **2
X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X)


model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha = 1e-1)
print(f"These model_w and model_b values are the optimal values for the model y = w0*x + w1*x^2 + w2*x^3 + b")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value after normalization" , color="blue"); plt.xlabel("x"); plt.ylabel("y"); plt.legend();
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

"""
 complex functions
"""


x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value" , color="blue"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


"""
Congratulations! You have completed the optional lab on feature engineering and polynomial regression.
"""
