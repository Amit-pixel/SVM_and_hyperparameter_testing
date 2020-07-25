import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn
from keras.datasets import mnist
from sklearn import metrics

##############################################################################################################################################
##############################################################################################################################################
# Function from HW2:
def cross_validation_error(X, y, model, folds):
    avg_train_error = 0 
    avg_val_error = 0
    kf = KFold(n_splits=folds, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        fit_model = model.fit(X_train, y_train)
        X_train_pred = fit_model.predict(X_train)
        X_val_pred = fit_model.predict(X_val)
        avg_train_error += 1 - accuracy_score(X_train_pred, y_train)
        avg_val_error += 1 - accuracy_score(X_val_pred, y_val)
    return [avg_train_error / folds, avg_val_error / folds]


##############################################################################################################################################
##############################################################################################################################################
# Q4
# d.
# Loading breast cancer dataset:
X, y = datasets.load_breast_cancer(return_X_y=True)
X = np.array(X)
y = np.array(y)
X = X[50:200, [1,3]]
y = y[50:200]
y[y == 0] = -1

# Splitting the data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)

##############################################################################################################################################
# e.
# Function for plotting (from the tutorial)
def plot_svc_decision_function(model, ax=None, plot_support=True, color='black', label=None, title=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1])
    y = np.linspace(ylim[0], ylim[1])
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    CS = ax.contour(X, Y, P, colors=color,
                   levels=[0], alpha=0.5,
                   linestyles=['-'])
    if label:
        CS.collections[0].set_label(label)

    if title:
        ax.set_title(title)
        
    # plot support vectors
    if plot_support:
        # plot decision boundary and margins
        ax.contour(X, Y, P, colors=color,
                   levels=[-1, 1], alpha=0.5,
                   linestyles=['--', '--'])
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=50, linewidth=1, facecolors='none', edgecolor=color)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


train_error = [None]*5
test_error = [None]*5
margin_width = [None]*5

# Plotting and fitting the models:
lambdas = [0.00001, 0.001, 0.1, 10, 10000]
for i, v in enumerate(lambdas):
    # Fitting the model:
    m = len(y_train)
    model = SVC(C = 1/(2*m*v), kernel='linear').fit(X_train, y_train)
    model.fit(X_train, y_train)
    # Plotting the train data:
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='autumn')
    plot_svc_decision_function(model, label='\u03BB={0}'.format(v), title='SVM for Training Data')
    plt.legend()
    plt.show()
    # Calculating train errors:
    pred = model.predict(X_train)
    train_error[i] = 1-metrics.accuracy_score(y_train, y_pred=pred)
    # Calculating the margin width:
    w = model.coef_[0]
    margin_width[i] = 2/np.linalg.norm(w)

    # Plotting the test data:
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='autumn')
    plot_svc_decision_function(model, plot_support=False, label='\u03BB={0}'.format(v), title='SVM for Test Data')
    plt.legend()
    plt.show()
    # Calculating test errors:
    pred = model.predict(X_test)
    test_error[i] = 1-metrics.accuracy_score(y_test, y_pred=pred)


##############################################################################################################################################
# f.
# Plotting the train and test error for each model:
labels = ('\u03BB=0.00001', '\u03BB=0.001', '\u03BB=0.1', '\u03BB=10', '\u03BB=10000')
X = np.arange(5)
fig, ax = plt.subplots()
ax.bar(X + 0.00, train_error, color = 'b', width = 0.25, label = 'Train Error')
ax.bar(X + 0.25, test_error, color = 'r', width = 0.25, label = 'Test Error')
leg = ax.legend()
ax.set_title('Errors vs SVM lambdas')
plt.xlabel('lambdas')
plt.ylabel('Errors')
ax.set_xticks(X)
ax.set_xticklabels(labels)
plt.grid(True)

for i in range(5):
    plt.text(i-0.1 , train_error[i] + 0.005, color ='b', s=str(np.around(train_error[i], decimals=2)))
    plt.text(i+0.15, test_error[i] + 0.005, color = 'r', s=str(np.around(test_error[i], decimals=2)))

fname='D:\ML\HW3\plot4f1.png'
# plt.savefig(fname)
plt.show()


# Plotting the margin width:
labels = ('\u03BB=0.00001', '\u03BB=0.001', '\u03BB=0.1', '\u03BB=10', '\u03BB=10000')
X = np.arange(5)
fig, ax = plt.subplots()
ax.bar(X + 0.00, margin_width, color = 'b', width = 0.25, label = 'Train Error')
leg = ax.legend()
ax.set_title('Margin Width vs SVM lambdas')
plt.xlabel('lambdas')
plt.ylabel('Margin Width')
ax.set_xticks(X)
ax.set_xticklabels(labels)
ax.set_yscale('log')
plt.grid(True)

for i in range(5):
    plt.text(i-0.1 , margin_width[i] + 0.005, color ='b', s=str(np.around(margin_width[i], decimals=2)))

fname='D:\ML\HW3\plot4f2.png'
# plt.savefig(fname)
plt.show()


##############################################################################################################################################
##############################################################################################################################################
# Q5
# a. 
# Fitting SVM models and returning train, validation and test errors:
def SVM_results(X_train, y_train, X_test, y_test):
    gammas = [0.001,0.01,0.1,1,10]
    degrees = [2,4,6,8,10]
    train_error = [None]*11
    val_error = [None]*11
    test_error = [None]*11

    # Linear Kernel:
    [train_error[0], val_error[0]] = cross_validation_error(X_train, y_train, SVC(kernel='linear'), 5)
    fit_model = SVC(kernel='linear').fit(X_train, y_train)
    test_pred = fit_model.predict(X_test)
    test_error[0] = 1 - accuracy_score(test_pred, y_test)

    # Polynomial Kernel:
    for i in range(len(degrees)):
        [train_error[i+1], val_error[i+1]] = cross_validation_error(X_train, y_train, SVC(kernel='poly', degree=degrees[i]), 5)
        fit_model = SVC(kernel='poly', degree=degrees[i]).fit(X_train, y_train)
        test_pred = fit_model.predict(X_test)
        test_error[i+1] = 1 - accuracy_score(test_pred, y_test)

    # RBF Kernel:
    for i in range(len(gammas)):
        [train_error[i+6], val_error[i+6]] = cross_validation_error(X_train, y_train, SVC(kernel='rbf', gamma=gammas[i]), 5)
        fit_model = SVC(kernel='rbf', gamma=gammas[i]).fit(X_train, y_train)
        test_pred = fit_model.predict(X_test)
        test_error[i+6] = 1 - accuracy_score(test_pred, y_test)
    
    return {'svm_linear_C_default':     [train_error[0], val_error[0], test_error[0]],

            'svm_polynomial_degree_2':  [train_error[1], val_error[1], test_error[1]],
            'svm_polynomial_degree_4':  [train_error[2], val_error[2], test_error[2]],
            'svm_polynomial_degree_6':  [train_error[3], val_error[3], test_error[3]],
            'svm_polynomial_degree_8':  [train_error[4], val_error[4], test_error[4]],
            'svm_polynomial_degree_10': [train_error[5], val_error[5], test_error[5]],
    
            'svm_rbf_gamma_0.001':      [train_error[6], val_error[6], test_error[6]], 
            'svm_rbf_gamma_0.01':       [train_error[7], val_error[7], test_error[7]], 
            'svm_rbf_gamma_0.1':        [train_error[8], val_error[8], test_error[8]],
            'svm_rbf_gamma_1':          [train_error[9], val_error[9], test_error[9]],
            'svm_rbf_gamma_10':         [train_error[10],val_error[10],test_error[10]]}


##############################################################################################################################################
# b.
# Loading data:
def load_mnist():
    np.random.seed(2)
    (X, y), (_, _) = mnist.load_data()
    indexes = np.random.choice(len(X), 8000, replace=False)
    X = X[indexes]
    y = y[indexes]
    X = X.reshape(len(X), -1)
    return X, y

# Splitting data:
[X, y] = load_mnist()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state=98)


##############################################################################################################################################
# c.
# Applying min-max normalization:
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


##############################################################################################################################################
# d.
# Fitting an SVM model on the scaled training data:
fit_model = SVC(kernel='linear').fit(X_train, y_train)
y_pred = fit_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float')
normalized_cm = cm / cm.sum(axis=1)[:, np.newaxis]
sn.heatmap(normalized_cm, annot=True, cmap="tab20b")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Normalized confusion matrix')
fname='D:\ML\HW3\plothw5d.png'
# plt.savefig(fname)
plt.show()


##############################################################################################################################################
# e.
# Plotting the errors as function of SVM model:
info = SVM_results(X_train, y_train, X_test, y_test)

train_error = []
val_error = []
test_error = []
for i in info.keys():
    train_error.append(info[i][0])
    val_error.append(info[i][1])
    test_error.append(info[i][2])

labels = ("Lin", "Poly_2", "Poly_4", "Poly_6", "Poly_8", "Poly_10", "RBF_0.001", "RBF_0.01", "RBF_0.1", "RBF_1", "RBF_10")
X = np.arange(11)
fig, ax = plt.subplots()
ax.bar(X + 0.00, train_error, color = 'b', width = 0.25, label = 'Train Error')
ax.bar(X + 0.25, val_error, color = 'g', width = 0.25, label = 'Validation Error')
ax.bar(X + 0.50, test_error, color = 'r', width = 0.25, label = 'Test Error')
leg = ax.legend()
ax.set_title('Errors vs SVM models')
plt.xlabel('SVM models')
plt.ylabel('Errors')
ax.set_xticks(X)
ax.set_xticklabels(labels)
plt.grid(True)

for i in range(11):
    plt.text(i-0.25 , train_error[i] + 0.01, color ='b', s=str(np.around(train_error[i], decimals=2)))
    plt.text(i+0.05 , val_error[i] + 0.01, color = 'g', s=str(np.around(val_error[i], decimals=2)))
    plt.text(i+0.40, test_error[i] + 0.01, color = 'r', s=str(np.around(test_error[i], decimals=2)))

fname='D:\ML\HW3\plot5e.png'
# plt.savefig(fname)
plt.show()

# Which model performed best on the test data?
# How do you explain it?
# Were we able to predict it during the training procedure from the cross-validation results?