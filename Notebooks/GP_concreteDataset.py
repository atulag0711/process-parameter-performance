#!/usr/bin/env python
# coding: utf-8

# In[775]:


# https://www.analyticsvidhya.com/blog/2021/04/concrete-strength-prediction-using-machine-learning-with-python-code/
# https://www.kaggle.com/elikplim/concrete-compressive-strength-data-set
# https://pyro.ai/examples/gp.html
# https://docs.pyro.ai/en/stable/contrib.gp.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from tqdm import tqdm

import torch

import os
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(0)

from datetime import datetime

import matplotlib as mpl
from matplotlib import rc
mpl.rcParams['font.family'] = ['times new roman'] # default is sans-serif
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
datetime = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


# - Gaussian Process Regression for concrete mix design parameters and concrete property (compressive strength).
# - Input feature space is 8 dimentional.
# - RBF kernel is used
# - Number of training data points is N ~ 700
# - Number of test data points is ~ 300.
# - Once learned the GP will act as surrogate.
#
#

# ## Reading and Analyzing Experimental Data
df = pd.read_csv('../Data/concrete_data.csv')

df['age']= np.log(df['age'])


df.describe()

sb.pairplot(df)
plt.savefig('./Figs/Pairplot',format='pdf')

sb.heatmap(df.corr(),annot=True)
#plt.tight_layout()
plt.savefig('Correlation_heatmap',format='pdf')

# independent variables
x = df.drop(['concrete_compressive_strength'],axis=1)
# dependent variables
y = df['concrete_compressive_strength']


# ## Gaussian Process Regression
# - log normal prior for RBF kernel variance and length scale
# - inferred these by optimising ELBO with ADAM.
# - ARD type prior for length scale ie $\textbf{L} = diag(l_i)$ and inverse gamma prior for each

from sklearn.model_selection import train_test_split
# TODO: k fold cross validation test.
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=42)


class TorchStandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x
    def inverse_transform(self,x):
        tmp = x*self.std + self.mean
        return tmp

scalar = TorchStandardScaler()
Fit = scalar.fit(torch.tensor(np.array(xtrain)))
xtrain_scl = scalar.transform(torch.tensor(np.array(xtrain)))
xtest_scl = scalar.transform(torch.tensor(np.array(xtest)))


# testing for different length scale for feature space
feature_space_input_dim = np.shape(xtrain_scl)[1]
kernel = gp.kernels.RBF(input_dim=feature_space_input_dim, variance=torch.tensor(1.),
                        lengthscale=torch.ones(feature_space_input_dim))
gpr = gp.models.GPRegression(xtrain_scl, torch.tensor(ytrain.array), kernel, noise=torch.tensor(1.))
#gpr = gp.models.GPRegression(torch.tensor(xtrain_scl), torch.tensor(ytrain.array), kernel, noise=torch.tensor(1.)) # this is observational noise


# note that our priors have support on the positive reals
# https://pyro.ai/examples/tensor_shapes.html
# https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/
# to event converts to multivariate

# choosing hyperprior. Inverse gamma for length scale and gamma for inverse of lengthscale. Gamma promoted l --> inf thus weeding out input which doesnt affect output.
gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.InverseGamma(2*torch.ones(feature_space_input_dim), 1.0).to_event())
gpr.kernel.variance = pyro.nn.PyroSample(dist.InverseGamma(2.0, 1.0))
#gpr.noise = pyro.nn.PyroSample(dist.Uniform(1.0, 30.0))
gpr.noise = pyro.nn.PyroSample(dist.InverseGamma(2.0, 1.0))


optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=1)
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
grad = []
num_steps = 1000 if not smoke_test else 2

for epoch in range(6):
    for i in tqdm(range(num_steps)):
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        tmp = 0
        # Note these are reparametrized values
        for p in gpr.parameters():
            #print(p.grad)
            tmp+=torch.norm(p.grad)**2
        grad.append(tmp.sqrt())
    scheduler.step()


def training_diagnostics(obj_value,Yaxis_obj, grad,Yaxis_grad, save = False, path =None, Name = None):
    """
    """
    fig = plt.figure(figsize=(12, 6))
    #plt.subplots_adjust(wspace=0.3, hspace=0.5)

    # plot the sampled values as functions of x_star
    plt.subplot(2, 2, 1)
    plt.semilogy(obj_value);
    plt.xlabel("iteration")
    plt.ylabel(Yaxis_obj)

    # visualise the covariance function
    plt.subplot(2, 2, 2)
    plt.semilogy(grad);
    plt.xlabel("iteration")
    plt.ylabel(Yaxis_grad)

    if save == True:
        plt.savefig(path + Name,format='pdf')


training_diagnostics(losses,"ELBO",grad,r"$||\nabla ELBO||_2$", save = True,path ='./Figs/',Name = 'GP training Diagnostics')



# tell gpr that we want to get samples from guides
# This is the learned parameters.
gpr.set_mode('guide')
print('variance = {}'.format(gpr.kernel.variance))
print('lengthscale = {}'.format(gpr.kernel.lengthscale))
print('noise = {}'.format(gpr.noise))


# ## Prediction and Visualization


y_pred_mean, y_pred_cov = gpr(xtest_scl, full_cov =True, noiseless = False) # Remove or include observational noise in prediction?

plt.contourf(y_pred_cov.detach().numpy())
plt.title("Posterior Covarience")


n_test=15
x = np.arange(np.shape(ytest[:n_test])[0])
plt.plot(x,ytest[:n_test],'r*',label="Exp comp strength")

plt.plot(x, y_pred_mean.detach().numpy()[:n_test], '-o',label="GP Pred comp strength",ls='--')  # plot the mean

sd = y_pred_cov.diag().sqrt()
plt.fill_between(x,  # plot the two-sigma uncertainty about the mean
                 (y_pred_mean - 2.0 * sd).detach().numpy()[:n_test],
                 (y_pred_mean + 2.0 * sd).detach().numpy()[:n_test],
                 color='C0', alpha=0.3)
plt.legend()
plt.xlabel("New experiment No.")
plt.ylabel("Compressive strength")
plt.title("Comparing predicted compressive strength for new experiments")



plt.plot(ytest,y_pred_mean.detach_().numpy(),"*")
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
plt.ylabel('compressive strength (predicted)')
plt.xlabel('compressive strength (experiment)')
#plt.savefig('./Figs/Observed_vs_predicted',format='pdf')


from sklearn.metrics import r2_score
cod = r2_score(ytest,y_pred_mean.detach().numpy())
print ("The coefficient of Determination is %2.3f" % cod)


# ### Walz curve predictions and comparrison with test data and the GP surrogate

# w/z < 1 is useless. So chucking those out. Also keeping just 28 days strength as walz curve is just defined for that

idx_N28_w_z_1 = np.where((np.array(xtest['age'])==np.log(28)) & (np.array(xtest['water'])/np.array(xtest['cement'])<1))[0]


# choosing which walz will represent the data in best possible way
N28 = [62.5,52.5,42.5]
y_walz_allclass = {
'62.5': N28[0]*(2.823-4.785*(np.array(xtest['water'])[idx_N28_w_z_1]/np.array(xtest['cement'])[idx_N28_w_z_1])+2.334*(np.array(xtest['water'])[idx_N28_w_z_1]/np.array(xtest['cement'])[idx_N28_w_z_1])**2),
'52.5': N28[1]*(2.587-4.109*(np.array(xtest['water'])[idx_N28_w_z_1]/np.array(xtest['cement'])[idx_N28_w_z_1])+1.873*(np.array(xtest['water'])[idx_N28_w_z_1]/np.array(xtest['cement'])[idx_N28_w_z_1])**2),
'42.5': N28[2]*(2.769-4.581*(np.array(xtest['water'])[idx_N28_w_z_1]/np.array(xtest['cement'])[idx_N28_w_z_1])+2.154*(np.array(xtest['water'])[idx_N28_w_z_1]/np.array(xtest['cement'])[idx_N28_w_z_1])**2)
}


for i,v in y_walz_allclass.items():
    cod_walz = r2_score(np.array(ytest)[idx_N28_w_z_1],v)
    print ("The coefficient of Determination for N28 %s is %2.3f" % (i, cod_walz))

# observed vs pred figure for walz curve
plt.plot(np.array(ytest)[idx_N28_w_z_1],y_walz_allclass['52.5'],"m*", label = 'walz curve prediction (N28 = 52.5), COD = 0.371')
plt.plot(np.array(ytest)[idx_N28_w_z_1],y_pred_mean.detach_().numpy()[idx_N28_w_z_1],"k*", label = 'GP prediction, COD = 0.89')
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
plt.ylabel('compressive strength (predicted)')
plt.xlabel('compressive strength (experiment)')
plt.legend()
#plt.savefig('./Figs/Observed_vs_walzCurvepredicted',format='pdf')


## for GERMAN
# observed vs pred figure for walz curve
plt.plot(np.array(ytest)[idx_N28_w_z_1],y_walz_allclass['52.5'],"m*", label = 'walz kurve prediction (N28 = 52.5), COD = 0.371')
plt.plot(np.array(ytest)[idx_N28_w_z_1],y_pred_mean.detach_().numpy()[idx_N28_w_z_1],"k*", label = 'GP prediction, COD = 0.89')
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
plt.ylabel('Druckfestigkeit (predicted)')
plt.xlabel('Druckfestigkeit (experiment)')
plt.legend()

plt.savefig('./Figs/Observed_vs_walzCurvepredicted_german'+datetime,format='pdf')


# Predicted vs new exp number (test data)
n_test=15
x = np.arange(np.shape(ytest[:n_test])[0])
plt.plot(x,np.array(ytest)[idx_N28_w_z_1[:n_test]],'y*',label="Exp comp strength")

plt.plot(x, y_pred_mean.detach().numpy()[idx_N28_w_z_1[:n_test]], 'o',label="GP Pred comp strength",color='black')  # plot the mean

sd = y_pred_cov.diag().sqrt()
plt.fill_between(x,  # plot the two-sigma uncertainty about the mean
                 (y_pred_mean - 2.0 * sd).detach().numpy()[idx_N28_w_z_1[:n_test]],
                 (y_pred_mean + 2.0 * sd).detach().numpy()[idx_N28_w_z_1[:n_test]],
                 color='black', alpha=0.1, label="$\pm 2 \sigma$")
plt.plot(x,y_walz_allclass['52.5'][:n_test],'m*',label="walz curve prediction")
plt.legend()
plt.xlabel("New experiment No.")
plt.ylabel("Compressive strength")
plt.title("Comparing predicted compressive strength for new experiments")


#plt.errorbar(x,y_pred_mean.detach().numpy()[idx_N28_w_z_1[:n_test]],yerr=2*sd[idx_N28_w_z_1[:n_test]],fmt='o', color='black',
#             ecolor='lightgray', capsize=3,label="GP Druckfestigkeit")
plt.savefig('./Figs/Posterior_predictive_wrt_walz'+ datetime,format='pdf')


## GERMAN VERSION
# Predicted vs new exp number (test data)
n_test=10
x = np.arange(np.shape(ytest[:n_test])[0])
plt.plot(x,np.array(ytest)[idx_N28_w_z_1[:n_test]],'y*',label="Experimentelle Druckfestigkeit")

#plt.plot(x, y_pred_mean.detach().numpy()[idx_N28_w_z_1[:n_test]], 'o',label="GP Druckfestigkeit",color='black')  # plot the mean

sd = y_pred_cov.diag().sqrt()
# plt.fill_between(x,  # plot the two-sigma uncertainty about the mean
#                  (y_pred_mean - 2.0 * sd).detach().numpy()[idx_N28_w_z_1[:n_test]],
#                  (y_pred_mean + 2.0 * sd).detach().numpy()[idx_N28_w_z_1[:n_test]],
#                  color='black', alpha=0.1, label="$\pm 2 \sigma$")
plt.errorbar(x,y_pred_mean.detach().numpy()[idx_N28_w_z_1[:n_test]],yerr=2*sd[idx_N28_w_z_1[:n_test]],fmt='o', color='black',
             ecolor='lightgray', capsize=3,label="GP Druckfestigkeit")
plt.plot(x,y_walz_allclass['52.5'][:n_test],'m*',label="walz kurve Druckfestigkeit")
plt.legend()
plt.xlabel("Neues experiment No.")
plt.ylabel("Druckfestigkeit")
#plt.title("Comparing predicted compressive strength for new experiments")



plt.savefig('./Figs/Posterior_predictive_wrt_walz_german'+ datetime,format='pdf')

# GP testing with training data
n_test= 50
w_by_c = np.array(xtrain['water'])[:n_test]/np.array(xtrain['cement'])[:n_test]
y_exp_plot = np.array(ytrain)[:n_test]
y_pred_mean_train, y_pred_cov_train = gpr(torch.tensor(xtrain_scl), full_cov =True, noiseless = False)
y_GP_plot = y_pred_mean_train.detach().numpy()[:n_test]
sd = y_pred_cov_train.diag().sqrt()
y_GP_sd_plot = sd.detach().numpy()[:n_test]


idx_tmp = np.argsort(w_by_c)
plt.plot(np.sort(w_by_c),y_exp_plot[idx_tmp],'r*',label="(New)Exp comp strength")
plt.plot(np.sort(w_by_c),y_GP_plot[idx_tmp],'-o',label="GP Pred comp strength")
#sd = np.sqrt(y_GP_var_plot[idx_tmp])

plt.fill_between(np.sort(w_by_c),  # plot the two-sigma uncertainty about the mean
                 (y_GP_plot[idx_tmp] - 2.0 * y_GP_sd_plot[idx_tmp]),
                 (y_GP_plot[idx_tmp] + 2.0 * y_GP_sd_plot[idx_tmp]),
                 color='C0', alpha=0.3)

#plt.errorbar(np.sort(w_by_c),y_GP_plot[idx_tmp],yerr=2*sd,fmt='o', color='blue',
#             ecolor='lightgray', elinewidth=3,alpha = 0.7, label="GP Pred comp strength $\pm 2 \sigma$")
plt.legend()
plt.xlabel("$w/z$")
plt.ylabel("Compressive strength")
plt.title("Comparing compressive strength for w/z")



# GP testing with test data

n_test= 15
w_by_c = np.array(xtest['water'])[idx_N28_w_z_1[:n_test]]/np.array(xtest['cement'])[idx_N28_w_z_1[:n_test]]
y_exp_plot = np.array(ytest)[idx_N28_w_z_1[:n_test]]
y_GP_plot = y_pred_mean.detach().numpy()[idx_N28_w_z_1[:n_test]]
sd = y_pred_cov.diag().sqrt()
y_GP_sd_plot = sd.detach().numpy()[idx_N28_w_z_1[:n_test]]
y_walz_plot = y_walz_allclass['52.5'][:n_test]


idx_tmp = np.argsort(w_by_c)
plt.plot(np.sort(w_by_c),y_exp_plot[idx_tmp],'y*',label="(New)Exp comp strength")
plt.plot(np.sort(w_by_c),y_walz_plot[idx_tmp],'m-X',label = "walz curve prediction")
plt.plot(np.sort(w_by_c),y_GP_plot[idx_tmp],'k-o',label="GP Pred comp strength",ls='--')
#sd = np.sqrt(y_GP_var_plot[idx_tmp])

plt.fill_between(np.sort(w_by_c),  # plot the two-sigma uncertainty about the mean
                 (y_GP_plot[idx_tmp] - 2.0 * y_GP_sd_plot[idx_tmp]),
                 (y_GP_plot[idx_tmp] + 2.0 * y_GP_sd_plot[idx_tmp]),
                 color='black', alpha=0.1)

#plt.errorbar(np.sort(w_by_c),y_GP_plot[idx_tmp],yerr=2*sd,fmt='o', color='blue',
#             ecolor='lightgray', elinewidth=3,alpha = 0.7, label="GP Pred comp strength $\pm 2 \sigma$")
plt.legend()
plt.xlabel("$w/z$")
plt.ylabel("Compressive strength")
plt.title("Comparing compressive strength for w/z")



# -- Optimisationm with grid search methods

# creating mesh grid from X_0 and X_3
x_c = np.linspace(np.min(xtest['cement']),np.max(xtest['cement']),100)
x_w = np.linspace(np.min(xtest['water']),np.max(xtest['water']),100)

X_c, X_w = np.meshgrid(x_c,x_w,indexing='ij')


from scipy.stats import norm
def constraint_new(X):
    x_tmp = Fit.transform(X.reshape(1,-1))
    m,v = gpr(torch.tensor(x_tmp), full_cov =True, noiseless = False)
    p = norm.cdf(50,loc=m.detach().numpy(),scale = np.sqrt(v.detach().numpy()))
    return p


test_exp_no =2
count = 0
x_accept_c = []
x_accept_w = []
xtest_singlexp = np.array(xtest)[test_exp_no,:]
for i in range(x_c.shape[0]):
    for j in range(x_w.shape[0]):

        x_tmp = np.array([X_c[i,j],xtest_singlexp[1],xtest_singlexp[2],X_w[i,j],xtest_singlexp[4],xtest_singlexp[5],xtest_singlexp[6],xtest_singlexp[7]])
        p = constraint_new(x_tmp)
        if p <= 0.05:
            count += 1
            x_accept_c.append(X_c[i,j])
            x_accept_w.append(X_w[i,j])

# choosing min cement values
x_c_star = np.min(x_accept_c)
index = np.where(x_accept_c==x_c_star)
x_w_star = x_accept_w[index[0][2]]


plt.plot(X_c,X_w,'bo',alpha=0.2)
plt.plot(x_accept_c,x_accept_w,'g*',label ='$p(Y\leq50|X)\leq 0.05$')
plt.plot(xtest_singlexp[0],xtest_singlexp[3],'rx', label = '$x_{exp}$')
plt.plot(x_c_star,x_w_star,'ko',label='$x^*$')

plt.xlabel('$X_1$ (Cement)')
plt.ylabel('$X_4$ (Water)')
plt.legend()
#plt.savefig('./Figs/Optimised_cementContent_quick(unscaled)',format='pdf')


y_star_walz = 42.5*(2.769-4.581*(x_w_star/x_c_star)+2.154*(x_w_star/x_c_star)**2)

X_star = np.array([x_c_star,xtest_singlexp[1],xtest_singlexp[2],x_w_star,xtest_singlexp[4],xtest_singlexp[5],xtest_singlexp[6],xtest_singlexp[7]])
x_tmp = Fit.transform(X_star.reshape(1,-1))
y_star,v = gpr(torch.tensor(x_tmp), full_cov =True, noiseless = False)



plt.plot(x_c_star,np.array(ytest)[test_exp_no],'x', label = 'Experiment')
plt.plot(x_c_star,y_star_walz,'g*',label='Walz curve pred')
#plt.plot(x_c_star,y_star.detach().numpy(),'r*')
plt.errorbar(x_c_star,y_star.detach().numpy(),yerr=np.sqrt(v.detach().numpy()),fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, label ='GP prediction')
plt.legend()
plt.xlabel('$X_1^* (Cement)$')
plt.ylabel('Compressive Strength')
plt.savefig('./Figs/Y_star_for_x_star_comparission',format='pdf')


# ## 2.2 -----Optimsation with Torch (with penalty term) ------------------
# $J(X) + \alpha  \textrm{min}(h(X),0.05)$
#
# - \aplha veing small leads to global search, being large lead to local search in the space where constraint is satisfied. A good compromise can be donw with "simulated annealing"
# - everything needs to live in the pytorch world for things to be differntiable
# - $h(X) = (p(y<50|X) < 0.05)$
#
# TODO:
# - 02.01.2021 Scipy constrained optimisation. Need grad from mean and var of GP. Can use gpr.parameters().grad
# - Need to add a volume constraint. vol (cement (density = 3.1kg/dm3) + water (d = 1kg/dm3) + fine agreegate (d = 2.6kg/dm3)+ coarse agreegate (d = 2.6kg/dm3)) = 1m3
#        - First just add a penalty term
#        - Then try with the scipy
# - Try with the basis change what Stelios suggested
#
# https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize


def h_X(X):
    """
    """
    mean, var = gpr(torch.unsqueeze(X,0), full_cov =True, noiseless = False)
    tmp = dist.Normal(mean,torch.sqrt(var))
    return tmp.cdf(torch.tensor(50))

def V_x(X):
    """
    The volume constraint
    """
    # TODO : Detached from the graph. Cant work. Need torch scalar. See above near scikit scalar
    tmp = scalar.inverse_transform(torch.unsqueeze(X,0))
    #print(tmp)
    return tmp[0,0]/3100 + tmp[0,3]/1000 + tmp[0,5]/2600 + tmp[0,6]/2600 -1


def J_X(X):
    """
    Need to select first column i.e cement

    """
    X_tmp = torch.unsqueeze(X,0)
    return X_tmp[:,0]


def obj(X):
    """
    """
    assert X.requires_grad == True
    # Hard coded 4 known inputs for a certain experiment.
    X_tmp = torch.hstack((X[0],torch.tensor(xtest_scl[2,1]),torch.tensor(xtest_scl[2,2]),X[1],torch.tensor(xtest_scl[2,4]),X[2],X[3],torch.tensor(xtest_scl[2,7])))
    return J_X(X_tmp) + 10*torch.min(h_X(X_tmp),torch.tensor(0.05)) + V_x(X_tmp)


X_para = xtest_scl[2,[0,3,5,6]]

XX = torch.tensor(X_para,requires_grad =True)
optimizer = torch.optim.Adam([XX], lr=0.01)
losses = []
x_inmdt = []
grad =[]
num_steps = 400 if not smoke_test else 2
for i in tqdm(range(num_steps)):
    optimizer.zero_grad()
    loss = obj(XX)
    loss.backward()
    #print(XX.grad)
    optimizer.step()
    losses.append(loss.item())
    x_inmdt.append(XX)
    grad.append(torch.norm(XX.grad))


plt.plot(losses);
plt.xlabel("iteration")
plt.ylabel("$J(X) + \lambda min (h(X),0.05)$")


# ### ------------- Constraint optimisation with scipy ------------------
# - https://scipy.github.io/devdocs/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
# - Using trust region constrained algo
# - Pass the Jacobians to it by Torch by calling backward.
# - specifically done for test number 2.
#

# In[389]:


# defining linear volume constraint and bounds

#if bounds are needed the include this
from scipy.optimize import Bounds
lb = [np.array((150-scalar.mean[0][0])/scalar.std[0][0]), np.array((100-scalar.mean[0][1])/scalar.std[0][1]), -np.Inf, -np.Inf]
ub = [np.array((550-scalar.mean[0][0])/scalar.std[0][0]), np.array((250-scalar.mean[0][1])/scalar.std[0][1]), np.Inf, np.Inf]
bounds = Bounds(lb, ub)


from scipy.optimize import LinearConstraint,minimize

# vol (cement (density = 3.1kg/dm3) + water (d = 1kg/dm3) + fine agreegate (d = 2.6kg/dm3)+ coarse agreegate (d = 2.6kg/dm3)) = 1m3
# Info provided
# lb <= A.dot(x) <= ub, where x are the paramters to be optimised
# x is scaled here so adjusting the linear volume constraints
# with X unscaled values, sum(X/density) =1 so sum (x*sigma + mu)/density =1; A = [sigma 1/density1, sigma2/density2 .. ]
# lb = ub = [1-sum (mu1/density1 + mu2/density2 .. )]
A = [scalar.std[0][0]/3100, scalar.std[0][3]/1000, scalar.std[0][5]/2600, scalar.std[0][6]/2600]
lb = 1 - (scalar.mean[0][0]/3100 + scalar.mean[0][3]/1000 + scalar.mean[0][5]/2600 + scalar.mean[0][6]/2600)
linear_constraint = LinearConstraint(A, [lb], [lb])


# For unscaled
#A = [1/3100, 1/1000, 1/2600, 1/2600]
#linear_constraint = LinearConstraint(A, [1], [1])


# In[776]:


class ineq_constrain:
    def __init__(self,X_known,min_comp_strength):
        """
        X_opt [tensor] [scaled] [1x4]
        X_known [tensor] [scaled]
        min_comp_strength : The min strength value, the probability to obtain below this is 0.05
        """
        #self.X_opt = X_opt
        self.X_known = X_known
        self.min_comp_strength = min_comp_strength

    def cons_f(self,X):
        """
        The cdf ranges from 0 to 1, here the statement says that the strength with value 50Mpa or less should be 5% or less
        x_known [tensor] : Known input or the experiments (scaled values), The *arg takes care of this.
        X [ndarray] [1,4] scaled values of values to be optimised
        return:
        array [m,]
        """
        if not isinstance(X,torch.Tensor):
            X = torch.tensor(X)
        X_tmp = torch.hstack((X[0],self.X_known[1],self.X_known[2],X[1],self.X_known[4],X[2],X[3],self.X_known[7]))
        #X_tmp = torch.hstack((X[0],torch.tensor(np.array(xtest)[2,1]),torch.tensor(np.array(xtest)[2,2]),X[1],torch.tensor(np.array(xtest)[2,4]),X[2],X[3],torch.tensor(np.array(xtest)[2,7])))
        # normalize it for the GP
        #X_tmp = scalar.transform(torch.unsqueeze(X_tmp,0))
        #mean, var = gpr(X_tmp, full_cov =True, noiseless = False)
        mean, var = gpr(torch.unsqueeze(X_tmp,0), full_cov =True, noiseless = False)
        tmp = dist.Normal(mean,torch.sqrt(var))

        if X.requires_grad == True:
            return tmp.cdf(torch.tensor(self.min_comp_strength))
        return tmp.cdf(torch.tensor(self.min_comp_strength)).detach().numpy().reshape(-1,) # to make teh shape [m,]

    def cons_jac(self,X):
        """
        return:
            [m X N] with m being the constraint output size and N the input dimention
        """
        X = torch.tensor(X,requires_grad =True)
        tmp = self.cons_f(X)
        tmp.backward()
        return torch.unsqueeze(X.grad,0).detach().numpy()


# defining onjective function
class objective:
    def __init__(self):

        self.accumulator = list() # empty list
    def obj_fn(self,X):
        """
        Args:
        X [N,]
        return:
        []
        """
        if not isinstance(X,torch.Tensor):
            self.accumulator.append(X)
            X = torch.tensor(X)
        X_tmp = torch.unsqueeze(X,0)
        if X.requires_grad == True:
            return X_tmp[:,0]
        return X_tmp[:,0].detach().numpy().reshape(-1,)

    def obj_jac(self,X):
        """
        Args:

        return:
        grad [N,] where X is with shape (N,)
        """
        X = torch.tensor(X,requires_grad =True)
        tmp = obj_(X)
        tmp.backward()
        return X.grad.detach().numpy()


def run_minimization_problem(X_init,min_strength,idx_opt = [0,3,5,6]):
    """
    X_init [1x8] Scaled values
    min_strength : The min strength value, the probability to obtain below this is 0.05
    idx_known : The indices which needs to be optimised

    return:

    res: the minimisation fun res object
    accumulated: The accumulated values of optimisation features
    """
    # init value
    x0 = np.array(X_init[idx_opt])

    # linear constraints
    A = [scalar.std[0][0]/3100, scalar.std[0][3]/1000, scalar.std[0][5]/2600, scalar.std[0][6]/2600]
    lb = 1 - (scalar.mean[0][0]/3100 + scalar.mean[0][3]/1000 + scalar.mean[0][5]/2600 + scalar.mean[0][6]/2600)
    linear_constraint = LinearConstraint(A, [lb], [lb])

    # Non lineara constraints
    cons = ineq_constrain(X_init,min_strength)
    nonlinear_constraint = NonlinearConstraint(cons.cons_f, 0, 0.05, jac=cons.cons_jac)

    # Objective
    obj = objective()

    # As the objective function is linear, hessian can be specified as zero, hessisan would be nxn.
    # Bounds can be removed too. Check it.
    res = minimize(obj.obj_fn, x0, method='trust-constr', jac=obj.obj_jac,hess = lambda x: np.zeros((x0.shape[0], x0.shape[0])),
               constraints=[linear_constraint,nonlinear_constraint],
               options={'verbose': 1})

    # unscaling the accumulated X's
    accumulated = np.array(obj.accumulator)
    # scalar is only defined for 8 dimentional inputs so manually doing it
    accumulated_unscaled = torch.tensor(accumulated)*scalar.std[:,idx_opt] + scalar.mean[:,idx_opt]

    return res, accumulated_unscaled


def pair_plot_optimised(X_opt, path=None, name=None):
    """
    Pair plot for optimised values. Unscaled values needs to be passed.
    args:
    X [Tensor][p x N], p being the steps and N the number of dimentions being optimised
    """
    assert isinstance(X_opt,torch.Tensor)

    fig = plt.figure(figsize=(18, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    for i in range(X_opt.shape[1]-1):
        plt.subplot(1, 3, i+1)
        plt.plot(X_opt[:,0],X_opt[:,i+1])
        plt.plot(X_opt[0,0],X_opt[0,i+1],'gx')
        plt.plot(X_opt[-1,0],X_opt[-1,i+1],'rx')
        plt.xlabel('$X_1$ (Cement)')

    # Custom labeling the y axis
        if i==0:
            plt.ylabel('$X_4$ (Water)')

        if i==1:
            plt.ylabel('$X_6$ (Coarse Aggregate)')
        if i==2:
            plt.ylabel('$X_7$ (Fine Aggregate)')

    if path is not None:
        plt.savefig(path + name +datetime,format='pdf')



# checking inequility constraint is satisfied or not
cdf_opt = []
for i in range(accumulated.shape[0]):
    cdf_opt.append(cons_f(accumulated[i,:]))

plt.plot(cdf_opt)
plt.xlabel('Iterations')
plt.ylabel('$p(y\leq 50 Mpa|X)$')
path='./Figs/'
name='inequality_constraint_optimisation_trust'
#plt.savefig(path + name +datetime,format='pdf')



plt.plot(accumulated_unscaled[:,0], label ='Obj function evolution')

plt.axhline(y=np.array(xtest)[2,0],color='yellow',label='Experimental cement content')
plt.ylabel('$X_1$ (Cement)')
plt.xlabel('Iterations')
plt.legend()
path='./Figs/'
name='Obective_evolution_optimisation_trust'
#plt.savefig(path + name +datetime,format='pdf')


torch.tensor(accumulated[-1,0])


X_star = torch.hstack((torch.tensor(accumulated[-1,0]),xtest_scl[2,1],xtest_scl[2,2],torch.tensor(accumulated[-1,1]),xtest_scl[2,4],torch.tensor(accumulated[-1,2]),torch.tensor(accumulated[-1,3]),xtest_scl[2,7]))
y_star,v = gpr(torch.unsqueeze(X_star,0), full_cov =True, noiseless = False)

# the 52.5 strength class
y_star_walz = 52.5*(2.587-4.109*(np.array(accumulated_unscaled[-1,1])/np.array(accumulated_unscaled[-1,0]))+1.873*(np.array(accumulated_unscaled[-1,1])/np.array(accumulated_unscaled[-1,0]))**2)

x_c_star = accumulated_unscaled[-1,0]
test_exp_no = 2

plt.plot(np.array(xtest)[2,0],np.array(ytest)[test_exp_no],'yx', label = 'Experiment')
#plt.plot(x_c_star,y_star_walz,'g*',label='Walz curve pred')
#plt.plot(x_c_star,y_star.detach().numpy(),'r*')
plt.errorbar(x_c_star,y_star.detach().numpy(),yerr=2*np.sqrt(v[0].detach().numpy()),fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, label ='GP prediction $\pm 2 \sigma$')
plt.legend()
plt.xlabel('$X_1^* (Cement)$')
plt.ylabel('Compressive Strength')
#plt.savefig('./Figs/Y_star_for_x_star_trust_'+datetime,format='pdf')


# Choosing the index which satisfying the above consitions
#idx_ = np.intersect1d(np.where((np.array(xtest['age'])==np.log(28)) & (np.array(xtest['water'])/np.array(xtest['cement'])<1)),np.where((np.array(xtest['blast_furnace_slag']) == 0) & (np.array(xtest['fly_ash'])==0)))
idx_ = np.intersect1d(np.where((np.array(x['age'])==np.log(28)) & (np.array(x['water'])/np.array(x['cement'])<1)),np.where((np.array(x['blast_furnace_slag']) == 0) & (np.array(x['fly_ash'])==0)))


xtmp =np.array(x)[idx_]
ytmp =np.array(y)[idx_]


# perform optimisation for the six cases
res = []
accu = []
for i in range(xtest_plot.shape[0]):
    tmp1,tmp2 = run_minimization_problem(xtest_plot_scaled[i,:],30)
    res.append(tmp1)
    accu.append(tmp2)

x_opt_scaled =np.array([i.x for i in res])

# X_1 vs X_5
plt.plot(xtest_plot[:,4],np.array((accu[0][-1,0],accu[1][-1,0],accu[2][-1,0],accu[3][-1,0],accu[4][-1,0])),'kX',label='Optimised')
plt.plot(xtest_plot[:,4],xtest_plot[:,0],'yo',label='Experimental')
plt.xlabel('$X_5$(plasticizer)')
plt.ylabel('$X_1$(Cement)')
plt.legend()
plt.savefig('./Figs/X1_vs_X5'+datetime,format='pdf')


# X_1 vs y (compressive strength)
X_star = torch.cat((torch.tensor(x_opt_scaled[:,0].reshape(-1,1)),xtest_plot_scaled[:,1].reshape(-1,1),xtest_plot_scaled[:,2].reshape(-1,1),torch.tensor(x_opt_scaled[:,1].reshape(-1,1)),xtest_plot_scaled[:,4].reshape(-1,1),torch.tensor(x_opt_scaled[:,2].reshape(-1,1)),torch.tensor(x_opt_scaled[:,3].reshape(-1,1)),xtest_plot_scaled[:,7].reshape(-1,1)),dim=1)
y_star,v = gpr(X_star, full_cov =True, noiseless = False)


x_c_star = np.array((accu[0][-1,0],accu[1][-1,0],accu[2][-1,0],accu[3][-1,0],accu[4][-1,0]))
#plt.plot(x_c_star,y_star.detach().numpy(),'kx', label = 'GP')
sd =v.diag().sqrt()
#plt.fill_between(x_c_star,  # plot the two-sigma uncertainty about the mean
#                 (y_star - 2.0 * sd).detach().numpy(),
##                 color='black', alpha=0.1, label="$\pm 2 \sigma$")
#plt.plot(xtest_plot[:,0],ytest_plot,'yo')
plt.errorbar(x_c_star,y_star.detach().numpy(),yerr=2*sd.detach().numpy(),fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, label ='GP prediction $\pm 2 \sigma$')
plt.axhline(y=30,label='Probability below this line is 0.05')
plt.legend()
plt.annotate(s='$X_5 =$'+f'{xtest_plot[0,4]}',xy=[x_c_star[0],34])
plt.annotate(s='$X_5 =$'+f'{xtest_plot[1,4]}',xy=[x_c_star[1],32])
plt.annotate(s='$X_5 =$'+f'{xtest_plot[2,4]}',xy=[x_c_star[2],38])
plt.annotate(s='$X_5 =$'+f'{xtest_plot[3,4]}',xy=[x_c_star[3],38])
plt.annotate(s='$X_5 =$'+f'{xtest_plot[4,4]}',xy=[x_c_star[4],38])




plt.xlabel('$X_1^* (Cement)$')
plt.ylabel('Compressive Strength')
plt.savefig('./Figs/X_1 vs compressive_strength_wrtX5'+datetime,format='pdf')
