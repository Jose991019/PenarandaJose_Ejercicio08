import numpy as np
import matplotlib.pyplot as plt

def model_A(x, params):
    y = params[0] + x*params[1] + params[2]*x**2
    return y
def model_B(x, params):
    y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    return y    
def model_C(x, params):
    y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    y += params[0]*(np.exp(-0.5*(x-params[3])**2/params[4]**2))
    return y
def loglike_A(x_obs, y_obs, sigma_y_obs, betas):
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model_A(x_obs[i], betas))**2/sigma_y_obs[i]**2
    return l
def loglike_B(x_obs, y_obs, sigma_y_obs, betas):
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model_B(x_obs[i], betas))**2/sigma_y_obs[i]**2
    return l
def loglike_C(x_obs, y_obs, sigma_y_obs, betas):
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model_C(x_obs[i], betas))**2/sigma_y_obs[i]**2
    return l

data = np.loadtxt("data_to_fit.txt")
x_obs = data[:,0]
y_obs = data[:,1]
sigma_y_obs = data[:,2]
n = len(x_obs)

N = 250000
N_param = 3
betasA = np.zeros([N, N_param])
for i in range(1, N):
    current_betas = betasA[i-1,:]
    next_betas = current_betas + np.random.normal(scale=0.05, size=N_param)
    loglike_current = loglike_A(x_obs, y_obs, sigma_y_obs, current_betas)
    loglike_next = loglike_A(x_obs, y_obs, sigma_y_obs, next_betas)
    r = np.min([np.exp(loglike_next - loglike_current), 1.0])
    alpha = np.random.random()
    if alpha < r:
        betasA[i,:] = next_betas
    else:
        betasA[i,:] = current_betas
betasA = betasA[N//2:,:]

N = 70000
betasB = np.zeros([N, N_param])
for i in range(1, N):
    current_betas = betasB[i-1,:]
    next_betas = current_betas + np.random.normal(scale=0.02, size=N_param)
    loglike_current = loglike_B(x_obs, y_obs, sigma_y_obs, current_betas)
    loglike_next = loglike_B(x_obs, y_obs, sigma_y_obs, next_betas)
    r = np.min([np.exp(loglike_next - loglike_current), 1.0])
    alpha = np.random.random()
    if alpha < r:
        betasB[i,:] = next_betas
    else:
        betasB[i,:] = current_betas
betasB = betasB[N//2:,:]

N = 100000
N_param = 5
betasC = np.zeros([N, N_param])
for i in range(1, N):
    current_betas = betasC[i-1,:]
    next_betas = current_betas + np.random.normal(scale=0.1, size=N_param)
    loglike_current = loglike_C(x_obs, y_obs, sigma_y_obs, current_betas)
    loglike_next = loglike_C(x_obs, y_obs, sigma_y_obs, next_betas)
    r = np.min([np.exp(loglike_next - loglike_current), 1.0])
    alpha = np.random.random()
    if alpha < r:
        betasC[i,:] = next_betas
    else:
        betasC[i,:] = current_betas
betasC = betasC[N//2:,:]

plt.figure(figsize = (10,10))
for i in range(0,3):
    plt.subplot(2,2,i+1)
    plt.hist(betasA[:,i],bins=15, density=True)
    plt.title(r"$\beta_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(betasA[:,i]), np.std(betasA[:,i])))
    plt.xlabel(r"$\beta_{}$".format(i))
plt.subplot(2,2,4)
plt.scatter(x_obs,y_obs)
plt.plot(x, model_A(x,np.mean(betasA,axis=0)))
plt.title("{}".format(2*(-loglike_A(x_obs, y_obs, sigma_y_obs, np.mean(betasA,axis=0)) + 3.0/2 * np.log(n))))
plt.subplots_adjust(hspace=0.5)
plt.savefig("modelo_A.png",  bbox_inches='tight')    


plt.figure(figsize = (10,10))
for i in range(0,3):
    plt.subplot(2,2,i+1)
    plt.hist(betasB[:,i],bins=15, density=True)
    plt.title(r"$\beta_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(betasB[:,i]), np.std(betasB[:,i])))
    plt.xlabel(r"$\beta_{}$".format(i))
plt.subplot(2,2,4)
plt.scatter(x_obs,y_obs)
plt.plot(x, model_B(x,np.mean(betasB,axis=0)))
plt.title("{}".format(2*( -loglike_B(x_obs, y_obs, sigma_y_obs, np.mean(betasB,axis=0)) + 3.0/2 * np.log(n))))
plt.subplots_adjust(hspace=0.5)
plt.savefig("modelo_B.png",  bbox_inches='tight')    

plt.figure(figsize = (15,10))
for i in range(0,5):
    plt.subplot(2,3,i+1)
    plt.hist(betasC[:,i],bins=15, density=True)
    plt.title(r"$\beta_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(betasC[:,i]), np.std(betasC[:,i])))
    plt.xlabel(r"$\beta_{}$".format(i))
plt.subplot(2,3,6)
plt.scatter(x_obs,y_obs)
plt.plot(x, model_C(x,np.mean(betasC,axis=0)))
plt.title("{}".format(2*(-loglike_C(x_obs, y_obs, sigma_y_obs, np.mean(betasC,axis=0)) + 5.0/2 * np.log(n))))
plt.subplots_adjust(hspace=0.5)
plt.savefig("modelo_C.png",  bbox_inches='tight')   