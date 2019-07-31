from __future__ import division
import numpy as np

def choleraEqs(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """

    :param y:
    :param t:
    :param params: the parameters of the model. they are described below
    :param TravelMat: a matrix of the flux of travelers between patch i and patch j. TravelMat
    :return:
    """
    #parameters
    #alpha, alpha1, alpha2: cholera related mortality in unvaccinated, vacc. with one dose, and vacc. with 2 doses
    #fracSym = fraction of symptomatic cases
    #delta = bacteria decay in water
    #mu, mu1, mu2 = rate of excretion of V.cholera symptomatic unvaccinated, vacc with one dose, and vacc with 2 doses
    # mu, mu1, mu2 = rate of excretion of V.cholera Asymptomatic unvaccinated, vacc with one dose, and vacc with 2 doses


    [alpha, alpha1, alpha2, beta, beta_A, betaW, delta, fracSym, gamma, gamma1, gamma2,
     m, m1, m2, mu, mu1, mu2, mu_A, mu_A1, mu_A2, phi1, phi2,
     omega1, omega2, sigma, theta1, theta2, V] = params

    # print y
    # print t
    #split the variables to compute the ODE's
    #the susceptible classes are the first
    mynewarray = np.reshape(y, (16, numDept))
    # print mynewarray
    S = mynewarray[0,:]
    I = mynewarray[1,:]
    A = mynewarray[2,:]
    R = mynewarray[3,:]
    RA = mynewarray[4,:]
    V1 = mynewarray[5,:]
    I1 = mynewarray[6,:]
    A1 = mynewarray[7,:]
    R1 = mynewarray[8,:]
    RA1 = mynewarray[9, :]
    V2 = mynewarray[10,:]
    I2 = mynewarray[11,:]
    A2 = mynewarray[12,:]
    R2 = mynewarray[13,:]
    RA2 = mynewarray[14, :]
    W  = mynewarray[15,:]

    #compute the force of infection:
    mylambda = (beta*I + beta_A*A + phi1*beta*I1 + phi1*beta_A*A1 + phi2*beta*I2 + phi2*beta_A*A2)/Pop + \
               (betaW*W)/(V+W)
    # print beta*I
    ########### Equations for those unvaccinated:

    dS = -np.multiply(mylambda, S) + sigma*(R + RA) + omega1*(V1 + R1) + omega2*(V2 + R2) + np.dot(TravelMat.transpose(), S) \
         - np.sum(TravelMat*S[:, None], 1)
    # print S
    # print -np.multiply(mylambda, S)

    dI = fracSym*np.multiply(mylambda, S) - (alpha + gamma)*I + m*(np.dot(TravelMat.transpose(), I) -
                                                                   np.sum(TravelMat*I[:, None], 1))

    dA = (1-fracSym)*np.multiply(mylambda, S) -gamma*A + np.dot(TravelMat.transpose(), A) - np.sum(TravelMat*A[:, None], 1)

    dR = gamma*I -sigma*R + np.dot(TravelMat.transpose(), R) - np.sum(TravelMat*R[:, None], 1)

    dRA = gamma * A - sigma * RA + np.dot(TravelMat.transpose(), RA) - np.sum(TravelMat * RA[:, None], 1)


    ########### Equations for those vaccinated with a single dose
    dV1 = -theta1*np.multiply(mylambda, V1) - omega1*V1 + sigma*(R1 + RA1) + np.dot(TravelMat.transpose(), V1) \
          - np.sum(TravelMat*V1[:, None], 1)

    dI1 = fracSym*theta1*np.multiply(mylambda, V1) -(alpha1 + gamma1)*I1 + m1*(np.dot(TravelMat.transpose(), I1) -
                                                                               np.sum(TravelMat*I1[:, None], 1))

    dA1 = (1 - fracSym)*theta1*np.multiply(mylambda, V1) - gamma1*A1 + np.dot(TravelMat.transpose(), A1) - \
          np.sum(TravelMat*A1[:, None], 1)

    dR1 = gamma1*(I1)  - (sigma + omega1)*R1 + np.dot(TravelMat.transpose(), R1) - np.sum(TravelMat*R1[:, None], 1)

    dRA1 = gamma1 * (A1) - (sigma + omega1) * RA1 + np.dot(TravelMat.transpose(), RA1) - np.sum(
        TravelMat * RA1[:, None], 1)


    ########### Equations for those vaccinated with two doses:
    dV2 = -theta2*np.multiply(mylambda, V2) - omega2*V2 + sigma*(R2 + RA2) + np.dot(TravelMat.transpose(), V2) \
          - np.sum(TravelMat*V2[:, None], 1)

    dI2 = fracSym*theta2*np.multiply(mylambda, V2) - (alpha2 + gamma2)*I2 + m2*(np.dot(TravelMat.transpose(), I2)
                                                                                    -np.sum(TravelMat * I2[:, None], 1))

    dA2 = (1 - fracSym)*theta2*np.multiply(mylambda, V2) - gamma2*A2 + np.dot(TravelMat.transpose(), A2) - \
          np.sum(TravelMat*A2[:, None], 1)

    dR2 = gamma2*(I2)  - (sigma + omega2)*R2 + np.dot(TravelMat.transpose(), R2) - np.sum(TravelMat*R2[:, None], 1)

    dRA2 = gamma2 * (A2) - (sigma + omega2) * RA2 + np.dot(TravelMat.transpose(), RA2) - np.sum(
        TravelMat * RA2[:, None], 1)


    ######## Water equations:
    dW = mu*I + mu_A*A + mu1*I1 + mu_A1*A1 + mu2*I2 + mu_A2*A2 - delta*W + np.dot(WaterMat.transpose(), W) - \
        np.sum(WaterMat*W[:, None], 1)

    dydt = np.array([dS, dI, dA, dR, dRA, dV1, dI1, dA1, dR1, dRA1, dV2, dI2, dA2, dR2, dRA2, dW]).reshape((16*numDept))
    # print dS
    # print dI
    # print dydt
    # print np.shape(dydt)
    return dydt


def choleraEqs2(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    this function is different from choleraEqs in the following aspects:
    1. here, only infectious people will travel between departements
    2. I added a sinusoidal function to simulate forcing in the water reservoir, by doing this, I added an extra parame
    ter, rho

    :param y:
    :param t:
    :param params: the parameters of the model. they are described below
    :param TravelMat: a matrix of the flux of travelers between patch i and patch j. TravelMat
    :return:
    """
    #parameters
    #alpha, alpha1, alpha2: cholera related mortality in unvaccinated, vacc. with one dose, and vacc. with 2 doses
    #fracSym = fraction of symptomatic cases
    #delta = bacteria decay in water
    #mu, mu1, mu2 = rate of excretion of V.cholera symptomatic unvaccinated, vacc with one dose, and vacc with 2 doses
    # mu, mu1, mu2 = rate of excretion of V.cholera Asymptomatic unvaccinated, vacc with one dose, and vacc with 2 doses


    [alpha, alpha1, alpha2, beta, beta_A, betaW, delta, fracSym, gamma, gamma1, gamma2,
     m, m1, m2, mu, mu1, mu2, mu_A, mu_A1, mu_A2, phi1, phi2,
     omega1, omega2, rho, sigma, theta1, theta2, V] = params


    #split the variables to compute the ODE's
    #the susceptible classes are the first
    mynewarray = np.reshape(y, (16, numDept))
    S = mynewarray[0,:]
    I = mynewarray[1,:]
    # print I
    A = mynewarray[2,:]
    R = mynewarray[3,:]
    RA = mynewarray[4,:]
    V1 = mynewarray[5,:]
    I1 = mynewarray[6,:]
    A1 = mynewarray[7,:]
    R1 = mynewarray[8,:]
    RA1 = mynewarray[9, :]
    V2 = mynewarray[10,:]
    I2 = mynewarray[11,:]
    A2 = mynewarray[12,:]
    R2 = mynewarray[13,:]
    RA2 = mynewarray[14, :]
    W  = mynewarray[15,:]
    # print t
    # print S
    # print V1

    #compute the force of infection:
    mylambda = (beta*I + beta_A*A + phi1*beta*I1 + phi1*beta_A*A1 + phi2*beta*I2 + phi2*beta_A*A2)/Pop + \
               0.5*(1 + rho*np.cos(2*np.pi*(t /182.5)))*(betaW*W)/(V+W)
    # print beta*I
    ########### Equations for those unvaccinated:

    dS = -np.multiply(mylambda, S)  + omega1*(V1 + R1 + RA1) +\
          omega2*(V2+ R2 + RA2)  + sigma*(R + RA)

    # print S
    # print -np.multiply(mylambda, S)

    dI = fracSym*np.multiply(mylambda, S) - (alpha + gamma)*I + m*(np.dot(TravelMat.transpose(), I) -
                                                                   np.sum(TravelMat*I[:, None], 1))

    dA = (1-fracSym)*np.multiply(mylambda, S) -gamma*A + np.dot(TravelMat.transpose(), A) - np.sum(TravelMat*A[:, None], 1)

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA


    ########### Equations for those vaccinated with a single dose
    dV1 = -theta1*np.multiply(mylambda, V1) - omega1*V1 + sigma*(R1 + RA1)

    dI1 = fracSym*theta1*np.multiply(mylambda, V1) -(alpha1 + gamma1)*I1 + m1*(np.dot(TravelMat.transpose(), I1) -
                                                                               np.sum(TravelMat*I1[:, None], 1))

    dA1 = (1 - fracSym)*theta1*np.multiply(mylambda, V1) - gamma1*A1 + np.dot(TravelMat.transpose(), A1) - \
          np.sum(TravelMat*A1[:, None], 1)

    dR1 = gamma1*(I1)  - (sigma + omega1)*R1

    dRA1 = gamma1 * (A1) - (sigma + omega1) * RA1


    ########### Equations for those vaccinated with two doses:
    dV2 = -theta2*np.multiply(mylambda, V2) - omega2*V2 + sigma*(R2 + RA2)

    dI2 = fracSym*theta2*np.multiply(mylambda, V2) - (alpha2 + gamma2)*I2 + m2*(np.dot(TravelMat.transpose(), I2)
                                                                                    -np.sum(TravelMat * I2[:, None], 1))

    dA2 = (1 - fracSym)*theta2*np.multiply(mylambda, V2) - gamma2*A2 + np.dot(TravelMat.transpose(), A2) - \
          np.sum(TravelMat*A2[:, None], 1)

    dR2 = gamma2*(I2)  - (sigma + omega2)*R2

    dRA2 = gamma2 * (A2) - (sigma + omega2) * RA2


    ######## Water equations:
    dW = mu*I + mu_A*A + mu1*I1 + mu_A1*A1 + mu2*I2 + mu_A2*A2 - delta*W + np.dot(WaterMat.transpose(), W) - \
        np.sum(WaterMat*W[:, None], 1)

    dydt = np.array([dS, dI, dA, dR, dRA, dV1, dI1, dA1, dR1, dRA1, dV2, dI2, dA2, dR2, dRA2, dW]).reshape((16*numDept))
    # print dS
    # print dI
    # print dydt
    # print np.shape(dydt)
    return dydt

def choleraEqs2bis(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    this function is different from choleraEqs2 because I added a box to keep track of the new infectiosn to be able to
    compute the number of reported cases. This is for bookeeping purposes only and does not affect the dynamics at all

    :param y:
    :param t:
    :param params: the parameters of the model. they are described below
    :param TravelMat: a matrix of the flux of travelers between patch i and patch j. TravelMat
    :return:
    """
    #parameters
    #alpha, alpha1, alpha2: cholera related mortality in unvaccinated, vacc. with one dose, and vacc. with 2 doses
    #fracSym = fraction of symptomatic cases
    #delta = bacteria decay in water
    #mu, mu1, mu2 = rate of excretion of V.cholera symptomatic unvaccinated, vacc with one dose, and vacc with 2 doses
    # mu, mu1, mu2 = rate of excretion of V.cholera Asymptomatic unvaccinated, vacc with one dose, and vacc with 2 doses


    [alpha, alpha1, alpha2, beta, beta_A, betaW, delta, fracSym, gamma, gamma1, gamma2,
     m, m1, m2, mu, mu1, mu2, mu_A, mu_A1, mu_A2, phi1, phi2,
     omega1, omega2, rho, sigma, theta1, theta2, V] = params


    #split the variables to compute the ODE's
    #the susceptible classes are the first
    mynewarray = np.reshape(y, (19, numDept))
    S = mynewarray[0,:]
    I = mynewarray[1,:]
    # print I
    A = mynewarray[2,:]
    R = mynewarray[3,:]
    RA = mynewarray[4,:]
    V1 = mynewarray[5,:]
    I1 = mynewarray[6,:]
    A1 = mynewarray[7,:]
    R1 = mynewarray[8,:]
    RA1 = mynewarray[9, :]
    V2 = mynewarray[10,:]
    I2 = mynewarray[11,:]
    A2 = mynewarray[12,:]
    R2 = mynewarray[13,:]
    RA2 = mynewarray[14, :]
    W  = mynewarray[15,:]
    #cases:
    C = mynewarray[16,:]
    C1 = mynewarray[17,:]
    C2 = mynewarray[18,:]

    # print t
    # print S
    # print V1

    #compute the force of infection:
    mylambda = (beta*I + beta_A*A + phi1*beta*I1 + phi1*beta_A*A1 + phi2*beta*I2 + phi2*beta_A*A2)/Pop + \
               0.5*(1 + rho*np.cos(2*np.pi*(t /182.5)))*(betaW*W)/(V+W)
    # print beta*I
    ########### Equations for those unvaccinated:

    dS = -np.multiply(mylambda, S)  + omega1*(V1 + R1 + RA1) +\
          omega2*(V2+ R2 + RA2)  + sigma*(R + RA)

    # print S
    # print -np.multiply(mylambda, S)

    dI = fracSym*np.multiply(mylambda, S) - (alpha + gamma)*I + m*(np.dot(TravelMat.transpose(), I) -
                                                                   np.sum(TravelMat*I[:, None], 1))

    dA = (1-fracSym)*np.multiply(mylambda, S) -gamma*A + np.dot(TravelMat.transpose(), A) - np.sum(TravelMat*A[:, None], 1)

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA


    ########### Equations for those vaccinated with a single dose
    dV1 = -theta1*np.multiply(mylambda, V1) - omega1*V1 + sigma*(R1 + RA1)

    dI1 = fracSym*theta1*np.multiply(mylambda, V1) -(alpha1 + gamma1)*I1 + m1*(np.dot(TravelMat.transpose(), I1) -
                                                                               np.sum(TravelMat*I1[:, None], 1))

    dA1 = (1 - fracSym)*theta1*np.multiply(mylambda, V1) - gamma1*A1 + np.dot(TravelMat.transpose(), A1) - \
          np.sum(TravelMat*A1[:, None], 1)

    dR1 = gamma1*(I1)  - (sigma + omega1)*R1

    dRA1 = gamma1 * (A1) - (sigma + omega1) * RA1


    ########### Equations for those vaccinated with two doses:
    dV2 = -theta2*np.multiply(mylambda, V2) - omega2*V2 + sigma*(R2 + RA2)

    dI2 = fracSym*theta2*np.multiply(mylambda, V2) - (alpha2 + gamma2)*I2 + m2*(np.dot(TravelMat.transpose(), I2)
                                                                                    -np.sum(TravelMat * I2[:, None], 1))

    dA2 = (1 - fracSym)*theta2*np.multiply(mylambda, V2) - gamma2*A2 + np.dot(TravelMat.transpose(), A2) - \
          np.sum(TravelMat*A2[:, None], 1)

    dR2 = gamma2*(I2)  - (sigma + omega2)*R2

    dRA2 = gamma2 * (A2) - (sigma + omega2) * RA2


    ######## Water equations:
    dW = mu*I + mu_A*A + mu1*I1 + mu_A1*A1 + mu2*I2 + mu_A2*A2 - delta*W + np.dot(WaterMat.transpose(), W) - \
        np.sum(WaterMat*W[:, None], 1)

    dC = fracSym*np.multiply(mylambda, S) + m*(np.dot(TravelMat.transpose(), I))

    dC1 = fracSym*theta1*np.multiply(mylambda, V1) + m1*(np.dot(TravelMat.transpose(), I1))

    dC2 = fracSym*theta2*np.multiply(mylambda, V2) + m2*(np.dot(TravelMat.transpose(), I2))

    dydt = np.array([dS, dI, dA, dR, dRA, dV1, dI1, dA1, dR1, dRA1, dV2, dI2, dA2, dR2, dRA2, dW, dC, dC1, dC2]).reshape((19*numDept))
    # print dS
    # print dI
    # print dydt
    # print np.shape(dydt)
    return dydt


def choleraEqs3(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    this function is different from choleraEqs2 in the following aspect:
    here, only vaccinated SUSCEPTIBLE people see their protection wane and move to the fully susceptible class.

    :param y:
    :param t:
    :param params: the parameters of the model. they are described below
    :param TravelMat: a matrix of the flux of travelers between patch i and patch j. TravelMat
    :return:
    """
    #parameters

    #fracSym = fraction of symptomatic cases
    #delta = bacteria decay in water
    #mu, mu1, mu2 = rate of excretion of V.cholera symptomatic unvaccinated, vacc with one dose, and vacc with 2 doses
    # mu, mu1, mu2 = rate of excretion of V.cholera Asymptomatic unvaccinated, vacc with one dose, and vacc with 2 doses


    [beta, beta_A, betaW, delta, fracSym, gamma, gamma1, gamma2,
     m, m1, m2, mu, mu1, mu2, mu_A, mu_A1, mu_A2, phi1, phi2,
     omega1, omega2, rho, sigma, theta1, theta2, V] = params


    #split the variables to compute the ODE's
    #the susceptible classes are the first
    mynewarray = np.reshape(y, (16, numDept))
    S = mynewarray[0,:]
    I = mynewarray[1,:]
    # print I
    A = mynewarray[2,:]
    R = mynewarray[3,:]
    RA = mynewarray[4,:]
    V1 = mynewarray[5,:]
    I1 = mynewarray[6,:]
    A1 = mynewarray[7,:]
    R1 = mynewarray[8,:]
    RA1 = mynewarray[9, :]
    V2 = mynewarray[10,:]
    I2 = mynewarray[11,:]
    A2 = mynewarray[12,:]
    R2 = mynewarray[13,:]
    RA2 = mynewarray[14, :]
    W  = mynewarray[15,:]
    # print t
    # print S
    # print V1

    #compute the force of infection:
    mylambda = (beta*I + beta_A*A + phi1*beta*I1 + phi1*beta_A*A1 + phi2*beta*I2 + phi2*beta_A*A2)/Pop + \
               0.5*(1 + rho*np.cos(2*np.pi*(t /182.5)))*(betaW*W)/(V+W)
    # print beta*I
    ########### Equations for those unvaccinated:

    dS = -np.multiply(mylambda, S)  + omega1*(V1) +\
          omega2*(V2)  + sigma*(R + RA)

    # print S
    # print -np.multiply(mylambda, S)

    dI = fracSym*np.multiply(mylambda, S) - (alpha + gamma)*I + m*(np.dot(TravelMat.transpose(), I) -
                                                                   np.sum(TravelMat*I[:, None], 1))

    dA = (1-fracSym)*np.multiply(mylambda, S) -gamma*A + np.dot(TravelMat.transpose(), A) - np.sum(TravelMat*A[:, None], 1)

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA


    ########### Equations for those vaccinated with a single dose
    dV1 = -theta1*np.multiply(mylambda, V1) - omega1*V1 + sigma*(R1 + RA1)

    dI1 = fracSym*theta1*np.multiply(mylambda, V1) -(alpha1 + gamma1)*I1 + m1*(np.dot(TravelMat.transpose(), I1) -
                                                                               np.sum(TravelMat*I1[:, None], 1))

    dA1 = (1 - fracSym)*theta1*np.multiply(mylambda, V1) - gamma1*A1 + np.dot(TravelMat.transpose(), A1) - \
          np.sum(TravelMat*A1[:, None], 1)

    dR1 = gamma1*(I1)  - (sigma )*R1

    dRA1 = gamma1 * (A1) - (sigma) * RA1


    ########### Equations for those vaccinated with two doses:
    dV2 = -theta2*np.multiply(mylambda, V2) - omega2*V2 + sigma*(R2 + RA2)

    dI2 = fracSym*theta2*np.multiply(mylambda, V2) - (alpha2 + gamma2)*I2 + m2*(np.dot(TravelMat.transpose(), I2)
                                                                                    -np.sum(TravelMat * I2[:, None], 1))

    dA2 = (1 - fracSym)*theta2*np.multiply(mylambda, V2) - gamma2*A2 + np.dot(TravelMat.transpose(), A2) - \
          np.sum(TravelMat*A2[:, None], 1)

    dR2 = gamma2*(I2)  - (sigma )*R2

    dRA2 = gamma2 * (A2) - (sigma ) * RA2


    ######## Water equations:
    dW = mu*I + mu_A*A + mu1*I1 + mu_A1*A1 + mu2*I2 + mu_A2*A2 - delta*W + np.dot(WaterMat.transpose(), W) - \
        np.sum(WaterMat*W[:, None], 1)

    dydt = np.array([dS, dI, dA, dR, dRA, dV1, dI1, dA1, dR1, dRA1, dV2, dI2, dA2, dR2, dRA2, dW]).reshape((16*numDept))
    # print dS
    # print dI
    # print dydt
    # print np.shape(dydt)
    return dydt


def choleraEqs4(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    This function is based on Jonathan's modified version of my model. That version can be found in the document
    cholera_haiti_equationsJonathan.docx

    Here, I will remove the death rate from the infectious compartments due to cholera

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW, D, delta, frate, k, gamma, gamma1, gamma2, m, m1, m2,
     mu, mu_A, mu1, mu_A1, mu2, mu_A2, phi1, phi2, pseason, omega1, omega2, sigma, theta1, theta2, V] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (16, numDept))
    S = mynewarray[0, :]
    I = mynewarray[1, :]
    # print I
    A = mynewarray[2, :]
    R = mynewarray[3, :]
    RA = mynewarray[4, :]
    V1 = mynewarray[5, :]
    I1 = mynewarray[6, :]
    A1 = mynewarray[7, :]
    R1 = mynewarray[8, :]
    RA1 = mynewarray[9, :]
    V2 = mynewarray[10, :]
    I2 = mynewarray[11, :]
    A2 = mynewarray[12, :]
    R2 = mynewarray[13, :]
    RA2 = mynewarray[14, :]
    W = mynewarray[15, :]

    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) + \
               ((beta * I + beta_A * A + phi1 * beta * I1 + phi1 * beta_A * A1 + phi2 * beta * I2 + phi2 * beta_A * A2)  +
                beta*m*(np.dot(TravelMat.transpose(), I) - np.sum(TravelMat*I[:, None], 1))  +
                beta_A*(np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1)) +
                beta*m1 * (np.dot(TravelMat.transpose(), I1) - np.sum(TravelMat * I1[:, None], 1)) +
                beta_A*(np.dot(TravelMat.transpose(), A1) - np.sum(TravelMat * A1[:, None], 1)) +
                beta*m2 * (np.dot(TravelMat.transpose(), I2)- np.sum(TravelMat * I2[:, None], 1)) +
                beta_A*(np.dot(TravelMat.transpose(), A2) - np.sum(TravelMat * A2[:, None], 1))
                )/Pop

    dS = -np.multiply(mylambda, S)  + omega1*(V1) +\
          omega2*(V2)  + sigma*(R + RA)


    dA = (1-k)*np.multiply(mylambda, S) -gamma*A

    dI = k * np.multiply(mylambda, S) - (gamma) * I

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA

    ########### Equations for those vaccinated with a single dose
    dV1 = -theta1*np.multiply(mylambda, V1) - omega1*V1 + sigma*(R1 + RA1)

    dA1 = (1 - k) * theta1 * np.multiply(mylambda, V1) - gamma1 * A1

    dI1 = k * theta1 * np.multiply(mylambda, V1) - (gamma1) * I1

    dR1 = gamma1 * (I1) - (sigma) * R1

    dRA1 = gamma1 * (A1) - (sigma) * RA1

    ########### Equations for those vaccinated with two doses:
    dV2 = -theta2*np.multiply(mylambda, V2) - omega2*V2 + sigma*(R2 + RA2)

    dA2 = (1 - k) * theta2 * np.multiply(mylambda, V2) - gamma2 * A2

    dI2 = k * theta2 * np.multiply(mylambda, V2) - (gamma2) * I2

    dR2 = gamma2*(I2)  - (sigma )*R2

    dRA2 = gamma2 * (A2) - (sigma ) * RA2

    ######## Water equations:
    dW = (mu * I + mu_A * A + mu1 * I1 + mu_A1 * A1 + mu2 * I2 + mu_A2 * A2)/D - delta * W + \
         frate*(np.dot(WaterMat.transpose(), W) - np.sum(WaterMat*W[:, None], 1))


    dydt = np.array([dS, dI, dA, dR, dRA, dV1, dI1, dA1, dR1, dRA1, dV2, dI2, dA2, dR2, dRA2, dW]).reshape((16 * numDept))

    return dydt



def choleraEqs4WithoutVaccination(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    This function is based on Jonathan's modified version of my model. That version can be found in the document
    cholera_haiti_equationsJonathan.docx

    Here, I will remove the death rate from the infectious compartments due to cholera

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW, D, delta, frate, k, gamma,  m,
     mu, mu_A, pseason, sigma, V] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (7, numDept))
    # print 'el primero es',np.shape(mynewarray)
    S = mynewarray[0, :]
    I = mynewarray[1, :]
    # print I
    A = mynewarray[2, :]
    R = mynewarray[3, :]
    RA = mynewarray[4, :]
    W = mynewarray[5, :]

    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) + \
               ((beta * I + beta_A * A)  +
                beta*m*(np.dot(TravelMat.transpose(), I) - np.sum(TravelMat*I[:, None], 1))  +
                beta_A*(np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))
                )/Pop

    dS = -np.multiply(mylambda, S) + sigma*(R + RA)

    dI = k * np.multiply(mylambda, S) - (gamma) * I

    dA = (1-k)*np.multiply(mylambda, S) -gamma*A

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA

    ######## Water equations:
    dW = (mu * I + mu_A * A)/D - delta * W + \
         frate*(np.dot(WaterMat.transpose(), W) - np.sum(WaterMat*W[:, None], 1))

    dC = k * np.multiply(mylambda, S)

    # print 'el segundo es', np.shape([dS, dI, dA, dR, dRA, dW])
    dydt = np.array([dS, dI, dA, dR, dRA, dW, dC]).reshape((7 * numDept))

    return dydt

def choleraEqs5WithoutVaccinationSingleGroup(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    This is different from choleraEqs4WithoutVaccination because I will add an exposed compartment here.
    This function is based on Jonathan's modified version of my model. That version can be found in the document
    cholera_haiti_equationsJonathan.docx

    Here, I will remove the death rate from the infectious compartments due to cholera

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, frate,  gamma, gammaA,  gammaE, k, m,
     mu, mu_A, pseason, sigma, V] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (8, numDept))
    # print 'el primero es',np.shape(mynewarray)
    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]
    W = mynewarray[6, :]
    C = mynewarray[7,:]


    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) + \
               ((beta * I + beta_A * A)  +
                beta*m*(np.dot(TravelMat.transpose(), I) - np.sum(TravelMat*I[:, None], 1))  +
                beta_A*(np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))
                )/Pop

    dS = -np.multiply(mylambda, S) + sigma*(R + RA)

    dE = np.multiply(mylambda, S) - gammaE * E

    dI = k * gammaE* E - (gamma) * I

    dA = (1-k)* gammaE* E -gammaA*A

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA

    ######## Water equations:
    dW = (mu * I + mu_A * A) - delta * W #+ \
         #frate*(np.dot(WaterMat.transpose(), W) - np.sum(WaterMat*W[:, None], 1))

    dC = k * gammaE * E

    # print 'el segundo es', np.shape([dS, dI, dA, dR, dRA, dW])
    dydt = np.array([dS, dE, dI, dA, dR, dRA, dW, dC]).reshape((8 * numDept))

    return dydt



def choleraEqs5WithoutVaccination(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    This is different from choleraEqs4WithoutVaccination because I will add an exposed compartment here.
    This function is based on Jonathan's modified version of my model. That version can be found in the document
    cholera_haiti_equationsJonathan.docx

    Here, I will remove the death rate from the infectious compartments due to cholera

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, frate,  gamma, gammaA,  gammaE, k, m,
     mu, mu_A, pseason, sigma, V] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (8, numDept))
    # print 'el primero es',np.shape(mynewarray)
    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]
    W = mynewarray[6, :]
    C = mynewarray[7,:]


    # mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) + \
    #            ((beta * I + beta_A * A)  +
    #             beta*m*(np.dot(TravelMat.transpose(), I) - np.sum(TravelMat*I[:, None], 1))  +
    #             beta_A*(np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))
    #             )/Pop

    #put lambda in weeks and shift the period some
    mylambda = 0.5 * (1 + aseason * np.cos((2*np.pi/pseason)*(t - 29))) * (betaW * W) / (V + W) + \
               ((beta * I + beta_A * A) +
                beta * m * (np.dot(TravelMat.transpose(), I) - np.sum(TravelMat * I[:, None], 1)) +
                beta_A * (np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))
                ) / Pop

    dS = -np.multiply(mylambda, S) + sigma*(R + RA)

    dE = np.multiply(mylambda, S) - gammaE * E

    dI = k * gammaE* E - (gamma) * I

    dA = (1-k)* gammaE* E -gammaA*A

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA

    ######## Water equations:
    dW = (mu * I + mu_A * A) - delta * W #+ \
         #frate*(np.dot(WaterMat.transpose(), W) - np.sum(WaterMat*W[:, None], 1))

    dC = k * gammaE * E

    # print 'el segundo es', np.shape([dS, dI, dA, dR, dRA, dW])
    dydt = np.array([dS, dE, dI, dA, dR, dRA, dW, dC]).reshape((8 * numDept))

    return dydt




def choleraEqs5WithoutVaccinationAll(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    This is different from choleraEqs4WithoutVaccination because I will add an exposed compartment here.
    This function is based on Jonathan's modified version of my model. That version can be found in the document
    cholera_haiti_equationsJonathan.docx

    Here, I will remove the death rate from the infectious compartments due to cholera

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, frate,  gamma, gammaA,  gammaE, k, m,
     mu, mu_A, pseason, sigma, V] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (8, numDept))

    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]
    W = mynewarray[6, :]
    C = mynewarray[7,:]


    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) + \
               ((beta * I + beta_A * A)  +
                beta*m*(np.dot(TravelMat.transpose(), I) - np.sum(TravelMat*I[:, None], 1))  +
                beta_A*(np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))
                )/Pop


    dS = -np.multiply(mylambda, S) + sigma*(R + RA)

    dE = np.multiply(mylambda, S) - gammaE * E

    dI = k * gammaE* E - (gamma) * I

    dA = (1-k)* gammaE* E -gammaA*A

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA

    ######## Water equations:
    dW = (mu * I + mu_A * A) - delta * W + \
         frate*(np.dot(WaterMat.transpose(), W) - np.sum(WaterMat*W[:, None], 1))

    dC = k * gammaE * E

    # print 'el segundo es', np.shape([dS, dI, dA, dR, dRA, dW])
    dydt = np.array([dS, dE, dI, dA, dR, dRA, dW, dC]).reshape((8 * numDept))

    return dydt




def choleraEqsSingleGroupWithVaccination(y, t, params, Pop):
    """
    This is different from choleraEqs5WithoutVaccination because I will add vaccination

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW, delta, gamma, gammaA,  gammaE, k,
     mu, mu_A, omega, pseason, sigma, theta, V] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (14, 1))
    # print 'el primero es',np.shape(mynewarray)
    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]
    V1 = mynewarray[6, :]
    E1 = mynewarray[7, :]
    I1 = mynewarray[8, :]
    A1 = mynewarray[9, :]
    R1 = mynewarray[10, :]
    RA_1 = mynewarray[11, :]
    W = mynewarray[12, :]
    C = mynewarray[13, :]


    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/( V + W) + \
               (beta * (I + I1) + beta_A * (A + A1))



    dS = -np.multiply(mylambda, S) + sigma*(R + RA) + omega * V1

    dE = np.multiply(mylambda, S) - gammaE * E

    dI = k * gammaE* E - (gamma) * I

    dA = (1-k)* gammaE* E - gammaA *A

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA


    dV1 = -theta* np.multiply(mylambda, V) + sigma*(R1 + RA_1) - omega * V1

    dE1 = theta* np.multiply(mylambda, V) - gammaE * E1

    dI1 = k * gammaE * E1 - (gamma) * I1

    dA1 = (1 - k) * gammaE * E1 - gammaA * A1

    dR1 = gamma * I1 - sigma * R1

    dRA_1 = gamma * A1 - sigma * RA_1

    ######## Water equations:
    dW = (mu * (I + I1) + mu_A * (A + A1)) - delta * W

    dC = k * gammaE * (E + E1)

    # print 'el segundo es', np.shape([dS, dI, dA, dR, dRA, dW])
    dydt = np.array([dS, dE, dI, dA, dR, dRA, dV1, dE1, dI1, dA1, dR1, dRA_1,dW, dC]).reshape((14))

    return dydt



def choleraEqs6WithoutVaccination(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    This is different from choleraEqs5WithoutVaccination because I will add stochastic noise to the
    Here, I will remove the death rate from the infectious compartments due to cholera

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, frate,  gamma, gammaA,  gammaE, k, m,
     mu, mu_A, pseason, sigma, V, var_p, var_w] = params
    mean = 0
    xi_p = np.random.normal(mean, var_p, size= numDept)#, size=num_samples), dtype='float64')
    xi_w = np.random.normal(mean, var_w, size= numDept)


    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (8, numDept))
    # print 'el primero es',np.shape(mynewarray)
    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]
    W = mynewarray[6, :]
    C = mynewarray[7,:]


    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*((1 + var_w *xi_w)*(betaW*W)/(V+W)) + \
               (1+ var_p*xi_p)*((beta * I + beta_A * A)  +
                beta*m*(np.dot(TravelMat.transpose(), I) - np.sum(TravelMat*I[:, None], 1))  +\
                beta_A*(np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))
                )/Pop


    dS = -np.multiply(mylambda, S) + sigma*(R + RA)

    dE = np.multiply(mylambda, S) - gammaE * E

    dI = k * gammaE* E - (gamma) * I

    dA = (1-k)* gammaE* E -gammaA*A

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA

    ######## Water equations:
    dW = (mu * I + mu_A * A) - delta * W #+ \
         #frate*(np.dot(WaterMat.transpose(), W) - np.sum(WaterMat*W[:, None], 1))

    dC = k * gammaE * E


    dydt = np.array([dS, dE, dI, dA, dR, dRA, dW, dC]).reshape((8 * numDept))

    return dydt


def stochasticSIR(y,t, params):
    beta, gamma, var_w = params

    xi_w = np.random.normal(0, var_w)
    print xi_w
    dy1 = -beta*y[0]*y[1]*((1 + var_w *xi_w))
    dy2 = beta*y[0]*y[1]*((1 + var_w *xi_w)) - gamma*y[1]
    dy3 = gamma*y[1]

    return [dy1, dy2, dy3]


def choleraEqs7WithoutVaccination(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    this function is analogous to choleraEqs3 but here I will remove the vaccinated equations (this set of equations is
    used only for fitting) and I will add an exposed class. Also, I modified my sinusoidal function to reflect that the
    peak of the rainy season in Haiti happens in May, where we assumed that our first data point is on 10/23/2010 and the peak
    occurs on May 15th
    :param y:
    :param t:
    :param params: the parameters of the model. they are described below
    :param TravelMat: a matrix of the flux of travelers between patch i and patch j. TravelMat
    :return:
    """
    #parameters

    #k = fraction of symptomatic cases
    #delta = bacteria decay in water
    #mu, mu1, mu2 = rate of excretion of V.cholera symptomatic unvaccinated, vacc with one dose, and vacc with 2 doses
    # mu, mu1, mu2 = rate of excretion of V.cholera Asymptomatic unvaccinated, vacc with one dose, and vacc with 2 doses


    [aseason, beta, beta_A, betaW, delta, gamma, gammaE, gammaA, k,
     m, mu, mu_A, pseason, sigma, V, wRate] = params
    # print gammaE

    #split the variables to compute the ODE's
    mynewarray = np.reshape(y, (8, numDept))

    S = mynewarray[0, :]
    E = mynewarray[1, :]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]
    W = mynewarray[6, :]
    C = mynewarray[7, :]

    #compute the force of infection:
    # mylambda = (beta*I + beta_A*A)/Pop + \
    #            0.5*(1 + aseason * np.cos((2*np.pi/pseason)*(t - 204)))*(betaW*W)/(V+W)   #days

    mylambda = (beta*I + beta_A*A) + \
               0.5*(1 + aseason * np.cos((2*np.pi/pseason)*(t - 29)))*(betaW*W)/(V+W)     #weeks
    # print beta*I
    ########### Equations for those unvaccinated:

    dS = -np.multiply(mylambda, S) + sigma*(R + RA)

    dE = np.multiply(mylambda, S) - gammaE * E


    dI = k * gammaE * E - (gamma) * I + m*(np.dot(TravelMat.transpose(), I) -
                                                                   np.sum(TravelMat*I[:, None], 1))

    dA = (1 - k) * gammaE * E - gammaA * A + np.dot(TravelMat.transpose(), A) - np.sum(TravelMat*A[:, None], 1)

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA


    ######## Water equations:
    dW = mu*I + mu_A*A  - delta*W + wRate*(np.dot(WaterMat.transpose(), W) - \
        np.sum(WaterMat*W[:, None], 1))

    #number of cases, for bookkeeping purposes only:
    dC = k * gammaE * E


    dydt = np.array([dS, dE, dI, dA, dR, dRA, dW, dC]).reshape((8 * numDept))
    return dydt



def choleraEqs7AllgroupsWithChildrenUnder5(y, t, params, TravelMat, WaterMat, numDept, Pop):
    """
    this function is analogous to choleraEqs7 but here I will add the vaccinated equations
    :param y:
    :param t:
    :param params: the parameters of the model. they are described below
    :param TravelMat: a matrix of the flux of travelers between patch i and patch j. TravelMat
    :return:
    """
    #parameters

    #k = fraction of symptomatic cases
    #delta = bacteria decay in water
    #mu, mu1, mu2 = rate of excretion of V.cholera symptomatic unvaccinated, vacc with one dose, and vacc with 2 doses
    # mu, mu1, mu2 = rate of excretion of V.cholera Asymptomatic unvaccinated, vacc with one dose, and vacc with 2 doses

    [aseason, beta, beta_A, betaW, delta, gamma,  gammaA, gammaE, k,
     m, mu, mu_A, omega1, omega2, pseason, sigma, theta1, theta1_under5, theta2, theta2_under5, V, wRate] = params

    # print gammaE

    # print [theta1, theta1_under5, theta2, theta2_under5]
    #split the variables to compute the ODE's
    mynewarray = np.reshape(y, (36, numDept))
    # print mynewarray
    S = mynewarray[0,:]
    E = mynewarray[1,:]
    I = mynewarray[2,:]
    A = mynewarray[3,:]
    R = mynewarray[4,:]
    RA = mynewarray[5,:]
    V1 = mynewarray[6,:]
    E1 = mynewarray[7,:]
    I1 = mynewarray[8,:]
    A1 = mynewarray[9,:]
    R1 = mynewarray[10,:]
    RA1 = mynewarray[11, :]
    V2 = mynewarray[12,:]
    E2 = mynewarray[13, :]
    I2 = mynewarray[14,:]
    A2 = mynewarray[15,:]
    R2 = mynewarray[16,:]
    RA2 = mynewarray[17, :]
    #children under 5:
    V1_under5 = mynewarray[18,:]
    E1_under5 = mynewarray[19, :]
    I1_under5 = mynewarray[20,:]
    A1_under5 = mynewarray[21,:]
    R1_under5 = mynewarray[22,:]
    RA1_under5 = mynewarray[23, :]
    V2_under5 = mynewarray[24,:]
    E2_under5 = mynewarray[25, :]
    I2_under5 = mynewarray[26,:]
    A2_under5 = mynewarray[27,:]
    R2_under5 = mynewarray[28,:]
    RA2_under5 = mynewarray[29, :]
    #water compartment:
    W  = mynewarray[30,:]
    #number of new cases for bookkepping purposes
    C = mynewarray[31,:]
    C1 = mynewarray[32,:]
    C2 = mynewarray[33,:]
    C1_under5 = mynewarray[34,:]
    C2_under5 = mynewarray[35,:]


    #compute the force of infection:
    mylambda = (beta*(I + I1 +  I2 + I1_under5 + I2_under5) + beta_A*(A + A1 + A2 + A1_under5 + A2_under5))/Pop + \
               0.5*(1 + aseason * np.cos((2*np.pi/pseason)*(t - 29)))*(betaW*W)/(V+W)


    ########### Equations for those unvaccinated:

    dS = -np.multiply(mylambda, S) + sigma * (R + RA) + omega1 * (V1 + V1_under5) + omega2 * (V2 + V2_under5)

    dE = np.multiply(mylambda, S) - gammaE * E

    dI = k * gammaE *E - gamma * I + m * (np.dot(TravelMat.transpose(), I) -
                                             np.sum(TravelMat * I[:, None], 1))

    dA = (1 - k) * gammaE *E - gammaA * A + np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1)

    dR = gamma * I - sigma * R

    dRA = gammaA * A - sigma * RA

    ########### Equations for those vaccinated with a single dose:
    dV1 = -theta1 * np.multiply(mylambda, V1) - omega1 * V1 + sigma * (R1 + RA1)

    dE1 = theta1 * np.multiply(mylambda, V1) - gammaE * E1

    dI1 = k * gammaE * E1 - gamma * I1 + m * (
                np.dot(TravelMat.transpose(), I1) -
                np.sum(TravelMat * I1[:, None], 1))

    dA1 = (1 - k) * gammaE * E1 - gammaA * A1 + np.dot(TravelMat.transpose(), A1) - \
          np.sum(TravelMat * A1[:, None], 1)

    dR1 = gamma * (I1) - (sigma) * R1

    dRA1 = gammaA * (A1) - (sigma) * RA1


    ########### Equations for those vaccinated with two doses:
    dV2 = -theta2 * np.multiply(mylambda, V2) - omega2 * V2 + sigma * (R2 + RA2)

    dE2 = theta2 * np.multiply(mylambda, V2) - gammaE * E2

    dI2 = k * gammaE * E2 - gamma * I2 + m * (
                np.dot(TravelMat.transpose(), I2)
                - np.sum(TravelMat * I2[:, None], 1))

    dA2 = (1 - k) * gammaE * E2 - gammaA * A2 + np.dot(TravelMat.transpose(), A2) - \
          np.sum(TravelMat * A2[:, None], 1)

    dR2 = gamma  * I2 - (sigma) * R2

    dRA2 = gammaA * A2 - (sigma) * RA2


    ######### Equations for children under 5 years:
    dV1_under5 = -theta1_under5 * np.multiply(mylambda, V1_under5) - omega1 * V1_under5 + sigma * (R1_under5 + RA1_under5)

    dE1_under5 = theta1_under5 * np.multiply(mylambda, V1_under5) - gammaE * E1_under5

    dI1_under5 = k * gammaE * E1_under5 - gamma * I1_under5 + m * (
                np.dot(TravelMat.transpose(), I1_under5) -
                np.sum(TravelMat * I1_under5[:, None], 1))

    dA1_under5 = (1 - k) * gammaE * E1_under5 - gammaA * A1_under5 + np.dot(TravelMat.transpose(), A1_under5) - \
          np.sum(TravelMat * A1_under5[:, None], 1)

    dR1_under5 = gamma * I1_under5 - (sigma) * R1_under5

    dRA1_under5 = gammaA * A1_under5 - (sigma) * RA1_under5



    ########### Equations for those vaccinated with two doses:
    dV2_under5 = -theta2_under5 * np.multiply(mylambda, V2_under5) - omega2 * V2_under5 + sigma * (R2_under5 + RA2_under5)

    dE2_under5 = theta2_under5 * np.multiply(mylambda, V2_under5) - gammaE * E2_under5

    dI2_under5 = k * gammaE * E2_under5 - gamma * I2_under5 + m * (
                np.dot(TravelMat.transpose(), I2_under5)
                - np.sum(TravelMat * I2_under5[:, None], 1))

    dA2_under5 = (1 - k) * gammaE * E2_under5 - gammaA * A2_under5 + np.dot(TravelMat.transpose(), A2_under5) - \
          np.sum(TravelMat * A2_under5[:, None], 1)

    dR2_under5 = gamma  * I2_under5 - (sigma) * R2_under5

    dRA2_under5 = gammaA * A2_under5 - (sigma) * RA2_under5

    ######## Water equations:
    dW = mu * (I + I1 + I2 + I1_under5 + I2_under5) + mu_A * (A + A1 + A2  + A1_under5 + A2_under5) - \
          delta*W + wRate*(np.dot(WaterMat.transpose(), W) - \
        np.sum(WaterMat*W[:, None], 1))

    # number of cases, for bookkeeping purposes only:
    dC = k * gammaE * E

    dC1 = k * gammaE * E1

    dC2 = k * gammaE * E2

    dC1_under5 = k * gammaE * E1_under5

    dC2_under5 = k * gammaE * E2_under5


    dydt = np.array([dS, dE, dI, dA, dR, dRA,
                     dV1, dE1, dI1, dA1, dR1, dRA1,
                     dV2, dE2, dI2, dA2, dR2, dRA2,
                     dV1_under5, dE1_under5, dI1_under5, dA1_under5, dR1_under5, dRA1_under5,
                     dV2_under5, dE2_under5, dI2_under5, dA2_under5, dR2_under5, dRA2_under5,
                     dW,
                     dC, dC1, dC2, dC1_under5, dC2_under5]).reshape((36 * numDept))

    return dydt






def choleraEqs8WithoutVaccinationSingleGroup(y, t, params):
    """
    This is different from choleraEqs5WithoutVaccination because the force of infection will include a mass action term
    instead of a standard incidence term and I will remove the parameters that are a nuissance for the single group

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, mu, mu_A, pseason, sigma, V] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (8, 1))

    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]
    W = mynewarray[6, :]
    C = mynewarray[7,:]


    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) + (beta * I + beta_A * A)

    dS = -np.multiply(mylambda, S) + sigma*(R + RA)

    dE = np.multiply(mylambda, S) - gammaE * E

    dI = k * gammaE* E - (gamma) * I

    dA = (1-k)* gammaE* E -gammaA*A

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA

    ######## Water equations:
    dW = (mu * I + mu_A * A) - delta * W

    dC = k * gammaE * E


    dydt = np.array([dS, dE, dI, dA, dR, dRA, dW, dC]).reshape((8 * 1))

    return dydt



def choleraEqs8WithVaccinationSingleGroup(y, t, params):
    """
    This is different from choleraEqs5WithoutVaccination because the force of infection will include a mass action term
    instead of a standard incidence term and I will remove the parameters that are a nuissance for the single group

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, mu, mu_A, \
     omega1, omega2, pseason, sigma, theta1,  theta1_5, theta2, theta2_5,  V] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (36, 1))
    # print 'el primero es',np.shape(mynewarray)
    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]

    #vaccinated equations with one dose
    V1 = mynewarray[6, :]
    E1 = mynewarray[7, :]
    I1 = mynewarray[8, :]
    A1 = mynewarray[9, :]
    R1 = mynewarray[10, :]
    RA1 = mynewarray[11, :]
    # vaccinated equations with two dose
    V2 = mynewarray[12, :]
    E2 = mynewarray[13, :]
    I2 = mynewarray[14, :]
    A2 = mynewarray[15, :]
    R2 = mynewarray[16, :]
    RA2 = mynewarray[17, :]

    #vaccinated equations one dose <5:
    V1_5 = mynewarray[18, :]
    E1_5 = mynewarray[19,:]
    I1_5 = mynewarray[20, :]
    A1_5 = mynewarray[21, :]
    R1_5 = mynewarray[22, :]
    RA1_5 = mynewarray[23, :]


    #vaccinated equations two doses <5:
    V2_5 = mynewarray[24, :]
    E2_5 = mynewarray[25, :]
    I2_5 = mynewarray[26, :]
    A2_5 = mynewarray[27, :]
    R2_5 = mynewarray[28, :]
    RA2_5 = mynewarray[29, :]

    #water equations
    W = mynewarray[30, :]

    #equations of the number of new infections for bookepping purposes
    C = mynewarray[31,:]
    C1 = mynewarray[32, :]
    C2 = mynewarray[33, :]
    C1_5 = mynewarray[34, :]
    C2_5 = mynewarray[35, :]

    #force of infection
    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) +\
               (beta * (I + I1 + I2 + I1_5 + I2_5) +
                beta_A * (A + A1 + A2 + A1_5 + A2_5))


    #unvaccinated equations
    dS = -np.multiply(mylambda, S) + sigma*(R + RA) + omega1*(V1 + V1_5) + omega2*(V2 + V2_5)

    dE = np.multiply(mylambda, S) - gammaE * E

    dI = k * gammaE* E - (gamma) * I

    dA = (1-k)* gammaE* E -gammaA*A

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA


    ##### Vaccinated with one dose over 5:
    dV1 = -theta1* np.multiply(mylambda, V1) + sigma*(R1 + RA1) -omega1*V1

    dE1 = theta1* np.multiply(mylambda, V1) - gammaE*E1

    dI1 = k * gammaE* E1 - gamma*I1

    dA1 = (1 - k)* gammaE* E1 - gammaA*A1

    dR1 = gamma*I1 - sigma*R1

    dRA1 = gammaA*A1 - sigma*RA1

    ##### Vaccinated with two doses over 5:
    dV2 = -theta2*np.multiply(mylambda, V2) + sigma*(R2 + RA2) - omega2*V2

    dE2 = theta2*np.multiply(mylambda, V2) - gammaE*E2

    dI2 = k* gammaE*E2 - gamma*I2

    dA2 = (1-k)*gammaE*E2 - gammaA*A2

    dR2 = gamma*I2 - sigma*R2

    dRA2 = gammaA*A2 - sigma*RA2

    ##### Vaccinated with one dose under 5:
    dV1_5 = -theta1_5* np.multiply(mylambda, V1_5) + sigma*(R1_5 + RA1_5) - omega1*V1_5

    dE1_5 = theta1_5* np.multiply(mylambda, V1_5) - gammaE*E1_5

    dI1_5 = k * gammaE* E1_5 - gamma*I1_5

    dA1_5 = (1 - k)* gammaE* E1_5 - gammaA*A1_5

    dR1_5 = gamma*I1_5 - sigma*R1_5

    dRA1_5 = gammaA*A1_5 - sigma*RA1_5

    ##### Vaccinated with two doses under 5:
    dV2_5 = -theta2_5 * np.multiply(mylambda, V2_5) + sigma * (R2_5 + RA2_5) - omega2 * V2_5

    dE2_5 = theta2_5 * np.multiply(mylambda, V2_5) - gammaE * E2_5

    dI2_5 = k * gammaE * E2_5 - gamma * I2_5

    dA2_5 = (1 - k) * gammaE * E2_5 - gammaA * A2_5

    dR2_5 = gamma * I2_5 - sigma * R2_5

    dRA2_5 = gammaA * A2_5 - sigma * RA2_5


    ######## Water equations:
    dW = (mu * (I + I1 + I2 + I1_5 + I2_5)  + mu_A * (A + A1 + A2 + A1_5 + A2_5)) - delta * W


    #bookeeping of number of new infections:
    dC = k * gammaE * E
    dC1 = k * gammaE* E1
    dC2 = k * gammaE * E2
    dC1_5 = k * gammaE* E1_5
    dC2_5 = k * gammaE* E2_5


    dydt = np.array([dS, dE, dI, dA, dR, dRA,
                     dV1, dE1, dI1, dA1, dR1, dRA1,
                     dV2, dE2, dI2, dA2, dR2, dRA2,
                     dV1_5, dE1_5, dI1_5, dA1_5, dR1_5, dRA1_5,
                     dV2_5, dE2_5, dI2_5, dA2_5, dR2_5, dRA2_5,
                     dW,
                     dC, dC1, dC2, dC1_5, dC2_5]).reshape((36 * 1))

    return dydt




def choleraEqs8WithVaccinationNetwork(y, t, params):
    """
    This is different from choleraEqs5WithoutVaccination because the force of infection will include a mass action term
    instead of a standard incidence term

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, m, mu, mu_A, numDep, \
     omega1, omega2, pseason, sigma, theta1,  theta1_5, theta2, theta2_5, TravelMat, V, wRate, waterMat] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (36, numDep))
    # print 'el primero es',np.shape(mynewarray)
    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]

    #vaccinated equations with one dose
    V1 = mynewarray[6, :]
    E1 = mynewarray[7, :]
    I1 = mynewarray[8, :]
    A1 = mynewarray[9, :]
    R1 = mynewarray[10, :]
    RA1 = mynewarray[11, :]
    # vaccinated equations with two dose
    V2 = mynewarray[12, :]
    E2 = mynewarray[13, :]
    I2 = mynewarray[14, :]
    A2 = mynewarray[15, :]
    R2 = mynewarray[16, :]
    RA2 = mynewarray[17, :]

    #vaccinated equations one dose <5:
    V1_5 = mynewarray[18, :]
    E1_5 = mynewarray[19,:]
    I1_5 = mynewarray[20, :]
    A1_5 = mynewarray[21, :]
    R1_5 = mynewarray[22, :]
    RA1_5 = mynewarray[23, :]


    #vaccinated equations two doses <5:
    V2_5 = mynewarray[24, :]
    E2_5 = mynewarray[25, :]
    I2_5 = mynewarray[26, :]
    A2_5 = mynewarray[27, :]
    R2_5 = mynewarray[28, :]
    RA2_5 = mynewarray[29, :]

    #water equations
    W = mynewarray[30, :]

    #equations of the number of new infections for bookepping purposes
    C = mynewarray[31,:]
    C1 = mynewarray[32, :]
    C2 = mynewarray[33, :]
    C1_5 = mynewarray[34, :]
    C2_5 = mynewarray[35, :]

    #force of infection
    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) +\
               (beta * (I + I1 + I2 + I1_5 + I2_5) +
                beta_A * (A + A1 + A2 + A1_5 + A2_5))


    #unvaccinated equations
    dS = -np.multiply(mylambda, S) + sigma*(R + RA) + omega1*(V1 + V1_5) + omega2*(V2 + V2_5)

    dE = np.multiply(mylambda, S) - gammaE * E

    dI = k * gammaE* E - (gamma) * I  + m * (np.dot(TravelMat.transpose(), I) - np.sum(TravelMat * I[:, None], 1))

    dA = (1-k)* gammaE* E -gammaA*A + (np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))

    dR = gamma*I -sigma*R

    dRA = gamma * A - sigma * RA


    ##### Vaccinated with one dose over 5:
    dV1 = -theta1* np.multiply(mylambda, V1) + sigma*(R1 + RA1) -omega1*V1

    dE1 = theta1* np.multiply(mylambda, V1) - gammaE*E1

    dI1 = k * gammaE* E1 - gamma*I1 + m * (np.dot(TravelMat.transpose(), I1) - np.sum(TravelMat * I1[:, None], 1))

    dA1 = (1 - k)* gammaE* E1 - gammaA*A1 + (np.dot(TravelMat.transpose(), A1) - np.sum(TravelMat * A1[:, None], 1))

    dR1 = gamma*I1 - sigma*R1

    dRA1 = gammaA*A1 - sigma*RA1

    ##### Vaccinated with two doses over 5:
    dV2 = -theta2*np.multiply(mylambda, V2) + sigma*(R2 + RA2) - omega2*V2

    dE2 = theta2*np.multiply(mylambda, V2) - gammaE*E2

    dI2 = k* gammaE*E2 - gamma*I2 + m * (np.dot(TravelMat.transpose(), I2) - np.sum(TravelMat * I2[:, None], 1))

    dA2 = (1-k)*gammaE*E2 - gammaA*A2 + (np.dot(TravelMat.transpose(), A2) - np.sum(TravelMat * A2[:, None], 1))

    dR2 = gamma*I2 - sigma*R2

    dRA2 = gammaA*A2 - sigma*RA2

    ##### Vaccinated with one dose under 5:
    dV1_5 = -theta1_5* np.multiply(mylambda, V1_5) + sigma*(R1_5 + RA1_5) - omega1*V1_5

    dE1_5 = theta1_5* np.multiply(mylambda, V1_5) - gammaE*E1_5

    dI1_5 = k * gammaE* E1_5 - gamma*I1_5 + m * (np.dot(TravelMat.transpose(), I1_5) - np.sum(TravelMat * I1_5[:, None], 1))

    dA1_5 = (1 - k)* gammaE* E1_5 - gammaA*A1_5 + (np.dot(TravelMat.transpose(), A1_5) - np.sum(TravelMat * A1_5[:, None], 1))

    dR1_5 = gamma*I1_5 - sigma*R1_5

    dRA1_5 = gammaA*A1_5 - sigma*RA1_5

    ##### Vaccinated with two doses under 5:
    dV2_5 = -theta2_5 * np.multiply(mylambda, V2_5) + sigma * (R2_5 + RA2_5) - omega2 * V2_5

    dE2_5 = theta2_5 * np.multiply(mylambda, V2_5) - gammaE * E2_5

    dI2_5 = k * gammaE * E2_5 - gamma * I2_5 + m * (np.dot(TravelMat.transpose(), I2_5) - np.sum(TravelMat * I2_5[:, None], 1))

    dA2_5 = (1 - k) * gammaE * E2_5 - gammaA * A2_5 + (np.dot(TravelMat.transpose(), A2_5) - np.sum(TravelMat * A2_5[:, None], 1))

    dR2_5 = gamma * I2_5 - sigma * R2_5

    dRA2_5 = gammaA * A2_5 - sigma * RA2_5


    ######## Water equations:
    dW = (mu * (I + I1 + I2 + I1_5 + I2_5)  + mu_A * (A + A1 + A2 + A1_5 + A2_5)) - delta * W  \
         + wRate*(np.dot(waterMat.transpose(), W) - np.sum(waterMat*W[:, None], 1))

    #bookeeping of number of new infections:
    dC = k * gammaE * E
    dC1 = k * gammaE* E1
    dC2 = k * gammaE * E2
    dC1_5 = k * gammaE* E1_5
    dC2_5 = k * gammaE* E2_5


    dydt = np.array([dS, dE, dI, dA, dR, dRA,
                     dV1, dE1, dI1, dA1, dR1, dRA1,
                     dV2, dE2, dI2, dA2, dR2, dRA2,
                     dV1_5, dE1_5, dI1_5, dA1_5, dR1_5, dRA1_5,
                     dV2_5, dE2_5, dI2_5, dA2_5, dR2_5, dRA2_5,
                     dW,
                     dC, dC1, dC2, dC1_5, dC2_5]).reshape((36 * numDep))

    return dydt




def choleraEqs10WithVaccinationNetwork(y, t, params):
    """
    This is different from choleraEqs8WithVaccinationNetwork because here the movement between departments will be followed
    in all the classes not only in the infected classes

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, m, mu, mu_A, numDep, \
     omega1, omega2, pseason, sigma, theta1,  theta1_5, theta2, theta2_5, TravelMat, V, wRate, waterMat] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (36, numDep))

    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]

    #vaccinated equations with one dose
    V1 = mynewarray[6, :]
    E1 = mynewarray[7, :]
    I1 = mynewarray[8, :]
    A1 = mynewarray[9, :]
    R1 = mynewarray[10, :]
    RA1 = mynewarray[11, :]
    # vaccinated equations with two dose
    V2 = mynewarray[12, :]
    E2 = mynewarray[13, :]
    I2 = mynewarray[14, :]
    A2 = mynewarray[15, :]
    R2 = mynewarray[16, :]
    RA2 = mynewarray[17, :]

    #vaccinated equations one dose <5:
    V1_5 = mynewarray[18, :]
    E1_5 = mynewarray[19,:]
    I1_5 = mynewarray[20, :]
    A1_5 = mynewarray[21, :]
    R1_5 = mynewarray[22, :]
    RA1_5 = mynewarray[23, :]


    #vaccinated equations two doses <5:
    V2_5 = mynewarray[24, :]
    E2_5 = mynewarray[25, :]
    I2_5 = mynewarray[26, :]
    A2_5 = mynewarray[27, :]
    R2_5 = mynewarray[28, :]
    RA2_5 = mynewarray[29, :]

    #water equations
    W = mynewarray[30, :]

    #equations of the number of new infections for bookepping purposes
    C = mynewarray[31,:]
    C1 = mynewarray[32, :]
    C2 = mynewarray[33, :]
    C1_5 = mynewarray[34, :]
    C2_5 = mynewarray[35, :]

    #force of infection
    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) +\
               (beta * (I + I1 + I2 + I1_5 + I2_5) +
                beta_A * (A + A1 + A2 + A1_5 + A2_5))


    #unvaccinated equations
    dS = -np.multiply(mylambda, S) + sigma*(R + RA) + omega1*(V1 + V1_5) + omega2*(V2 + V2_5) \
         + (np.dot(TravelMat.transpose(), S) - np.sum(TravelMat * S[:, None], 1))

    dE = np.multiply(mylambda, S) - gammaE * E + (np.dot(TravelMat.transpose(), E) - np.sum(TravelMat * E[:, None], 1))

    dI = k * gammaE* E - (gamma) * I  + m * (np.dot(TravelMat.transpose(), I) - np.sum(TravelMat * I[:, None], 1))

    dA = (1-k)* gammaE* E -gammaA*A + (np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))

    dR = gamma*I -sigma*R + (np.dot(TravelMat.transpose(), R) - np.sum(TravelMat * R[:, None], 1))

    dRA = gamma * A - sigma * RA + (np.dot(TravelMat.transpose(), RA) - np.sum(TravelMat * RA[:, None], 1))


    ##### Vaccinated with one dose over 5:
    dV1 = -theta1* np.multiply(mylambda, V1) + sigma*(R1 + RA1) -omega1*V1 + (np.dot(TravelMat.transpose(), V1) - np.sum(TravelMat * V1[:, None], 1))

    dE1 = theta1* np.multiply(mylambda, V1) - gammaE*E1 + (np.dot(TravelMat.transpose(), E1) - np.sum(TravelMat * E1[:, None], 1))

    dI1 = k * gammaE* E1 - gamma*I1 + m * (np.dot(TravelMat.transpose(), I1) - np.sum(TravelMat * I1[:, None], 1))

    dA1 = (1 - k)* gammaE* E1 - gammaA*A1 + (np.dot(TravelMat.transpose(), A1) - np.sum(TravelMat * A1[:, None], 1))

    dR1 = gamma*I1 - sigma*R1 + (np.dot(TravelMat.transpose(), R1) - np.sum(TravelMat * R1[:, None], 1))

    dRA1 = gammaA*A1 - sigma*RA1 + (np.dot(TravelMat.transpose(), RA1) - np.sum(TravelMat * RA1[:, None], 1))

    ##### Vaccinated with two doses over 5:
    dV2 = -theta2*np.multiply(mylambda, V2) + sigma*(R2 + RA2) - omega2*V2 + (np.dot(TravelMat.transpose(), V2) - np.sum(TravelMat * V2[:, None], 1))

    dE2 = theta2*np.multiply(mylambda, V2) - gammaE*E2 + (np.dot(TravelMat.transpose(), E2) - np.sum(TravelMat * E2[:, None], 1))

    dI2 = k* gammaE*E2 - gamma*I2 + m * (np.dot(TravelMat.transpose(), I2) - np.sum(TravelMat * I2[:, None], 1))

    dA2 = (1-k)*gammaE*E2 - gammaA*A2 + (np.dot(TravelMat.transpose(), A2) - np.sum(TravelMat * A2[:, None], 1))

    dR2 = gamma*I2 - sigma*R2 + (np.dot(TravelMat.transpose(), R2) - np.sum(TravelMat * R2[:, None], 1))

    dRA2 = gammaA*A2 - sigma*RA2 + (np.dot(TravelMat.transpose(), RA2) - np.sum(TravelMat * RA2[:, None], 1))

    ##### Vaccinated with one dose under 5:
    dV1_5 = -theta1_5* np.multiply(mylambda, V1_5) + sigma*(R1_5 + RA1_5) - omega1*V1_5 + (np.dot(TravelMat.transpose(), V1_5) - np.sum(TravelMat * V1_5[:, None], 1))

    dE1_5 = theta1_5* np.multiply(mylambda, V1_5) - gammaE*E1_5 + (np.dot(TravelMat.transpose(), E1_5) - np.sum(TravelMat * E1_5[:, None], 1))

    dI1_5 = k * gammaE* E1_5 - gamma*I1_5 + m * (np.dot(TravelMat.transpose(), I1_5) - np.sum(TravelMat * I1_5[:, None], 1))

    dA1_5 = (1 - k)* gammaE* E1_5 - gammaA*A1_5 + (np.dot(TravelMat.transpose(), A1_5) - np.sum(TravelMat * A1_5[:, None], 1))

    dR1_5 = gamma*I1_5 - sigma*R1_5 + (np.dot(TravelMat.transpose(), R1_5) - np.sum(TravelMat * R1_5[:, None], 1))

    dRA1_5 = gammaA*A1_5 - sigma*RA1_5 + (np.dot(TravelMat.transpose(), RA1_5) - np.sum(TravelMat * RA1_5[:, None], 1))

    ##### Vaccinated with two doses under 5:
    dV2_5 = -theta2_5 * np.multiply(mylambda, V2_5) + sigma * (R2_5 + RA2_5) - omega2 * V2_5 + (np.dot(TravelMat.transpose(), V2_5) - np.sum(TravelMat * V2_5[:, None], 1))

    dE2_5 = theta2_5 * np.multiply(mylambda, V2_5) - gammaE * E2_5 + (np.dot(TravelMat.transpose(), E2_5) - np.sum(TravelMat * E2_5[:, None], 1))

    dI2_5 = k * gammaE * E2_5 - gamma * I2_5 + m * (np.dot(TravelMat.transpose(), I2_5) - np.sum(TravelMat * I2_5[:, None], 1))

    dA2_5 = (1 - k) * gammaE * E2_5 - gammaA * A2_5 + (np.dot(TravelMat.transpose(), A2_5) - np.sum(TravelMat * A2_5[:, None], 1))

    dR2_5 = gamma * I2_5 - sigma * R2_5 + (np.dot(TravelMat.transpose(), R2_5) - np.sum(TravelMat * R2_5[:, None], 1))

    dRA2_5 = gammaA * A2_5 - sigma * RA2_5 + (np.dot(TravelMat.transpose(), RA2_5) - np.sum(TravelMat * RA2_5[:, None], 1))


    ######## Water equations:
    dW = (mu * (I + I1 + I2 + I1_5 + I2_5)  + mu_A * (A + A1 + A2 + A1_5 + A2_5)) - delta * W  \
         + wRate*(np.dot(waterMat.transpose(), W) - np.sum(waterMat*W[:, None], 1))

    #bookeeping of number of new infections:
    dC = k * gammaE * E
    dC1 = k * gammaE* E1
    dC2 = k * gammaE * E2
    dC1_5 = k * gammaE* E1_5
    dC2_5 = k * gammaE* E2_5

    # print 'el segundo es', np.shape([dS, dI, dA, dR, dRA, dW])
    dydt = np.array([dS, dE, dI, dA, dR, dRA,
                     dV1, dE1, dI1, dA1, dR1, dRA1,
                     dV2, dE2, dI2, dA2, dR2, dRA2,
                     dV1_5, dE1_5, dI1_5, dA1_5, dR1_5, dRA1_5,
                     dV2_5, dE2_5, dI2_5, dA2_5, dR2_5, dRA2_5,
                     dW,
                     dC, dC1, dC2, dC1_5, dC2_5]).reshape((36 * numDep))

    return dydt




def choleraEqs10WithoutVaccinationNetwork(y, t, params):
    """
    This is different from choleraEqs8WithVaccinationNetwork because here the movement between departments will be followed
    in all the classes not only in the infected classes

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, m, mu, mu_A, numDep, \
     pseason, sigma,  TravelMat, V, wRate, waterMat] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (8, numDep))

    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]



    #water equations
    W = mynewarray[6, :]

    #equations of the number of new infections for bookepping purposes
    C = mynewarray[7,:]


    #force of infection
    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) +\
               (beta * (I ) +
                beta_A * (A ))


    #unvaccinated equations
    dS = -np.multiply(mylambda, S) + sigma*(R + RA)  \
         + (np.dot(TravelMat.transpose(), S) - np.sum(TravelMat * S[:, None], 1))

    dE = np.multiply(mylambda, S) - gammaE * E + (np.dot(TravelMat.transpose(), E) - np.sum(TravelMat * E[:, None], 1))

    dI = k * gammaE* E - (gamma) * I  + m * (np.dot(TravelMat.transpose(), I) - np.sum(TravelMat * I[:, None], 1))

    dA = (1-k)* gammaE* E -gammaA*A + (np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))

    dR = gamma*I -sigma*R + (np.dot(TravelMat.transpose(), R) - np.sum(TravelMat * R[:, None], 1))

    dRA = gamma * A - sigma * RA + (np.dot(TravelMat.transpose(), RA) - np.sum(TravelMat * RA[:, None], 1))




    ######## Water equations:
    dW = (mu * (I )  + mu_A * (A )) - delta * W  \
         + wRate*(np.dot(waterMat.transpose(), W) - np.sum(waterMat*W[:, None], 1))

    #bookeeping of number of new infections:
    dC = k * gammaE * E


    dydt = np.array([dS, dE, dI, dA, dR, dRA,
                     dW,
                     dC]).reshape((8 * numDep))

    return dydt



def choleraEqs11WithoutVaccinationNetwork(y, t, params):
    """
    This is different from choleraEqs11WithVaccinationNetwork because here we will keep track of all new infections, not
    only the symptomatic ones (the equations for dC/dt take into account ALL of the exposed).

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, m, mu, mu_A, numDep, \
     pseason, sigma,  TravelMat, V, wRate, waterMat] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (8, numDep))
    # print 'el primero es',np.shape(mynewarray)
    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]



    #water equations
    W = mynewarray[6, :]

    #equations of the number of new infections for bookepping purposes
    C = mynewarray[7,:]


    #force of infection
    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) +\
               (beta * (I ) +
                beta_A * (A ))


    #unvaccinated equations
    dS = -np.multiply(mylambda, S) + sigma*(R + RA)  \
         + (np.dot(TravelMat.transpose(), S) - np.sum(TravelMat * S[:, None], 1))

    dE = np.multiply(mylambda, S) - gammaE * E + (np.dot(TravelMat.transpose(), E) - np.sum(TravelMat * E[:, None], 1))

    dI = k * gammaE* E - (gamma) * I  + m * (np.dot(TravelMat.transpose(), I) - np.sum(TravelMat * I[:, None], 1))

    dA = (1-k)* gammaE* E -gammaA*A + (np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))

    dR = gamma*I -sigma*R + (np.dot(TravelMat.transpose(), R) - np.sum(TravelMat * R[:, None], 1))

    dRA = gamma * A - sigma * RA + (np.dot(TravelMat.transpose(), RA) - np.sum(TravelMat * RA[:, None], 1))




    ######## Water equations:
    dW = (mu * (I )  + mu_A * (A )) - delta * W  \
         + wRate*(np.dot(waterMat.transpose(), W) - np.sum(waterMat*W[:, None], 1))

    #bookeeping of number of new infections:
    dC = gammaE * E


    dydt = np.array([dS, dE, dI, dA, dR, dRA,
                     dW,
                     dC]).reshape((8 * numDep))

    return dydt


def choleraEqs11WithVaccinationNetwork(y, t, params):
    """
   This is different from choleraEqs11WithVaccinationNetwork because here we will keep track of all new infections, not
    only the symptomatic ones (the equations for dC/dt take into account ALL of the exposed).

    :param y:
    :param t:
    :param params:
    :param TravelMat:
    :param WaterMat:
    :param numDept:
    :param Pop:
    :return:
    """

    [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, m, mu, mu_A, numDep, \
     omega1, omega2, pseason, sigma, theta1,  theta1_5, theta2, theta2_5, TravelMat, V, wRate, waterMat] = params

    # split the variables to compute the ODE's
    # the susceptible classes are the first
    mynewarray = np.reshape(y, (36, numDep))

    S = mynewarray[0, :]
    E = mynewarray[1,:]
    I = mynewarray[2, :]
    A = mynewarray[3, :]
    R = mynewarray[4, :]
    RA = mynewarray[5, :]

    #vaccinated equations with one dose
    V1 = mynewarray[6, :]
    E1 = mynewarray[7, :]
    I1 = mynewarray[8, :]
    A1 = mynewarray[9, :]
    R1 = mynewarray[10, :]
    RA1 = mynewarray[11, :]
    # vaccinated equations with two dose
    V2 = mynewarray[12, :]
    E2 = mynewarray[13, :]
    I2 = mynewarray[14, :]
    A2 = mynewarray[15, :]
    R2 = mynewarray[16, :]
    RA2 = mynewarray[17, :]

    #vaccinated equations one dose <5:
    V1_5 = mynewarray[18, :]
    E1_5 = mynewarray[19,:]
    I1_5 = mynewarray[20, :]
    A1_5 = mynewarray[21, :]
    R1_5 = mynewarray[22, :]
    RA1_5 = mynewarray[23, :]


    #vaccinated equations two doses <5:
    V2_5 = mynewarray[24, :]
    E2_5 = mynewarray[25, :]
    I2_5 = mynewarray[26, :]
    A2_5 = mynewarray[27, :]
    R2_5 = mynewarray[28, :]
    RA2_5 = mynewarray[29, :]

    #water equations
    W = mynewarray[30, :]

    #equations of the number of new infections for bookepping purposes
    C = mynewarray[31,:]
    C1 = mynewarray[32, :]
    C2 = mynewarray[33, :]
    C1_5 = mynewarray[34, :]
    C2_5 = mynewarray[35, :]

    #force of infection
    mylambda = 0.5*(1 + aseason*np.cos(2*np.pi*(t /pseason)))*(betaW*W)/(V+W) +\
               (beta * (I + I1 + I2 + I1_5 + I2_5) +
                beta_A * (A + A1 + A2 + A1_5 + A2_5))


    #unvaccinated equations
    dS = -np.multiply(mylambda, S) + sigma*(R + RA) + omega1*(V1 + V1_5) + omega2*(V2 + V2_5) \
         + (np.dot(TravelMat.transpose(), S) - np.sum(TravelMat * S[:, None], 1))

    dE = np.multiply(mylambda, S) - gammaE * E + (np.dot(TravelMat.transpose(), E) - np.sum(TravelMat * E[:, None], 1))

    dI = k * gammaE* E - (gamma) * I  + m * (np.dot(TravelMat.transpose(), I) - np.sum(TravelMat * I[:, None], 1))

    dA = (1-k)* gammaE* E -gammaA*A + (np.dot(TravelMat.transpose(), A) - np.sum(TravelMat * A[:, None], 1))

    dR = gamma*I -sigma*R + (np.dot(TravelMat.transpose(), R) - np.sum(TravelMat * R[:, None], 1))

    dRA = gamma * A - sigma * RA + (np.dot(TravelMat.transpose(), RA) - np.sum(TravelMat * RA[:, None], 1))


    ##### Vaccinated with one dose over 5:
    dV1 = -theta1* np.multiply(mylambda, V1) + sigma*(R1 + RA1) -omega1*V1 + (np.dot(TravelMat.transpose(), V1) - np.sum(TravelMat * V1[:, None], 1))

    dE1 = theta1* np.multiply(mylambda, V1) - gammaE*E1 + (np.dot(TravelMat.transpose(), E1) - np.sum(TravelMat * E1[:, None], 1))

    dI1 = k * gammaE* E1 - gamma*I1 + m * (np.dot(TravelMat.transpose(), I1) - np.sum(TravelMat * I1[:, None], 1))

    dA1 = (1 - k)* gammaE* E1 - gammaA*A1 + (np.dot(TravelMat.transpose(), A1) - np.sum(TravelMat * A1[:, None], 1))

    dR1 = gamma*I1 - sigma*R1 + (np.dot(TravelMat.transpose(), R1) - np.sum(TravelMat * R1[:, None], 1))

    dRA1 = gammaA*A1 - sigma*RA1 + (np.dot(TravelMat.transpose(), RA1) - np.sum(TravelMat * RA1[:, None], 1))

    ##### Vaccinated with two doses over 5:
    dV2 = -theta2*np.multiply(mylambda, V2) + sigma*(R2 + RA2) - omega2*V2 + (np.dot(TravelMat.transpose(), V2) - np.sum(TravelMat * V2[:, None], 1))

    dE2 = theta2*np.multiply(mylambda, V2) - gammaE*E2 + (np.dot(TravelMat.transpose(), E2) - np.sum(TravelMat * E2[:, None], 1))

    dI2 = k* gammaE*E2 - gamma*I2 + m * (np.dot(TravelMat.transpose(), I2) - np.sum(TravelMat * I2[:, None], 1))

    dA2 = (1-k)*gammaE*E2 - gammaA*A2 + (np.dot(TravelMat.transpose(), A2) - np.sum(TravelMat * A2[:, None], 1))

    dR2 = gamma*I2 - sigma*R2 + (np.dot(TravelMat.transpose(), R2) - np.sum(TravelMat * R2[:, None], 1))

    dRA2 = gammaA*A2 - sigma*RA2 + (np.dot(TravelMat.transpose(), RA2) - np.sum(TravelMat * RA2[:, None], 1))

    ##### Vaccinated with one dose under 5:
    dV1_5 = -theta1_5* np.multiply(mylambda, V1_5) + sigma*(R1_5 + RA1_5) - omega1*V1_5 + (np.dot(TravelMat.transpose(), V1_5) - np.sum(TravelMat * V1_5[:, None], 1))

    dE1_5 = theta1_5* np.multiply(mylambda, V1_5) - gammaE*E1_5 + (np.dot(TravelMat.transpose(), E1_5) - np.sum(TravelMat * E1_5[:, None], 1))

    dI1_5 = k * gammaE* E1_5 - gamma*I1_5 + m * (np.dot(TravelMat.transpose(), I1_5) - np.sum(TravelMat * I1_5[:, None], 1))

    dA1_5 = (1 - k)* gammaE* E1_5 - gammaA*A1_5 + (np.dot(TravelMat.transpose(), A1_5) - np.sum(TravelMat * A1_5[:, None], 1))

    dR1_5 = gamma*I1_5 - sigma*R1_5 + (np.dot(TravelMat.transpose(), R1_5) - np.sum(TravelMat * R1_5[:, None], 1))

    dRA1_5 = gammaA*A1_5 - sigma*RA1_5 + (np.dot(TravelMat.transpose(), RA1_5) - np.sum(TravelMat * RA1_5[:, None], 1))

    ##### Vaccinated with two doses under 5:
    dV2_5 = -theta2_5 * np.multiply(mylambda, V2_5) + sigma * (R2_5 + RA2_5) - omega2 * V2_5 + (np.dot(TravelMat.transpose(), V2_5) - np.sum(TravelMat * V2_5[:, None], 1))

    dE2_5 = theta2_5 * np.multiply(mylambda, V2_5) - gammaE * E2_5 + (np.dot(TravelMat.transpose(), E2_5) - np.sum(TravelMat * E2_5[:, None], 1))

    dI2_5 = k * gammaE * E2_5 - gamma * I2_5 + m * (np.dot(TravelMat.transpose(), I2_5) - np.sum(TravelMat * I2_5[:, None], 1))

    dA2_5 = (1 - k) * gammaE * E2_5 - gammaA * A2_5 + (np.dot(TravelMat.transpose(), A2_5) - np.sum(TravelMat * A2_5[:, None], 1))

    dR2_5 = gamma * I2_5 - sigma * R2_5 + (np.dot(TravelMat.transpose(), R2_5) - np.sum(TravelMat * R2_5[:, None], 1))

    dRA2_5 = gammaA * A2_5 - sigma * RA2_5 + (np.dot(TravelMat.transpose(), RA2_5) - np.sum(TravelMat * RA2_5[:, None], 1))


    ######## Water equations:
    dW = (mu * (I + I1 + I2 + I1_5 + I2_5)  + mu_A * (A + A1 + A2 + A1_5 + A2_5)) - delta * W  \
         + wRate*(np.dot(waterMat.transpose(), W) - np.sum(waterMat*W[:, None], 1))

    #bookeeping of number of new infections:
    dC = gammaE * E
    dC1 = gammaE* E1
    dC2 = gammaE * E2
    dC1_5 = gammaE* E1_5
    dC2_5 = gammaE* E2_5


    dydt = np.array([dS, dE, dI, dA, dR, dRA,
                     dV1, dE1, dI1, dA1, dR1, dRA1,
                     dV2, dE2, dI2, dA2, dR2, dRA2,
                     dV1_5, dE1_5, dI1_5, dA1_5, dR1_5, dRA1_5,
                     dV2_5, dE2_5, dI2_5, dA2_5, dR2_5, dRA2_5,
                     dW,
                     dC, dC1, dC2, dC1_5, dC2_5]).reshape((36 * numDep))

    return dydt
