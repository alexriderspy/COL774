from typing import final
import numpy as np
import cvxopt

final_array = np.zeros((5,len(test_arrX))).tolist()

for l0 in range(5):
    for l1 in range(l0,5):
        arrY = []
        arrX = []

        #l0 -> -1, l1 -> 1

        for i in range(len(labels)):
            if labels[i] == l0:
                arrX.append(data[i].flatten())
                arrY.append(-1)
            elif labels[i]==l1:
                arrX.append(data[i].flatten())
                arrY.append(1)

        m = len(arrX)
        arrX = np.array(arrX).reshape(m,3072)
        arrY = np.array(arrY).reshape(m,1)

        arrX = np.multiply(arrX,1.0)

        C = 1.0

        # compute inputs for cvxopt solver
        K = (arrX * arrY).T
        P = cvxopt.matrix(K.T.dot(K)) # P has shape m*m
        q = cvxopt.matrix(-1 * np.ones(m)) # q has shape m*1
        G = cvxopt.matrix(np.concatenate((-1*np.identity(m), np.identity(m)), axis=0))
        h = cvxopt.matrix(np.concatenate((np.zeros(m), C*np.ones(m)), axis=0))
        A = cvxopt.matrix(1.0 * arrY, (1, m))
        b = cvxopt.matrix(0.0)
        # solve quadratic programming
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        _lambda = np.ravel(solution['x']).reshape(m,1)

        S = np.where((_lambda > 1e-10) & (_lambda <= C))[0]
        print('The number of support vectors are : ' + str(len(S)))
        print("Fraction of support vectors : " + str(len(S)/m))

        w = K[:, S].dot(_lambda[S])

        M = np.where((_lambda > 1e-10) & (_lambda < C))[0]
        b = np.mean(arrY[M] - arrX[M, :].dot(w))

        score = (test_arrX.dot(w) +b)
        
        for i in range(len(results)):
            if (score[i]<0 and test_arrY[i]==l0):
                if final_array[l0][i]==0:
                    final_array[l0][i] = (1,abs(score[i]))
                else:
                    prev = final_array[l0][i][0]
                    if abs(score[i]) > abs(final_array[l0][i][1]):
                        final_array[l0][i] = (prev+1,abs(score[i]))
                    else:
                        final_array[l0][i] = (prev+1,abs(final_array[l0][i][1]))
            elif (score[i]>0 and test_arrY[i]==l1):
                if final_array[l1][i]==0:
                    final_array[l1][i] = (1,abs(score[i]))
                else:
                    prev = final_array[l1][i][0]
                    if abs(score[i]) > abs(final_array[l1][i][1]):
                        final_array[l1][i] = (prev+1,abs(score[i]))
                    else:
                        final_array[l1][i] = (prev+1,abs(final_array[l1][i][1]))


        



