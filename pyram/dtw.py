import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import numpy.linalg as LA

def euclidean(x, y):
    return LA.norm(x - y, ord=2)

def manhattan(x, y):
    return LA.norm(x - y, ord=1)

def dtw(Y, X, step_list, subsequence=False, d = euclidean):
    # create the cost matrix
    M, N = Y.shape[0], X.shape[0]
    Y, X = np.reshape(Y, (M,-1)), np.reshape(X, (N,-1))
    cost = float('inf')*np.ones((N, M))

    # initialize the first column
    cost[0, 0] = d(X[0,:], Y[0,:])
    for i in range(1, N):
        cost[i, 0] = cost[i-1, 0] + d(X[i,:], Y[0,:])

    # initialize the first row
    for j in range(1, M):
        cost[0, j] = (0 if subsequence else cost[0, j-1]) + d(X[0,:], Y[j,:])

    # fill in the rest of the matrix
    for i in range(1, N):
        for j in range(1, M):
            cost_list = float('inf') * np.ones(len(step_list))
            for index, (step_i, step_j) in enumerate(step_list) :
                if (i-step_i >= 0 and i-step_i < N) and (j-step_j >= 0 and j-step_j < M):
                    cost_list[index] = cost[i-step_i, j-step_j]
            cost[i, j] = np.min(cost_list) + d(X[i,:], Y[j,:])

    dist = cost[-1, -1] / sum(cost.shape)
    return dist, cost

def dtwTopRight(Y, X, step_list, subsequence=False, d = euclidean):
    # create the cost matrix
    M, N = Y.shape[0], X.shape[0]
    Y, X = np.reshape(Y, (M,-1)), np.reshape(X, (N,-1))
    cost = np.zeros((N, M))

    # initialize the last column
    cost[N-1, M-1] = d(X[N-1,:], Y[M-1,:])
    for i in range(N-2, -1, -1):
        cost[i, M-1] = cost[i+1, M-1] + d(X[i,:], Y[M-1,:])

    # initialize the last row
    for j in range(M-2, -1, -1):
        cost[N-1, j] = (0 if subsequence else cost[N-1, j+1]) + d(X[N-1,:], Y[j,:])

    # fill in the rest of the matrix
    for i in range(N-2, -1, -1):
        for j in range(M-2, -1, -1):
            cost_list = float('inf') * np.ones(len(step_list))
            for index, (step_i, step_j) in enumerate(step_list) :
                if (i+step_i >= 0 and i+step_i < N) and (j+step_j >= 0  and j+step_j < M):
                    cost_list[index] = cost[i+step_i, j+step_j]
            cost[i, j] = np.min(cost_list) + d(X[i,:], Y[j,:])

    dist = cost[0, 0] / sum(cost.shape)
    return dist, cost

def FindPath(cost, i, j, step_list):
    max_i, max_j = map(max, zip(*step_list))
    a, b = [i], [j]
    while (i-max_i > 0 or j-max_j > 0):
        cost_list = float('inf') * np.ones(len(step_list))
        for index, (step_i, step_j) in enumerate(step_list) :
            if (i-step_i >= 0) and (j-step_j >= 0):
                cost_list[index] = cost[i-step_i, j-step_j]

        index = np.argmin( cost_list )
        i = i - step_list[index][0]
        j = j - step_list[index][1]
        
        if (i >= 0) and (j >= 0):
            a.insert(0, i)
            b.insert(0, j)

    return np.asarray((a, b)).T

def FindPathRightAligned(cost, i, j, step_list):
    max_i, max_j = map(max, zip(*step_list))
    rows, cols = np.array(cost.shape)
    a, b = [i], [j]

    while (i < rows - max_i or j < cols - max_j):
        cost_list = float('inf') * np.ones(len(step_list))
        for index, (step_i, step_j) in enumerate(step_list) :
            if (i+step_i >= 0) and (i+step_i < rows) and (j+step_j >= 0) and (j+step_j < cols):
                cost_list[index] = cost[i+step_i, j+step_j]

        index = np.argmin( cost_list )
        i = i + step_list[index][0]
        j = j + step_list[index][1]
        
        if (i < rows) and (j < cols):
            a.insert(0, i)
            b.insert(0, j)

    return np.asarray((a, b)).T

def FindLowerBound(query_path):
    k = -1
    while ( (k+1 < len(query_path)) and (query_path[k+1] == query_path[0]) ):
        k = k + 1
    return k

def dtw_plot_matches(Y, X, path=[]):
    plt.figure(figsize=(12,4))
    plt.ylim(ymax = 15, ymin = 0)
    plt.plot(Y, len(Y) * [1], label='Train')
    plt.plot(X, len(X) * [10], label='Test')
    for (x, y) in path:
        plt.plot([Y[y], X[x]], [1, 10], c='r')
    plt.legend()
    plt.show()

def dtw_plot_matrix(cost, path, distance):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.title('DTW distance between Y and X is %.2f' % distance)
    plt.imshow(cost, origin='lower', cmap=cm.gray, interpolation='nearest')
    plt.plot(path[:,1], path[:,0], 'w')
    ax.set_xlim((-0.5, cost.shape[1]-0.5))
    ax.set_ylim((-0.5, cost.shape[0]-0.5))
    plt.show()

def testSequence():
    x = np.arange(10, 100, 1)
    y = np.arange(10, 100, 1)
    
    distance, cost = dtw(y, x, step_size=0)
    
    rows, cols = np.array(cost.shape)
    
    path = FindPath(cost, rows-1, cols-1, step_size=0)
    
    for (test, train) in path:
        print x[test], y[train] 
    
    dtw_plot_matches(y, x, path)
    
    dtw_plot_matrix(cost, path, distance)

def testSubequence(y, x, step_list, show_graph=False):
    
    distance, cost = dtw(y, x, step_list, subsequence=True)
    
    rows, cols = np.array(cost.shape)
    
    b_star = np.argmin(cost[rows-1,:])
    
    path = FindPath(cost, rows-1, b_star, step_list)
    
    a_star = FindLowerBound(path[:,0])
    
    path_size = 0
    for k in range(len(path)):
        if k >= a_star and k <= b_star:
            path[path_size, 0] = path[k, 0]
            path[path_size, 1] = path[k, 1]
            path_size = path_size + 1
    path = np.resize(path, (path_size, 2))
    
    if show_graph and path_size > 0:
        dtw_plot_matches(y, x, path)
        dtw_plot_matrix(cost, path, distance)
    
    return path

def testSubequenceRightAligned(y, x, step_list, show_graph=False):
    
    distance, cost = dtwTopRight(y, x, step_list, subsequence=True)
    
    rows, cols = np.array(cost.shape)
    
    b_star = np.argmin(cost[0,:]) 
    
    path = FindPathRightAligned(cost, 0, b_star, step_list)
    print path
    
    a_star = FindLowerBound(path[:,0]) 
    
    b_star = cols - b_star - 1 #upper bound
    
    path_size = 0
    for k in range(len(path)):
        if k >= a_star and k <= b_star:
            path[path_size, 0] = path[k, 0]
            path[path_size, 1] = path[k, 1]
            path_size = path_size + 1
    path = np.resize(path, (path_size, 2))
    
    if show_graph and path_size > 0:
        dtw_plot_matches(y, x, path)
        dtw_plot_matrix(cost, path, distance)
    
    return np.flipud(path)

def stepSizeConditionOld(step):
    return [(step,1), (1,step), (1,1)]

def stepSizeCondition(step):
    return [(0,step), (1,0), (1,step)]

if __name__ == '__main__':
    #o exemplo abaixo simula os experimentos cujo espacamento de treino eh 1m e os de testes variam
    train = np.arange(110, 150, 1)
    test = np.arange(120, 150, 2)
    '''
    #o teste abaixo simula os experimentos com mesmo espacamento, podendo faltar frames no conjunto de treino que existam no de teste
    train = np.arange(110, 200, 1)
    train = np.delete(train, [35])
    test = np.arange(120, 190, 1)
    #o exemplo abaixo simula os experimentos cujo espacamento de treino eh 5m e os de teste eh 1m
    train = np.arange(110, 200, 5)
    test = np.arange(120, 190, 1)
    '''
    batch_size = 5
    train_size = train.shape[0]
    test_size = test.shape[0]
    pred_label = -1 * np.ones((test_size,), dtype=int)
    for i in xrange((batch_size-1), test_size):
        test_data = test[i-(batch_size-1):i+1]
        #path = testSubequence(train, test_data, step_list=stepSizeCondition(2), show_graph=True)
        path = testSubequenceRightAligned(train, test_data, step_list=stepSizeCondition(2), show_graph=True)
        print train[ path[:,1] ], test_data
