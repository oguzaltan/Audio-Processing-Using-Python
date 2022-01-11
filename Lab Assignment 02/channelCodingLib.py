import numpy as np


# Exercise 1
def cyclgenmat(gp, m, sys=False):
    n = m-len(gp)+1
    G = np.zeros((n, m))

    for i in range(n):
        G[i, i:i+m-n+1] = gp
    G = np.array(G, np.uint8)
    G_sys = genmatsys(G)
    G_par = G_sys[:, n:]
    H = np.array(np.hstack((np.transpose(G_par), np.identity(m - n))), np.uint8)
    if sys:
        return G_sys, H
    return G, H


# Exercise 2
def genmatsys(G):
    n, m = np.shape(G)
    G_sys = G.copy()
    for i in range(n):
        for j in range(m-i-1):
            if G_sys[i, i+j+1] == 1 and i+j+1 < n:
                G_sys[i, i+j+1:] += G[i+j+1, i+j+1:]
                G_sys[i] = G_sys[i] % 2
    return G_sys


# Exercise 4
def encoder(G, x):
    y = np.matmul(x, G) % 2
    return y


def de2bi(decimal, n=None):
    """Convert decimal numbers to binary vectors

        Arguments:
            decimal -- integer or iterable of integers
            [n] -- optional, maximum length of representation

        Returns:
            r x n ndarray where each row is a binary vector

        Example:
            de2bi(5)
            -> array([1, 0, 1])
            de2bi(range(6))
            array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 1]])
    """

    if type(decimal) is int:
        decimal = (decimal,)

    if n is None:
        # calculate max length of binary representation
        n = int(np.ceil(np.log2(np.max(decimal)+1)))

    # create output matrix
    x = np.zeros(shape=(len(decimal), n), dtype=int)
    for i in range(len(decimal)):
        b = bin(decimal[i])[2:]
        x[i, (n-len(b)):] = np.array(list(b))

    return x


# Exercise 5
def codefeatures(G):
    n, m = np.shape(G)
    all_words = de2bi(range(2**n))

    G_sys_code = genmatsys(G)
    par_mat_code = G_sys_code[:, n:]
    H = np.hstack((np.transpose(par_mat_code), np.identity(m - n)))
    y_all_words = encoder(G, all_words)

    code_words = []
    hamming_dis = 1000
    for i in range(2**n-1):
        if np.sum(np.matmul(y_all_words[i+1], np.transpose(H)) % 2) == 0:
            code_words.append(y_all_words[i+1])
            if np.sum(y_all_words[i+1]) < hamming_dis:
                hamming_dis = np.sum(y_all_words[i+1])

    correct_bit_num = (hamming_dis-1)/2

    return np.array(code_words), hamming_dis, correct_bit_num


# Exercise 6
def syndict(H, e):
    err_vec = []
    all_possibilities = de2bi(range(2**np.shape(H)[1]))
    for i in range(len(all_possibilities[:, 0])):
        if np.sum(all_possibilities[i]) <= e:
            err_vec.append(all_possibilities[i])
    err_vec = np.array(err_vec)
    # synd = np.matmul(err_vec, H.T) % 2
    # test = {str(synd[0]): err_vec[0]}
    # print(test)
    synd_table = {tuple(np.matmul(error, H.T) % 2): error for error in err_vec}
    return synd_table


# Exercise 7 and 8
def decoder(G, H, Y):
    e = codefeatures(G)[2]
    syn_table = syndict(H, e)
    # same dimensions
    if np.ndim(Y) == 1:
        Y = Y[np.newaxis, :]
    for i in range(len(Y)):
        # Correct the Errors
        syn = tuple(np.matmul(Y[i], H.T) % 2)
        error = syn_table[syn]
        Y[i] = (Y[i] + error) % 2
        # Calculate the Codeword
    X = Y[:, :np.shape(G)[0]]
    return X