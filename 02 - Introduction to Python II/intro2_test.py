import numpy as np
import channelCodingLib as lib

G1 = np.array([1, 1, 0, 1])
G2 = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])

x1 = np.array([1, 0, 0, 1])
x2 = np.array([0, 0, 0, 1])
x3 = np.array([1, 0, 1, 0])

x4 = np.array([0, 1, 1, 0, 0])
x5 = np.array([1, 1, 0, 0, 0])
x6 = np.array([0, 0, 1, 1, 1])

xG1 = np.stack((x1, x2, x3))
xG2 = np.stack((x4, x5, x6))


def ex1_test():
    '''Print the non-systematic generator matrices'''
    print(lib.cyclgenmat(G1, 7)[0])
    print(lib.cyclgenmat(G2, 15)[0])
    return


def ex2_test():
    '''Print the systematic generator matrices'''
    print(lib.cyclgenmat(G1, 7, sys=True)[0])
    print(lib.cyclgenmat(G2, 15, sys=True)[0])
    return


def ex3_test():
    '''Print the parity-check matrices'''
    print(lib.cyclgenmat(G1, 7, sys=True)[1])
    print(lib.cyclgenmat(G2, 15, sys=True)[1])
    return

def ex4_test():
    '''Print the code word y'''



    G_enc1 = lib.cyclgenmat(G1, 7, sys=True)[0]
    G_enc2 = lib.cyclgenmat(G2, 15, sys=True)[0]

    print("y for single information word x1\n", lib.encoder(G_enc1, x1))
    print("y for single information word x4\n", lib.encoder(G_enc2, x4))

    xG1 = np.stack((x1, x2, x3))
    xG2 = np.stack((x4, x5, x6))

    print("y for multiple information words with G1\n", lib.encoder(G_enc1, xG1))
    print("y for multiple information words with G2\n", lib.encoder(G_enc2, xG2))

    return


def ex5_test():
    G_code1 = lib.cyclgenmat(G1, 7, sys=True)[0]
    G_code2 = lib.cyclgenmat(G2, 15, sys=True)[0]

    print("Possible code words:\n", lib.codefeatures(G_code1)[0])
    print("Minimum Hamming Distance:\n", lib.codefeatures(G_code1)[1])
    print("Number of Correctable Bit Errors:\n", lib.codefeatures(G_code1)[2])

    print("Possible code words:\n", lib.codefeatures(G_code2)[0])
    print("Minimum Hamming Distance:\n", lib.codefeatures(G_code2)[1])
    print("Number of Correctable Bit Errors:\n", lib.codefeatures(G_code2)[2])

    return

def ex6_test():
    G_sys1, H1 = lib.cyclgenmat(G1, 7, sys=True)
    G_sys2, H2 = lib.cyclgenmat(G2, 15, sys=True)
    e1 = lib.codefeatures(G_sys1)[2]
    e2 = lib.codefeatures(G_sys2)[2]
    dict1 = lib.syndict(H1, e1)
    dict2 = lib.syndict(H2, e2)
    print(dict1)
    print(dict2)
    return


def ex7_test():
    G_sys1, H1 = lib.cyclgenmat(G1, 7, sys=True)
    G_sys2, H2 = lib.cyclgenmat(G2, 15, sys=True)

    y1 = lib.encoder(G_sys1, xG1)
    y2 = lib.encoder(G_sys2, xG2)
    x1_dec = lib.decoder(G_sys1, H1, y1)
    x2_dec = lib.decoder(G_sys2, H2, y2)
    print("Decoded words for G1:\n", x1_dec)
    print("Decoded words for G2:\n", x2_dec)
    return


def ex8_test():
    G_sys1, H1 = lib.cyclgenmat(G1, 7, sys=True)
    G_sys2, H2 = lib.cyclgenmat(G2, 15, sys=True)

    y1 = lib.encoder(G_sys1, xG1)
    y2 = lib.encoder(G_sys2, xG2)
    y1_e1 = y1
    y2_e1 = y2
    y1_e2 = y1
    y2_e2 = y2
    # exactly correctable errors
    y1_e1[:, 3] = (y1[:, 3] + 1) % 2
    y2_e1[:, 3] = (y2[:, 3] + 1) % 2
    y2_e1[:, 5] = (y2[:, 5] + 1) % 2
    y2_e1[:, 6] = (y2[:, 6] + 1) % 2
    # more than correctable errors
    # err_vec_1 = np.array([0, 0, 1, 1])
    # err_vec_2 = np.array([0, 0, 1, 1, 1, 1, 1])
    # y1_e2 = (y1 + err_vec_1) % 2
    # y2_e2 = (y2 + err_vec_2) % 2
    y2_e2 = (y2 + np.array([[0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])) % 2


    x1_dec = lib.decoder(G_sys1, H1, y1_e1)
    x2_dec = lib.decoder(G_sys2, H2, y2_e2)
    print("Decoded words for G1:\n", x1_dec)
    print("Original:\n", xG2)
    print("Decoded words for G2:\n", x2_dec)
    return


ex6_test()