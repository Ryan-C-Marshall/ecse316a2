import numpy as np
import matplotlib.pyplot as plt
import cv2

FFT_CUTOFF = 32

def compute_X_k(x: list, k: int) -> complex:
    N = len(x)
    sum_val = 0
    for n in range(N):
        angle = 2j * np.pi * k * n / N
        sum_val += x[n] * np.exp(-angle)
    return sum_val

def dft_one_dimension(x: list) -> list:
    N = len(x)
    X = []
    for k in range(N):
        X.append(compute_X_k(x, k))
    return X

def compute_X_k_fft(x: list, k: int) -> complex:
    N = len(x)
    if N <= FFT_CUTOFF:
        return compute_X_k(x, k)
    even = compute_X_k(x[0::2], k)
    odd = compute_X_k(x[1::2], k)

    angle = 2j * np.pi * k / N
    return even + np.exp(-angle) * odd


def fft_one_dimension(x: list) -> list:
    N = len(x)
    X = []
    for k in range(N):
        X.append(compute_X_k_fft(x, k))

    return X

def compute_x_k_inverse(X: list, k: int) -> complex:
    N = len(X)
    sum_val = 0
    for n in range(N):
        angle = 2j * np.pi * k * n / N
        sum_val += X[k] * np.exp(angle)
    return sum_val / N

def compute_x_k_inverse_fft(X: list, k: int) -> complex:
    N = len(X)
    if N <= FFT_CUTOFF:
        return compute_x_k_inverse(X, k)
    even = compute_x_k_inverse_fft(X[0::2], k)
    odd = compute_x_k_inverse_fft(X[1::2], k)

    angle = 2j * np.pi * k / N
    return (even + np.exp(angle) * odd) / 2

def inverse_fft_one_dimension(X: list) -> list:
    N = len(X)
    x = []
    for k in range(N):
        x.append(compute_x_k_inverse_fft(X, k))
    return x
