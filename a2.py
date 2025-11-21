import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

FFT_CUTOFF = 32

def compute_X_k(x: np.ndarray, k: int) -> complex:
    N = len(x)
    sum_val = 0
    for n in range(N):
        angle = 2j * np.pi * k * n / N
        sum_val += x[n] * np.exp(-angle)
    return sum_val

def dft_one_dimension(x: np.ndarray) -> np.ndarray:
    N = len(x)
    X = np.ndarray(shape=(N,), dtype=complex)

    for k in range(N):
        X[k] = compute_X_k(x, k)
        
    return X

def compute_X_k_fft(x: np.ndarray, k: int) -> complex:
    N = len(x)
    if N <= FFT_CUTOFF:
        return compute_X_k(x, k)
    even = compute_X_k(x[0::2], k)
    odd = compute_X_k(x[1::2], k)

    angle = 2j * np.pi * k / N
    return even + np.exp(-angle) * odd


def fft_one_dimension(x: np.ndarray) -> np.ndarray:
    N = len(x)
    X = np.ndarray(shape=(N,), dtype=complex)
    for k in range(N):
        X[k] = compute_X_k_fft(x, k)

    return X

def compute_x_k_inverse(X: np.ndarray, k: int) -> complex:
    N = len(X)
    sum_val = 0
    for n in range(N):
        angle = 2j * np.pi * k * n / N
        sum_val += X[n] * np.exp(angle)
    return sum_val / N

def compute_x_k_inverse_fft(X: np.ndarray, k: int) -> complex:
    N = len(X)
    if N <= FFT_CUTOFF:
        return compute_x_k_inverse(X, k)
    even = compute_x_k_inverse_fft(X[0::2], k)
    odd = compute_x_k_inverse_fft(X[1::2], k)

    angle = 2j * np.pi * k / N
    return (even + np.exp(angle) * odd) / 2

def inverse_fft_one_dimension(X: np.ndarray) -> np.ndarray:
    N = len(X)
    x = np.ndarray(shape=(N,), dtype=complex)
    for k in range(N):
        x[k] = compute_x_k_inverse_fft(X, k)
    return x

def dft_two_dimensions(image: np.ndarray) -> np.ndarray:
    M, N = image.shape
    dft_temp = np.zeros((M, N), dtype=complex)
    dft_result = np.zeros((M, N), dtype=complex)

    for i in range(M):
        dft_temp[i, :] = dft_one_dimension(image[i, :])

    for j in range(N):
        dft_result[:, j] = dft_one_dimension(dft_temp[:, j])

    return dft_result

def fft_two_dimensions(image: np.ndarray) -> np.ndarray:
    M, N = image.shape
    fft_temp = np.zeros((M, N), dtype=complex)
    fft_result = np.zeros((M, N), dtype=complex)

    for i in range(M):
        print(f'\rFFT Row {i+1}/{M}', end='', flush=True)
        fft_temp[i, :] = fft_one_dimension(image[i, :])

    for j in range(N):
        print(f'\rFFT Row {j+1}/{N}', end='', flush=True)
        fft_result[:, j] = fft_one_dimension(fft_temp[:, j])

    return fft_result

def inverse_fft_two_dimensions(X: np.ndarray) -> np.ndarray:
    M, N = X.shape
    ifft_temp = np.zeros((M, N), dtype=complex)
    ifft_result = np.zeros((M, N), dtype=complex)

    for i in range(M):
        print(f'\rIFFT Row {i+1}/{M}', end='', flush=True)
        ifft_temp[i, :] = inverse_fft_one_dimension(X[i, :])

    for j in range(N):
        print(f'\rIFFT Row {j+1}/{N}', end='', flush=True)
        ifft_result[:, j] = inverse_fft_one_dimension(ifft_temp[:, j])

    return ifft_result

def log_plot_spectrum(spectrum: np.ndarray, title: str) -> None:
    magnitude_spectrum = np.log(np.abs(spectrum) + 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def test_time_complexity():
    sizes = [2**i for i in range(1, 9)]
    dft_times = []
    fft_times = []

    for size in sizes:
        x = np.random.rand(size).astype(np.float32)

        time0 = time.time()
        dft_one_dimension(x)
        time1 = time.time()
        dft_times.append(time1 - time0)

        time0 = time.time()
        fft_one_dimension(x)
        time1 = time.time()
        fft_times.append(time1 - time0)

    plt.plot(sizes, dft_times, label='DFT Time', marker='o')
    plt.plot(sizes, fft_times, label='FFT Time', marker='o')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Input Size (N)')
    plt.ylabel('Time (seconds)')
    plt.title('DFT vs FFT Time Complexity')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_time_complexity()
    exit()

    image = cv2.imread('moonlanding.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image file 'moonlanding.png' not found.")
    
    image = image.astype(np.float32) / 255.0

    print("Computing FFT...")

    time0 = time.time()
    fft_result = fft_two_dimensions(image)
    time1 = time.time()
    print(f"\nFFT computation time: {time1 - time0}")

    log_plot_spectrum(fft_result, 'FFT Spectrum')

    ifft_result = inverse_fft_two_dimensions(fft_result)
    ifft_image = np.clip(np.real(ifft_result), 0, 1)

    plt.imshow(ifft_image, cmap='gray')
    plt.title('Reconstructed Image from IFFT')
    plt.axis('off')
    plt.show()