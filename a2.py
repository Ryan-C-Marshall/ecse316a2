import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

FFT_CUTOFF = 8

class fourierTransformResult2d:
    def __init__(self, X: np.ndarray, pad_x: int, pad_y: int):
        self.X = X
        self.pad_x = pad_x
        self.pad_y = pad_y

def compute_X_k(x: np.ndarray, k: int) -> complex:
    N = len(x)
    sum_val = 0
    for n in range(N):
        angle = 2j * np.pi * k * n / N
        sum_val += x[n] * np.exp(-angle)
    return sum_val

def dft_one_dimension(x: np.ndarray) -> np.ndarray:
    N = len(x)
    X = np.zeros(shape=(N,), dtype=complex)

    for k in range(N):
        X[k] = compute_X_k(x, k)
        
    return X


def fft_one_dimension(x: np.ndarray) -> np.ndarray:
    x, pad_amount = zero_pad_to_power_of_two(x)

    N = len(x)
    if N <= FFT_CUTOFF:
        return dft_one_dimension(x)

    X = np.zeros(shape=(N,), dtype=complex)

    even = fft_one_dimension(x[0::2].copy())
    odd = fft_one_dimension(x[1::2].copy())

    for k in range(N//2):
        twiddle = np.exp(-2j*np.pi*k/N) * odd[k]
        X[k] = even[k] + twiddle
        X[k + N//2] = even[k] - twiddle

    if pad_amount > 0:
        X = X[:-pad_amount]

    return X


def compute_x_k_inverse(X: np.ndarray, k: int) -> complex:
    N = len(X)
    sum_val = 0.0
    for n in range(N):
        angle = 2j * np.pi * k * n / N
        sum_val += X[n] * np.exp(angle)
    return sum_val

def inverse_fft_one_dimension(X: np.ndarray) -> np.ndarray:
    N = len(X)

    x = np.zeros(shape=(N,), dtype=complex)

    if N <= FFT_CUTOFF:
        for k in range(N):
            x[k] = compute_x_k_inverse(X, k)
        return x
    
    even = inverse_fft_one_dimension(X[0::2].copy())
    odd = inverse_fft_one_dimension(X[1::2].copy())

    for k in range(N//2):
        twiddle = np.exp(2j * np.pi * k / N) * odd[k % (N // 2)]
        x[k] = (even[k] + twiddle) / N
        x[k + N//2] = (even[k] - twiddle) / N
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

def fft_two_dimensions(image: np.ndarray) -> fourierTransformResult2d:
    image, (x_pad, y_pad) = zero_pad_to_power_of_two_2d(image)

    M, N = image.shape
    fft_temp = np.zeros((M, N), dtype=complex)
    fft_result = np.zeros((M, N), dtype=complex)

    for i in range(M):
        print(f'\rFFT Row {i+1}/{M}', end='', flush=True)
        fft_temp[i, :] = fft_one_dimension(image[i, :])

    for j in range(N):
        print(f'\rFFT Row {j+1}/{N}', end='', flush=True)
        fft_result[:, j] = fft_one_dimension(fft_temp[:, j])

    return fourierTransformResult2d(fft_result, x_pad, y_pad)

def inverse_fft_two_dimensions(fourierTransformResult: fourierTransformResult2d) -> np.ndarray:
    X = fourierTransformResult.X
    M, N = X.shape
    ifft_temp = np.zeros((M, N), dtype=complex)
    ifft_result = np.zeros((M, N), dtype=complex)

    for i in range(M):
        print(f'\rIFFT Row {i+1}/{M}', end='', flush=True)
        ifft_temp[i, :] = inverse_fft_one_dimension(X[i, :])

    for j in range(N):
        print(f'\rIFFT Row {j+1}/{N}', end='', flush=True)
        ifft_result[:, j] = inverse_fft_one_dimension(ifft_temp[:, j])
    if fourierTransformResult.pad_x > 0:
        ifft_result = ifft_result[:-fourierTransformResult.pad_x, :]
    if fourierTransformResult.pad_y > 0:
        ifft_result = ifft_result[:, :-fourierTransformResult.pad_y]

    return ifft_result / (M * N)


def remove_high_frequencies(fourierTransformResult: fourierTransformResult2d, proportion: float) -> fourierTransformResult2d:
    X = fourierTransformResult.X
    M, N = X.shape

    m_start = int(M * (1 - proportion) / 2)
    n_start = int(N * (1 - proportion) / 2)

    for i in range(m_start, M - m_start):
        for j in range(n_start, N - n_start):
            X[i, j] = 0

    return fourierTransformResult

def log_plot_spectrum(spectrum: np.ndarray, title: str) -> None:
    magnitude_spectrum = np.log(np.abs(spectrum) + 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def zero_pad_to_power_of_two(x: np.ndarray) -> tuple[np.ndarray, int]:
    N = len(x)
    next_power_of_two = 1 << (N - 1).bit_length()
    padded_x = np.zeros(next_power_of_two, dtype=x.dtype)
    padded_x[:N] = x
    return padded_x, next_power_of_two - N

def zero_pad_to_power_of_two_2d(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    M, N = image.shape
    next_power_of_two_M = 1 << (M - 1).bit_length()
    next_power_of_two_N = 1 << (N - 1).bit_length()
    padded_image = np.zeros((next_power_of_two_M, next_power_of_two_N), dtype=image.dtype)
    padded_image[:M, :N] = image
    return padded_image, (next_power_of_two_M - M, next_power_of_two_N - N)


def test_time_complexity():
    sizes = [2**i for i in range(1, 10)]
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


def small_fft_test():
    x = np.array([0, 1, 2, 3], dtype=np.float32)
    print("Input:", x)

    built_in_fft = np.fft.fft(x)
    print("Built-in FFT Result:", built_in_fft)

    dft_result = dft_one_dimension(x)
    print("DFT Result:", dft_result)

    fft_result = fft_one_dimension(x)
    print("FFT Result:", fft_result)

    ifft_dft_result = inverse_fft_one_dimension(dft_result)
    print("Inverse DFT Result:", ifft_dft_result)

    ifft_fft_result = inverse_fft_one_dimension(fft_result)
    print("Inverse FFT Result:", ifft_fft_result)




if __name__ == "__main__":
    image = cv2.imread('moonlanding.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image file 'moonlanding.png' not found.")
    
    image = image.astype(np.float32) / 255.0

    print("Computing FFT...")
    fft_result = fft_two_dimensions(image)

    log_plot_spectrum(fft_result.X, 'FFT Spectrum')
    
    print("Removing High Frequencies...")
    filtered_fft_result = remove_high_frequencies(fft_result, proportion=0.85)

    ifft_image = np.clip(np.real(filtered_fft_result.X), 0, 1)

    plt.imshow(ifft_image, cmap='gray')
    plt.title('IFFT after High Frequency Removal')
    plt.axis('off')
    plt.show()

    print("Computing Inverse FFT...")
    ifft_result = inverse_fft_two_dimensions(filtered_fft_result)

    print("Displaying Reconstructed Image...")
    ifft_image = np.clip(np.real(ifft_result), 0, 1)

    plt.imshow(ifft_image, cmap='gray')
    plt.title('Reconstructed Image from IFFT')
    plt.axis('off')
    plt.show()
