import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import argparse


FFT_CUTOFF = 8
DEFAULT_DENOISING_PROPORTION = 0.85

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
    X_conj = np.conjugate(X)
    x = fft_one_dimension(X_conj)
    x = np.conjugate(x) / N
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

def fft_two_dimensions(image: np.ndarray, title="FFT") -> fourierTransformResult2d:
    image, (x_pad, y_pad) = zero_pad_to_power_of_two_2d(image)

    M, N = image.shape
    fft_temp = np.zeros((M, N), dtype=complex)
    fft_result = np.zeros((M, N), dtype=complex)

    for i in range(M):
        print(f'\r{title}: {i+1}/{M+N}', end='', flush=True)
        fft_temp[i, :] = fft_one_dimension(image[i, :])

    for j in range(N):
        print(f'\r{title}: {M+j+1}/{M+N}', end='', flush=True)
        fft_result[:, j] = fft_one_dimension(fft_temp[:, j])

    print(f"\r{title} complete.{' ' * 20}")

    return fourierTransformResult2d(fft_result, x_pad, y_pad)

def inverse_fft_two_dimensions(fourierTransformResult: fourierTransformResult2d, title="IFFT") -> np.ndarray:
    X = fourierTransformResult.X
    M, N = X.shape
    ifft_temp = np.zeros((M, N), dtype=complex)
    ifft_result = np.zeros((M, N), dtype=complex)

    for i in range(M):
        print(f'\r{title}: {i+1}/{M+N}', end='', flush=True)
        ifft_temp[i, :] = inverse_fft_one_dimension(X[i, :])

    for j in range(N):
        print(f'\r{title}: {M+j+1}/{M+N}', end='', flush=True)
        ifft_result[:, j] = inverse_fft_one_dimension(ifft_temp[:, j])

    if fourierTransformResult.pad_x > 0:
        ifft_result = ifft_result[:-fourierTransformResult.pad_x, :]
    if fourierTransformResult.pad_y > 0:
        ifft_result = ifft_result[:, :-fourierTransformResult.pad_y]

    print(f"\r{title} complete.{' ' * 20}")

    return ifft_result

import numpy as np

def remove_high_frequencies(fourierTransformResult: fourierTransformResult2d, proportion: float) -> fourierTransformResult2d:
    new_ft_result = fourierTransformResult2d(fourierTransformResult.X.copy(), fourierTransformResult.pad_x, fourierTransformResult.pad_y)
    X = new_ft_result.X
    M, N = X.shape

    # This math ensures that proportion 'proportion' of pixels are set to 0
    m = np.sqrt(proportion * np.square(M))
    n = np.sqrt(proportion * np.square(N))

    m_start = int((M - m) / 2)
    n_start = int((N - n) / 2)

    for i in range(m_start, M - m_start):
        for j in range(n_start, N - n_start):
            X[i, j] = 0

    return new_ft_result

def compress(fourierTransformResult: fourierTransformResult2d, proportion: float) -> fourierTransformResult2d:
    print("Compressing with proportion:", proportion)

    base_proportion = 0.85

    if proportion <= base_proportion:
        return remove_high_frequencies(fourierTransformResult, proportion)
    else:
        ft_result = remove_high_frequencies(fourierTransformResult, base_proportion)
        new_proportion = proportion - base_proportion
        X = ft_result.X
        M, N = X.shape

        # Number of elements to remove (proportion of total elements)
        total_elements = X.size
        num_to_remove = int(new_proportion * total_elements)

        if num_to_remove <= 0:
            return ft_result

        # Compute geometric mask of locations removed by the initial
        # high-frequency removal (use the same base_proportion geometry).
        m = int(np.sqrt(base_proportion * np.square(M)))
        n = int(np.sqrt(base_proportion * np.square(N)))
        m_start = int((M - m) / 2)
        n_start = int((N - n) / 2)

        mask = np.ones((M, N), dtype=bool)
        mask[m_start:M - m_start, n_start:N - n_start] = False

        # Candidate indices are those not masked out (i.e. not removed geometrically)
        abs_vals = np.abs(X).ravel()
        candidate_indices = np.flatnonzero(mask.ravel())
        if candidate_indices.size == 0:
            return ft_result

        # Number to remove should not exceed available candidates
        num_to_remove = min(num_to_remove, candidate_indices.size)

        # Sort candidates by magnitude and pick the smallest
        candidate_abs = abs_vals[candidate_indices]
        sorted_order = np.argsort(candidate_abs)
        indices_to_zero = candidate_indices[sorted_order[:num_to_remove]]

        # Zero the selected coefficients
        flat_X = X.ravel()
        flat_X[indices_to_zero] = 0

        return ft_result


def remove_low_frequencies(fourierTransformResult: fourierTransformResult2d, proportion: float) -> fourierTransformResult2d:
    new_ft_result = fourierTransformResult2d(fourierTransformResult.X.copy(), fourierTransformResult.pad_x, fourierTransformResult.pad_y)
    X = new_ft_result.X
    M, N = X.shape

    # This math ensures that proportion 'proportion' of pixels are kept
    m = np.sqrt(proportion * np.square(M))
    n = np.sqrt(proportion * np.square(N))

    m_start = int((M - m) / 2)
    n_start = int((N - n) / 2)

    for i in range(0, m_start):
        for j in range(0, N):
            X[i, j] = 0
    for i in range(M - m_start, M):
        for j in range(0, N):
            X[i, j] = 0
    for i in range(m_start, M - m_start):
        for j in range(0, n_start):
            X[i, j] = 0
    for i in range(m_start, M - m_start):
        for j in range(N - n_start, N):
            X[i, j] = 0

    return new_ft_result

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
    sizes = [2**i for i in range(5, 9)]

    dft_means, fft_means = [], []
    dft_stddevs, fft_stddevs = [], []

    for N in sizes:
        dft_times, fft_times = [], []

        for _ in range(10):
            img = np.random.rand(N, N).astype(np.float32)

            t0 = time.time()
            dft_two_dimensions(img)
            t1 = time.time()
            dft_times.append(t1 - t0)

            t0 = time.time()
            fft_two_dimensions(img)
            t1 = time.time()
            fft_times.append(t1 - t0)

        dft_means.append(np.mean(dft_times))
        fft_means.append(np.mean(fft_times))
        dft_stddevs.append(np.std(dft_times))
        fft_stddevs.append(np.std(fft_times))

        print(
            f"N={N}: "
            f"DFT mean={dft_means[-1]:.6f}s, FFT mean={fft_means[-1]:.6f}s | "
            f"DFT variance={dft_stddevs[-1]**2:.6f}s², FFT variance={fft_stddevs[-1]**2:.6f}s²"
        )

    print("\n===== Runtime Summary =====")
    print(f"{'Size(N)':>8} | {'DFT mean':>10} | {'FFT mean':>10} | {'DFT var':>10} | {'FFT var':>10}")
    print("-" * 60)
    for i, N in enumerate(sizes):
        print(f"{N:8} | {dft_means[i]:10.6f} | {fft_means[i]:10.6f} | {dft_stddevs[i]**2:10.6f} | {fft_stddevs[i]**2:10.6f}")
    print("=" * 60)

    sizes = np.array(sizes)
    plt.errorbar(sizes, dft_means, yerr=2*np.array(dft_stddevs), label="2D DFT", marker='o')
    plt.errorbar(sizes, fft_means, yerr=2*np.array(fft_stddevs), label="2D FFT", marker='o')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel("Side length (N)")
    plt.ylabel("Time (seconds)")
    plt.title("2D DFT vs 2D FFT – Time Complexity")
    plt.legend()
    plt.grid(True)
    plt.show()

def take_command_line_arguments():
    parser = argparse.ArgumentParser(description='FFT and IFFT on images.')

    # Calling syntax: python fft.py [-m mode] [-i image]
    parser.add_argument('-m', '--mode',
                        type=int,
                        choices=[1, 2, 3, 4],
                        default=1,
                        help='Mode to run (integer 1-4).')

    parser.add_argument('-i', '--image',
                        type=str,
                        default='moonlanding.png',
                        help='Path to input image file (grayscale).')

    args = parser.parse_args()

    return args

def log_plot_spectrum(spectrum: np.ndarray) -> np.ndarray:
    magnitude_spectrum = np.log(np.abs(spectrum) + 1)
    return magnitude_spectrum

def display_images(images: list[np.ndarray], titles: list[str], plot_size: tuple[int, int]) -> None:
    rows, cols = plot_size
    plt.figure(figsize=(3 * cols, 3 * rows))

    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        img = images[i]

        im = plt.imshow(img, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

        if "fft spectrum" in titles[i].lower():
            plt.colorbar(im, label='Log Magnitude')

    plt.tight_layout()
    plt.show()
    

def main():
    # take args
    args = take_command_line_arguments()
    image_path = args.image if args.image else 'moonlanding.png'
    mode = args.mode

    # get image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
    if mode == 1:
        # Convert image to FFT and display spectrum
        fft_result = fft_two_dimensions(image)

        # bulit_in_fft_result = np.fft.fft2(image)

        display_images([image, log_plot_spectrum(fft_result.X)], ["Original Image", "FFT Spectrum"], (1, 2))
        

    elif mode == 2:
        # Denoise image
        fft_result = fft_two_dimensions(image)
        filtered_fft_result = remove_high_frequencies(fft_result, proportion=DEFAULT_DENOISING_PROPORTION)

        # for the experiment
        # filtered_fft_result = remove_low_frequencies(fft_result, proportion=DEFAULT_DENOISING_PROPORTION)

        ifft_result = inverse_fft_two_dimensions(filtered_fft_result)

        total_coeffs = filtered_fft_result.X.size
        nonzero_coeffs = np.count_nonzero(filtered_fft_result.X)
        fraction_nonzero = nonzero_coeffs / total_coeffs

        print("Denoising statistics:")
        print(f"Non-zero frequency components: {nonzero_coeffs} / {total_coeffs} "
              f"({fraction_nonzero:.4f} of original spectrum)")
        print(f"Zeroed components: {total_coeffs - nonzero_coeffs} "
              f"({1 - fraction_nonzero:.4f} of original spectrum)")

        display_images([image, np.real(ifft_result)], ['Original Image', 'Denoised Image'], (1, 2))


    elif mode == 3:
        # Compress image
        fft_result = fft_two_dimensions(image)

        proportions = [0.75, 0.90, 0.95, 0.99, 0.999]

        filtered_fft_result_1 = compress(fft_result, proportion=proportions[0])
        filtered_fft_result_2 = compress(fft_result, proportion=proportions[1])
        filtered_fft_result_3 = compress(fft_result, proportion=proportions[2])
        filtered_fft_result_4 = compress(fft_result, proportion=proportions[3])
        filtered_fft_result_5 = compress(fft_result, proportion=proportions[4])

        ifft_result_0 = inverse_fft_two_dimensions(fft_result, title="IFFT Original")
        ifft_result_1 = inverse_fft_two_dimensions(filtered_fft_result_1, title=f"IFFT {proportions[0]*100}%")
        ifft_result_2 = inverse_fft_two_dimensions(filtered_fft_result_2, title=f"IFFT {proportions[1]*100}%")
        ifft_result_3 = inverse_fft_two_dimensions(filtered_fft_result_3, title=f"IFFT {proportions[2]*100}%")
        ifft_result_4 = inverse_fft_two_dimensions(filtered_fft_result_4, title=f"IFFT {proportions[3]*100}%")
        ifft_result_5 = inverse_fft_two_dimensions(filtered_fft_result_5, title=f"IFFT {proportions[4]*100}%")

        spectra = [
            ("Original (0% zeroed)", fft_result),
            (f"{proportions[0]*100:.1f}% zeroed", filtered_fft_result_1),
            (f"{proportions[1]*100:.1f}% zeroed", filtered_fft_result_2),
            (f"{proportions[2]*100:.1f}% zeroed", filtered_fft_result_3),
            (f"{proportions[3]*100:.1f}% zeroed", filtered_fft_result_4),
            (f"{proportions[4]*100:.1f}% zeroed", filtered_fft_result_5),
        ]

        print("Compression statistics:")
        for label, ft in spectra:
            total = ft.X.size
            nonzero = np.count_nonzero(ft.X)
            frac = nonzero / total
            print(f"{label}: non-zero = {nonzero} / {total} ({frac:.4f}), "
                  f"zeroed = {total - nonzero} ({1 - frac:.4f})")

        display_images([np.real(ifft_result_0), np.real(ifft_result_1), np.real(ifft_result_2), np.real(ifft_result_3), np.real(ifft_result_4), np.real(ifft_result_5)],
                           [f'Original', f'{proportions[0]*100}% compressed', f'{proportions[1]*100}%', f'{proportions[2]*100}%', f'{proportions[3]*100}%', f'{proportions[4]*100}%'],
                           (2, 3))

    
    elif mode == 4:
        # Plot runtime graphs for DFT and FFT
        test_time_complexity()


def experiment_1():
    image = cv2.imread('moonlanding.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image file 'moonlanding.png' not found.")
    
    fft_result = fft_two_dimensions(image)
    display_images([image, log_plot_spectrum(fft_result.X)], ["Original Image", "FFT Spectrum"], (1, 2))

def experiment_2():
    image = cv2.imread('moonlanding.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image file 'moonlanding.png' not found.")
    
    fft_result = fft_two_dimensions(image)
    fft_result_some_low_frequencies_removed = remove_low_frequencies(fft_result, proportion=0.5)
    fft_result_most_low_frequencies_removed = remove_low_frequencies(fft_result, proportion=0.85)
    fft_result_some_high_frequencies_removed = remove_high_frequencies(fft_result, proportion=0.5)
    fft_result_most_high_frequencies_removed = remove_high_frequencies(fft_result, proportion=0.85)

    ifft_result = inverse_fft_two_dimensions(fft_result, title="IFFT Original")
    ifft_result_some_low_frequencies_removed = inverse_fft_two_dimensions(fft_result_some_low_frequencies_removed, title="IFFT Some Low Frequencies Removed")
    ifft_result_most_low_frequencies_removed = inverse_fft_two_dimensions(fft_result_most_low_frequencies_removed, title="IFFT Most Low Frequencies Removed")
    ifft_result_some_high_frequencies_removed = inverse_fft_two_dimensions(fft_result_some_high_frequencies_removed, title="IFFT Some High Frequencies Removed")
    ifft_result_most_high_frequencies_removed = inverse_fft_two_dimensions(fft_result_most_high_frequencies_removed, title="IFFT Most High Frequencies Removed")

    display_images([image, np.real(ifft_result), np.real(ifft_result_some_low_frequencies_removed), np.real(ifft_result_most_low_frequencies_removed), np.real(ifft_result_some_high_frequencies_removed), np.real(ifft_result_most_high_frequencies_removed)], ['Original Image', 'IFFT No Changes', 'Some Low Frequencies Removed', 'Most Low Frequencies Removed', 'Some High Frequencies Removed', 'Most High Frequencies Removed'], (2, 3))

# Experiment 3 is the same as mode 3 in main()

# Experiment 4 is the same as mode 4 in main()

if __name__ == "__main__":
    # experiment_2()
    
    main()
