import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import medfilt
from scipy.signal import spectrogram
import os

data_file_path = 'Broadband_Data_2016.03.08_00.00.00.bin'

def comparator(envelope, threshold):
    return np.where(envelope > threshold, 1, 0)


def butter_lowpass_filter(data, cutoff_freq, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data


# строит спектры
def plot_signal_spectrum(signal, sample_rate):
    n = len(signal)
    T = 1 / sample_rate
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(n, T)[:n//2]
    yf = 2.0/n * np.abs(yf[:n//2])

    plt.plot(xf, yf)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Signal Spectrum')
    plt.grid()
    plt.show()

# строим осциллограмм
def plot_signal(data, fs):
    """
    Функция для построения осциллограммы сигнала.

    Параметры:
        data: numpy.ndarray
            Входные данные, представляющие собой временной ряд.
        fs: float
            Частота дискретизации входных данных (в Гц).
    """
    T = len(data) / fs  # Вычисление длительности сигнала
    t = np.linspace(0, T, len(data), endpoint=False)  # Создание временной шкалы
    
    # Построение осциллограммы
    plt.figure(figsize=(10, 6))
    plt.plot(t, data, label='Отфильтрованный сигнал', color='blue')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.title('Осциллограмма отфильтрованного сигнала')
    plt.grid(True)
    plt.show()

# фильтр полосовой
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Функция для применения цифрового полосового фильтра Баттерворта.

    Параметры:
        data: numpy.ndarray
            Входные данные, представляющие собой временной ряд.
        lowcut: float
            Нижняя граница полосы пропускания (в Гц).
        highcut: float
            Верхняя граница полосы пропускания (в Гц).
        fs: float
            Частота дискретизации входных данных (в Гц).
        order: int, optional
            Порядок фильтра. По умолчанию 5.

    Возвращает:
        numpy.ndarray
            Отфильтрованный сигнал.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def butter_lowpass(normal_cutoff, order=5):
    """
    Функция для расчета коэффициентов фильтра Баттерворта.

    Параметры:
        normal_cutoff (float): Нормализованная частота среза.
        order (int, optional): Порядок фильтра (по умолчанию 5).

    Возвращает:
        b, a (arrays): Коэффициенты фильтра.
    """
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# ФНЧ
def lowpass_filter(signal, cutoff_freq, fs, order=5):
    """
    Функция для применения цифрового фильтра нижних частот к сигналу.

    Параметры:
        signal (array): Входной сигнал.
        cutoff_freq (float): Частота среза фильтра в Гц.
        fs (float): Частота дискретизации в Гц.
        order (int, optional): Порядок фильтра (по умолчанию 5).

    Возвращает:
        filtered_signal (array): Отфильтрованный сигнал.
    """
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter_lowpass(normal_cutoff, order)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


# broadband_folder = 'broadband_folder'
# combined_data = []

# data_files = [file for file in os.listdir(broadband_folder) if file.endswith('.bin')]

# for data_file in data_files:
#     data_file_path = os.path.join(broadband_folder, data_file)
#     with open(data_file_path, 'rb') as f:
#         data = np.fromfile(f, dtype=np.int16)
#         combined_data.extend(data)

# with open('combined_data.bin', 'wb') as f:
#     f.write(np.array(combined_data, dtype=np.int16))

with open('combined_data.bin', 'rb') as f:
    data = np.fromfile(f, dtype=np.int16)

data = data.astype(np.float32)

print(type(data))

data_1 = data[::2]  # выбираем элементы с четными индексами (Север-Юг)
data_2 = data[1::2]  # выбираем элементы с нечетными индексами (Запад-Восток)

fs = 100000
signal_NS = data_1
signal_WE = data_2

print(type(signal_NS))

duration_seconds_NS = len(data_1) / fs
duration_seconds_WE = len(data_2) / fs

# plot_signal_spectrum(signal_NS, fs)
# plot_signal_spectrum(signal_WE, fs)

signal2_after_filt = butter_bandpass_filter(signal_NS, 22000, 22400, fs, order=5)

# plot_signal_spectrum(signal2_after_filt, fs)

# plot_signal(signal2_after_filt, fs)


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
f, t, Sxx = spectrogram(signal_NS, fs, nfft=4096)
print(f"TIME: {len(t/60)}")
# print(f, len(f))
# f = f[:len(f) // 2] 
print(f, len(f))
plt.pcolormesh(t, f, 20 * np.log10(Sxx), shading='auto', cmap='jet')
plt.pcolormesh(t, f, 20 * np.log10(Sxx[:len(f), :]), shading='auto', cmap='jet', vmin=0, vmax=90)

plt.title('Спектрограмма')
plt.xlabel('Время (сек)')
plt.ylabel('Частота (Гц)')
plt.colorbar().set_label('Интенсивность')


plt.subplot(1, 2, 2)
f, t, Sxx = spectrogram(signal_WE, fs, nfft=4096)
print(f"TIME: {len(t/60)}")
# print(f, len(f))
# f = f[:len(f) // 2] 
print(f, len(f))
plt.pcolormesh(t, f, 20 * np.log10(Sxx), shading='auto', cmap='jet')
plt.pcolormesh(t, f, 20 * np.log10(Sxx[:len(f), :]), shading='auto', cmap='jet', vmin=0, vmax=90)

plt.title('Спектрограмма')
plt.xlabel('Время (сек)')
plt.ylabel('Частота (Гц)')
plt.colorbar().set_label('Интенсивность')
plt.show()
            
#############################################################################

def correlation_demodulation(carrier_freq, signal, sampling_rate):
    signal_length = len(signal)

    carrier_signal_1 = np.sin(2 * np.pi * carrier_freq * np.arange(signal_length) / sampling_rate)
    carrier_signal_2 = np.cos(2 * np.pi * carrier_freq * np.arange(signal_length) / sampling_rate)

    correlated_signal_1 = signal * carrier_signal_1
    correlated_signal_2 = signal * carrier_signal_2

    Q_part = butter_lowpass_filter(correlated_signal_1, 500, sampling_rate)
    I_part = butter_lowpass_filter(correlated_signal_2, 500, sampling_rate)

    # integrator_function1 = np.cumsum(Q_part) / sampling_rate
    # integrator_function2 = np.cumsum(I_part) / sampling_rate

    phase = np.arctan2(Q_part, I_part)

    integrator_function1 = np.cumsum(phase) / sampling_rate
    
    window_size = 70
    moving_average = np.convolve(integrator_function1, np.ones(window_size) / window_size, mode='same')[:signal_length]

    derivative_function = np.gradient(moving_average, 1 / sampling_rate)

    window_size = 9 # Adjust this value based on the characteristics of your signal
    derivative_function_filtered = medfilt(derivative_function, window_size)

    # replace_size = 10  # Adjust this value based on the duration of the spike
    # mean_value = np.mean(derivative_function_filtered[replace_size:])
    # derivative_function_filtered[:replace_size] = mean_value

    threshold = 0

    demodulated_signal = comparator(derivative_function_filtered, threshold)

    return demodulated_signal, derivative_function

demodul_sig, _ = correlation_demodulation(22000, signal2_after_filt, fs)

plt.figure(figsize=(10, 4))
time = np.arange(len(demodul_sig)) / fs
plt.plot(time, demodul_sig)
plt.title('Осциллограмма демодулированного сигнала')
plt.xlabel('Время (сек)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.show()

#############################################################################
time_points = np.linspace(0, len(signal_NS) / fs, len(signal_NS), endpoint=False)  # Создание временной шкалы

cosine_wave = np.cos(2 * np.pi * 22000 * time_points)
cosine_wave_05_pi = np.cos(2 * np.pi * 22000 * time_points + np.pi/2)

signal_after_mult = signal2_after_filt * cosine_wave
signal_after_mult_05_pi = signal2_after_filt * cosine_wave_05_pi

plot_signal_spectrum(signal_after_mult_05_pi, fs)

I_component_signal = lowpass_filter(signal_after_mult, 400, fs, order=5)
Q_component_signal = lowpass_filter(signal_after_mult_05_pi, 400, fs, order=5)

result_signal = np.sqrt(np.array(I_component_signal)**2 + np.array(Q_component_signal)**2)
phase_result_signal = np.arctan2(Q_component_signal, I_component_signal)

plt.figure()
# Построение первого графика (красного цвета)
plt.plot(time_points, I_component_signal, color='red', label='sin(x)')

# Построение второго графика (синего цвета)
plt.plot(time_points, Q_component_signal, color='blue', label='cos(x)')

# Добавление легенды
plt.legend()

# Добавление заголовка и подписей к осям
plt.title('Графики I и Q')
plt.xlabel('x')
plt.ylabel('y')

# Отображение графика
plt.show()

plot_signal(phase_result_signal, fs)


###############################################################################

def envelope(signal):
    return np.abs(signal)

def zero_crossings(signal):
    return np.where(np.diff(np.sign(signal)))[0]

def demodulate_msk(I, Q):
    envelope_I = envelope(I)
    envelope_Q = envelope(Q)
    
    # Выбираем, какой из сигналов огибающей использовать (можно использовать любой)
    envelope_signal = envelope_I
    
    # Детектируем переходы через нуль
    zero_crossings_indices = zero_crossings(envelope_signal)
    
    # Восстанавливаем битовую последовательность
    bits = np.diff(zero_crossings_indices) > 10  # Примерный порог для детектирования переходов
    
    return bits

bits = demodulate_msk(I_component_signal, Q_component_signal)

print(bits[: 20])
