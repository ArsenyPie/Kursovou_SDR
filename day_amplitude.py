import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import medfilt
from scipy.signal import spectrogram
import os

broadband_folder = 'broadband_folder'
combined_data = []

data_files = [file for file in os.listdir(broadband_folder) if file.endswith('.bin')]

# count = 0
# for data_file in data_files:
#     data_file_path = os.path.join(broadband_folder, data_file)
#     with open(data_file_path, 'rb') as f:
#         data = np.fromfile(f, dtype=np.int16)
#         combined_data.extend(data)

# with open('combined_data_1.bin', 'wb') as f:
#     f.write(np.array(combined_data, dtype=np.int16))

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

def main():

    with open('combined_data_1.bin', 'rb') as f:
        data = np.fromfile(f, dtype=np.int16)

    print(len(data))

    data_1 = data[::2]  # выбираем элементы с четными индексами (Север-Юг)
    data_2 = data[1::2]  # выбираем элементы с нечетными индексами (Запад-Восток)

    fs = 100000
    signal_NS = data_1

    print(type(signal_NS))

    duration_seconds_NS = len(data_1) / fs

    signal2_after_filt = butter_bandpass_filter(signal_NS, 22000, 22400, fs, order=5)
    
    day_data = np.array([])
    temp_data = np.array([])
    count = 0
    for i in data:
        if count < 500309:
            temp_data = np.append(temp_data, i)
        
        elif count == 500309:
            count = 0
            temp_data = sum(temp_data)/500309
            day_data = np.append(day_data, temp_data)

    print(len(day_data))
    print(day_data)            
    
main()

