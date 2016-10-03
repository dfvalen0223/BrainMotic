import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sched
import time

from scipy import *
from numpy.fft import fft


# def DFT_slow(x):
#     """Compute the discrete Fourier Transform of the 1D array x"""
#     x = np.asarray(x, dtype=float)
#    N = x.shape[0]
#    n = np.arange(N)
#    k = n.reshape((N, 1))
#    M = np.exp(-2j * np.pi * k * n / N)
#    return np.dot(M, x)

# FFT without normalize
# print(' '.join("%5.3f" % abs(f) for f in fft(a)))

class DataCursor(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax):
        self.ax = ax
        self.annotation = ax.annotate(self.text_template,
                                      xy=(self.x, self.y), xytext=(self.xoffset, self.yoffset),
                                      textcoords='offset points', ha='right', va='bottom',
                                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                                      )
        self.annotation.set_visible(False)

    def __call__(self, event):
        self.event = event
        # xdata, ydata = event.artist.get_data()
        # self.x, self.y = xdata[event.ind], ydata[event.ind]
        self.x, self.y = event.mouseevent.xdata, event.mouseevent.ydata
        if self.x is not None:
            self.annotation.xy = self.x, self.y
            self.annotation.set_text(self.text_template % (self.x, self.y))
            self.annotation.set_visible(True)
            event.canvas.draw()


trial = 1       # number of trial
tepoch = 1      # length o vector epoch
numCh = 3       # number of channels
x_train = 0     # x_train

Fs = 128  # Sampling frequency
T = 1 / Fs  # Sample time
L = 1*Fs  # Length of signal
t = range(1, L - 1) * T  # Time vector
NFFT = round(math.log(L, 2), 0)  # Next power 2 of length signal
f = float(Fs) / 2 * linspace(0, 1, power(2, NFFT) / 2 + 1)  # array to x ticks

matplotlib.interactive(True)
matplotlib.is_interactive()

main_while = True
# 1.a. Plot 3 ch  EEG signal
# 1.b. Obtain index, like a energy bands over a line .8
#       80% of energy in specific bands
# 2. Organize data to Tensorflow
#       extract features, like E.bands and labels

# To draw the results
fig, ax = plt.subplots()
ax.set_title('Trial: ' + str(0) + ' - Second: ' + str(0))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('|Y(f)|')
plt.ion()
line1, = ax.plot(f, [0] * len(f), marker='o')
plt.show()
plt.cla()



def update_line():
    ax.clear()
    plt.hold(True)
    if tepoch < 3:
        ax.set_axis_bgcolor('black')
    else:
        ax.set_axis_bgcolor('white')

    ax.set_title('Trial: ' + str(trial) + ' - Second: ' + str(tepoch))
    ax.set_xlabel('[Hz]')
    ax.set_ylabel('|Y(f)|')
    # ax.plot(f, ch1, linestyle='None', c='r', marker="s", label='ch1')
    # ax.plot(f, ch2, linestyle='None', c='g', marker="o", label='ch2')
    # ax.plot(f, ch3, linestyle='None', c='b', marker="d", label='ch3')
    ax.plot(f, ch1, c='r', marker="s", label='ch1')
    ax.plot(f, ch2, c='g', marker="o", label='ch2')
    ax.plot(f, ch3, c='b', marker="d", label='ch3')
    plt.legend(loc='upper right', numpoints=1)
    plt.draw()


# Function of interruption
def data_ready():
    global trial, tepoch, numCh, main_while, x_train, ch1, ch2, ch3
    x_train = pd.read_csv('dataset_BCIcomp1_x_train.csv', header=None,
                          usecols=range(((trial - 1) * numCh), trial * numCh),
                          skiprows=((tepoch - 1) * Fs - 1), nrows=Fs)
    # Just Channel 1
    ch1 = abs(fft(x_train._get_values[:, 0]))
    ch1 /= max(ch1)  # To normalize results
    ch1 = ch1[0:int(power(2, NFFT) / 2 + 1)]  # Right side FFT
    # Just Channel 2
    ch2 = abs(fft(x_train._get_values[:, 1]))
    ch2 /= max(ch2)  # To normalize results
    ch2 = ch2[0:int(power(2, NFFT) / 2 + 1)]  # Right side FFT
    # Just Channel 3
    ch3 = abs(fft(x_train._get_values[:, 2]))
    ch3 /= max(ch3)  # To normalize results
    ch3 = ch3[0:int(power(2, NFFT) / 2 + 1)]  # Right side FFT
    update_line()

    if tepoch >= 9:
        tepoch = 1
        trial += 1
        time.sleep(1)
    else:
        tepoch += 1

    if trial > 140:
        trial = 1
        main_while = False

def main():
    scheduler = sched.scheduler(time.time, time.sleep)
    while main_while:
        # Config timer interrupt - ini
        # Parameters:
        # 1. A number representing the delay, seconds
        # 2. A priority value
        # 3. The function to call
        # 4. A tuple of arguments for the function, create a copy of var
        scheduler.enter(1.0, 1, data_ready, ())
        scheduler.run()
        plt.pause(0.001)
        # Config timer interrupt - end


if __name__ == '__main__':
    main()
