import serial, time
import numpy as np
import urllib
import time
#import cv2

from scipy import *
from numpy.fft import fft
from espeak import espeak
espeak.set_voice("en")

td = 4500.0
trial = 1       # number of trial
tepoch = 1      # length o vector epoch
numCh = 0       # number of channels
x_train = 0     # x_train

Fs = 250  # Sampling frequency
T = 1 / Fs  # Sample time
L = 1*Fs  # Length of signal
t = range(1, L - 1) * T  # Time vector
NFFT = round(math.log(L, 2), 0)  # Next power 2 of length signal
f = float(Fs) / 2 * linspace(0, 1, power(2, NFFT) / 2 + 1)  # array to x ticks

et = 5# end time
st = 1# start time


lenData = Fs * et

hemiI = [0] * lenData
hemiD = [0] * lenData

ser = serial.Serial(#45,
                    '/dev/ttyACM0',
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=1,
                    xonxoff=0,
                    rtscts=0,
                    dsrdtr=True
                    )

BCI = 0

isRunning = False

countComma = 0
temp = [0] * lenData
countData = 0

counter = [0] * lenData
i0 = 0
temp1 = [0] * lenData
i1 = 0
temp2 = [0] * lenData
i2 = 0

temp3 = [0] * lenData
i3 = 0
temp4 = [0] * lenData
i4 = 0

temp5 = [0] * lenData
i5 = 0
temp6 = [0] * lenData
i6 = 0

temp7 = [0] * lenData
i7 = 0
temp8 = [0] * lenData
i8 = 0

dataRD = False
ser.writelines("s")
print len(temp)
while True:

    if isRunning:

        countData += 1  # cuenta caracteres del numero
        temp[countData] = ser.read(1) # read one, blocking

        # Si encuentra una coma reinicia el contador de caracteres y guarda en un canal
        # counter, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8
        if temp == ',':
            countComma += 1
            countData = 0

            if countComma == 1:
                if i0 < 250:
                    i0 += 1
                    counter[i0] = int(temp)
                else:
                    i0 = 0
            elif countComma == 2:
                if i1 < 250:
                    i1 += 1
                    temp1[i1] = int(temp)
                else:
                    i1 = 0
            elif countComma == 3:
                if i2 < 250:
                    i2 += 1
                    temp2[i2] = int(temp)
                else:
                    i2 = 0
            elif countComma == 4:
                if i3 < 250:
                    i3 += 1
                    temp3[i3] = int(temp)
                else:
                    i3 = 0
            elif countComma == 5:
                if i4 < 250:
                    i4 += 1
                    temp4[i4] = int(temp)
                else:
                    i4 = 0
            elif countComma == 6:
                if i5 < 250:
                    i5 += 1
                    temp2[i5] = int(temp)
                else:
                    i5 = 0
            elif countComma == 7:
                if i6 < 250:
                    i6 += 1
                    temp6[i6] = int(temp)
                else:
                    i6 = 0
            elif countComma == 8:
                if i7 < 250:
                    i7 += 1
                    temp7[i7] = int(temp)
                else:
                    i7 = 0
            elif countComma == 9:
                if i8 < 250:
                    i8 += 1
                    temp8[i8] = int(temp)
                else:
                    i8 = 0
                countComma = 0
                dataRD = True
                
        '''
        temp = temp.split(',')
        '''
        if dataRD:
            '''
            hemiD = temp1 + temp3 + temp5 + temp7
            hemiI = temp2 + temp4 + temp6 + temp8
            '''
            hemiD = temp1 + temp3
            hemiI = temp2 + temp4

            # Just Channel D, I
            # hemiD = abs(fft(temp2))
            # hemiI = abs(fft(temp3))
            # hemiD /= max(hemiD)  # To normalize results
            # hemiI /= max(hemiI)  # To normalize results
            # hemiD = hemiD[0:int(power(2, NFFT) / 2 + 1)]  # Right side FFT
            # hemiI = hemiI[0:int(power(2, NFFT) / 2 + 1)]  # Right side FFT

            # hemiD = hemiD[8:30]  # Right side FFT 8-30 Hz
            # hemiI = hemiI[8:30]  # left side FFT 8-30 Hz

            BCI = np.mean(hemiD) - np.mean(hemiI)

            dataRD = False

            if -td < BCI < td:
                print "pause"
                #f = urllib.urlopen("http://192.168.0.101:3000/update?key=J1NBDXM8UJ3FGTVJ&field1=10")
                #0 0 : pause
                #GPIO.output(17, False)  ## Enciendo el 17
                #GPIO.output(27, False)  ## Apago el 27

            if BCI < -td:
                print "===>"
                # Linux version
                espeak.synth("rigth")
                #windows version engine.say("right")
                #windows version engine.runAndWait()
                #1 0 : derecha ==> enter
                #GPIO.output(17, False)  ## Enciendo el 17
                #GPIO.output(27, True)  ## Apago el 27
                #f = urllib.urlopen("http://192.168.0.101:3000/update?key=J1NBDXM8UJ3FGTVJ&field1=11")

            if BCI > td:
                print "<==="
                # Linux version
                espeak.synth("left")
                #windows version engine.say("left")
                #windows version engine.runAndWait()
                #0 1 : izquierda <==
                #GPIO.output(17, True)  ## Enciendo el 17
                #GPIO.output(27, False)  ## Apago el 27
                #f = urllib.urlopen("http://192.168.0.101:3000/update?key=J1NBDXM8UJ3FGTVJ&field1=12")


            temp1 = [0] * lenData
            temp2 = [0] * lenData
            temp3 = [0] * lenData
            temp4 = [0] * lenData
            temp5 = [0] * lenData
            temp6 = [0] * lenData
            temp7 = [0] * lenData
            temp8 = [0] * lenData
                    
            hemiI = [0] * lenData
            hemiD = [0] * lenData
    
            countData = 0

    else:
        print "Welcome to Brainmotic"
        while ser.inWaiting() == 0:
            print ser.read(1)
        time.sleep(1)
        espeak.synth("Welcome to Brainmotic")
        ser.writelines("x")
        isRunning = True

'''
    k = cv2.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord('q'):
        ser.writelines("s")
        break
'''