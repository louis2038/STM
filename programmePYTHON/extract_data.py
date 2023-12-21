import pandas as pd
import numpy as np
import plotnine
import csv
import random
csvfile = open("encoded_mcu_features.csv", "r")
reader = csv.reader(csvfile, delimiter=",")
x = list(reader)
result = np.array(x)
N,nbs_cat = result.shape

fieldnames = ['MPN','Manufacturer','ADC Channels','ADC Resolution','CAN',
 'Core Architecture','Data Bus Width','Device Core','Ethernet','Family',
 'Family Name','I2C','I2S','Instruction Set Architecture','Interface Type',
 'Maximum CPU Frequency','Maximum Clock Rate',
 'Maximum Operating Supply Voltage','Maximum Operating Temperature',
 'Minimum Operating Supply Voltage','Minimum Operating Temperature',
 'Mounting','Number of ADCs','Number of Programmable I/Os',
 'Number of Timers','Operating Supply Voltage','PACKAGE_DIMENSION_H',
 'PACKAGE_DIMENSION_L','PACKAGE_DIMENSION_W','PCB','Pin count',
 'Program Memory Size','Program Memory Type','RAM Size','SPI',
 'Supplier_Package','Temperature Flag','Typical Operating Supply Voltage',
 'UART','USART','USB']

csvfile = open("small_edit_mcu_features.csv","w",encoding='UTF8',newline='')
nbsfeat = len(fieldnames)
writer = csv.writer(csvfile)

N_small = 1000
writer.writerow(fieldnames)
Item = [i for i in range(N)]
for _ in range(N_small):
    D = [0]*nbsfeat
    n = random.choice(Item)
    Item.remove(n)
    for fet in range(nbsfeat):
        D[fet] = result[n,fet]
    
    writer.writerow(D)
