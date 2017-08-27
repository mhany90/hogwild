import multiprocessing as mp
from multiprocessing import Manager
import sys
import time


manager = Manager()
num = mp.Manager().Value('i', 0, lock=True)
lock = mp.Manager().Lock()

def func1():
    global num
    print ('start func1')
    while num.value < 500:
    #for x in range(1000):
        #with num.get_lock():
        lock.acquire()
        num.value += 1
        print(num.value)
        lock.release()
    print ('end func1')

def func2():
    global num
    print ('start func2')
    while num.value > -500:
    #for x in range(1000):
        #with num.get_lock():
        lock.acquire()
        num.value -= 1
        print(num.value)
        lock.release()
    print ('end func2')


if __name__=='__main__':
    ctx = mp.get_context('fork')
    p1 =  ctx.Process(target=func1)
    p1.start()
    p2 = ctx.Process(target=func2)
    p2.start()
    p1.join()
    p2.join()
    sys.stdout.flush()
    print('final: ' , num.value)