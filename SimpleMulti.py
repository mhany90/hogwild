
from multiprocessing import Process, Value, Array

#num = Value('d', 0.0, lock = False)

num = 0

def f1():
    for x in range(10000):
        global num
        num =+ 1
        #num.value =+ 1
        print(num)

def f2():
    for x in range(10000):
        global num
        num =- 1
        #num.value =- 1
        print(num)

if __name__ == '__main__':


    inp = 1
    p1 = Process(target=f1)
    p2 = Process(target=f2)
    p2.start()
    p1.start()


    #p1.join()
    #p2.join()




