import numpy as np
from multiprocessing import Process, Array, Value, cpu_count
from ctypes import  c_float
from time import sleep


def get_shared_arr(shape):
    w = Array(c_float, np.ones(shape).ravel())
    return np.ctypeslib.as_array(w.get_obj()).reshape(shape)


def proc1(a, shape):
    print 'i am process 1, i get array and want to modify it'
    print a
    a[:] = np.random.rand(*shape)


def proc2(a):
    print 'i am process 2, lets see on array'
    print a


def main():
    shape = (3, 2)
    a = get_shared_arr(shape)

    p1 = Process(target=proc1, args=(a, shape))
    p1.daemon = True
    p1.start()
    sleep(10)

    p2 = Process(target=proc2, args=(a, ))
    p2.daemon = True
    p2.start()
    sleep(10)

    p1.join()
    p2.join()

if __name__ == '__main__':
    main()