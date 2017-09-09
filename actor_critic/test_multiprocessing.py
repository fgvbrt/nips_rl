from multiprocessing import Process, Queue
import numpy as np


def proc1(q1, q2):
    w = q1.get()
    print 'process 1, w is:'
    print w

    print 'put'
    q2.put(np.ones(10))
    print 'done put'


def run():
    q1 = Queue()
    q2 = Queue()

    worker = Process(target=proc1,
                     args=(q1, q2)
                     )
    worker.daemon = True
    worker.start()

    q1.put(np.zeros(10))
    w = q2.get()
    print 'main process, w is:'
    print w

    worker.join()


if __name__ == '__main__':
    run()
