from operator import index
import numpy as np


def task1():
    print(np.__version__)


def task2():
    print(np.array(np.arange(10)))


def task3():
    a = np.ones((3,3), dtype="bool")
    b = np.full((3, 3), True, dtype=bool)
    print(b)


def task4():
    x = np.arange(10)
    print(x[x%2 == 1])


def task5():
    x = np.arange(10)
    a = np.where(x%2==1, 0, x)
    # x[x%2==1] = -1
    print(a)


def task6():
    x = np.arange(10)
    print(np.where(x%2==1, 0, x))


def task7():
    print(np.arange(10).reshape((-1,5)))


def task8():
    a = np.arange(10).reshape(2,-1)
    b = np.repeat(1, 10).reshape(2,-1)
    # print(np.append(a,b,axis=0))
    # print(np.concatenate([a, b], axis=0))
    print(np.vstack([a, b]))
    # print(np.r_[a, b])


def task9():
    a = np.arange(10).reshape(2,-1)
    b = np.repeat(1, 10).reshape(2,-1)
    # print(np.hstack([a,b]))
    # print(np.append(a,b,axis=1))
    # print(np.concatenate([a,b],axis=1))
    print(np.c_[a,b])


def task10():
    a = np.array([1,2,3])
    
    b = np.repeat(a,3,axis=0)
    c = np.repeat([a],3,axis=0).ravel()
    # np.r_[np.repeat(a, 3), np.tile(a, 3)]
    print(np.hstack([b,c]))


def task11():
    a = np.array([1,2,3,2,3,4,3,4,5,6])
    b = np.array([7,2,10,2,7,4,9,4,9,8])

    print(np.intersect1d(a,b))

def task12():
    a = np.array([1,2,3,4,5])
    b = np.array([5,6,7,8,9])

    print(np.setdiff1d(a,b))

def task13():
    a = np.array([1,2,3,2,3,4,3,4,5,6])
    b = np.array([7,2,10,2,7,4,9,4,9,8])

    print(np.argwhere(a==b))

def task14():
    a = np.array([2, 6, 1, 9, 10, 3, 27])
    #OR
    index = np.where((a >= 5) & (a <= 10))
    print(a[(a>5)&(a<10)])


def task15():
    a = np.array([5, 7, 9, 8, 6, 4, 5])
    b = np.array([6, 3, 4, 8, 9, 7, 1])
    c = [max(i) for i in zip(a,b)]
    print(c)

    # pair_max = np.vectorize(maxx, otypes=[float])

def task16():
    arr = np.arange(9).reshape(3,3)
    arr[:,[0,1]] = arr[:,[1,0]]
    print(arr)
    print(arr[:, [1,0,2]])


def task17():
    arr = np.arange(9).reshape(3,3)
    arr[[0,1],:] = arr[[1,0],:]
    print(arr)


def task18():
    arr = np.arange(9).reshape(3,3)
    arr = np.flip(arr)
    print(arr)
    print(arr[:, ::-1])

def task19():
    arr = np.random.rand(3,3)
    
    rand_arr = np.random.uniform(5,10, size=(5,3))
    rand_arr = np.random.randint(low=5, high=10, size=(5,3)) + np.random.random((5,3))


def task20():
    rand_arr = np.random.random((5,3))
    print(rand_arr.round(3))
    #OR
    # np.set_printoptions(precision=3)


def task21():
    np.random.seed(100)
    rand_arr = np.random.random([3,3])/1e3
    np.set_printoptions(suppress=True)
    print(rand_arr)


def task22():
    np.set_printoptions(threshold=6)
    a = np.arange(15)
    print(a)


def task25():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')
    names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')



def task26():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
    print([i[4] for i in iris_1d[:10]])


def task27():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
    
    iris_2d = np.array([row.tolist()[:-1] for row in iris_1d])
    print(iris_2d[:4])
    # iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])


def task28():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype=None, usecols=[0,1,2,3])

    iris_nums = iris[:,0]
    functions = (np.mean, np.median, np.std)
    data = [f(iris_nums) for f in functions]
    print(data)
    

def task29():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

    b = np.linalg.norm(sepallength)
    print(sepallength/b)


def task31():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

    p = np.percentile(sepallength, q=[5,95])
    print(p)

def task32():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_2d = np.genfromtxt(url, delimiter=',', dtype=None, usecols=[0,1,2,3])
    iris_2d = np.insert(iris_2d, 2, 0.03)
    print(iris_2d)
    

def task():
    # Create a 3x3x3 array with random values 
    print(np.random.random([3,3,3]))


def task():
    print(np.random.random([10,10]))
    np.min()
    np.max()


def task():
    v = np.random.random(30)
    np.mean(v)


def task():
    a = np.ones((4,4))
    a[1:-1,1:-1] = np.arange(4).reshape(2,2)
    print(a)


def task():
    a = np.random.randint(0,10,(3,3))
    z = np.zeros((5,5))
    z[1:-1,1:-1] = a
    print(z)
    # or
    Z = np.ones((5,5))
    # Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
    # print(Z)

    # # Using fancy indexing
    # Z[:, [0, -1]] = 0
    # Z[[0, -1], :] = 0
    # print(Z)


def task():
    m = np.ones((5,5))
    n = np.array([1,2,3,4]*4).reshape((4,4))
    # print(np.tril(n))
    Z = np.diag(1+np.arange(4),k=-1)
    print(Z)


def task():
    a = np.zeros((8,8))
    a[::2,::2] = 1
    a[1::2,1::2] = 1
    print(a)


def task():
    print(100%42)

# task 32


if __name__ == "__main__":
    task()

