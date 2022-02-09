import re
from cv2 import mean
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np

# print(pd.__version__)

def task2():
    mylist = list('abcedfghijklmnopqrstuvwxyz')
    myarr = np.arange(26)
    mydict = dict(zip(mylist, myarr))
    
    print(pd.Series(mylist))
    print(pd.Series(myarr))
    print(pd.Series(mydict))


def task3():
    mylist = list('abcedfghijklmnopqrstuvwxyz')
    myarr = np.arange(26)
    mydict = dict(zip(mylist, myarr))
    ser = pd.Series(mydict)

    df = ser.to_frame().reset_index()
    print(df.head())


def task4():
    ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
    ser2 = pd.Series(np.arange(26))
    df = pd.concat((ser1,ser2), axis=1)
    # df = pd.DataFrame({"col1":ser1,
    #                 "col2":ser2})

    print(df.head())

def task5():
    ser = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
    ser.index.name = "alp" # index name
    ser.name = 'alphabet' # ser name
    print(ser.head())

def task6():
    ser1 = pd.Series([1, 2, 3, 4, 5])
    ser2 = pd.Series([4, 5, 6, 7, 8])
    # ser1.
    ser3 = ser1.map(lambda x: x not in ser2.values)
    print(ser1[~ser1.isin(ser2)])
    
    
def task7():
    ser1 = pd.Series([1, 2, 3, 4, 5])
    ser2 = pd.Series([4, 5, 6, 7, 8])

    s = np.setdiff1d(np.union1d(ser1,ser2), np.intersect1d(ser1,ser2))
    # OR
    ser_u = pd.Series(np.union1d(ser1, ser2))  # union
    ser_i = pd.Series(np.intersect1d(ser1, ser2))  # intersect
    ser_u[~ser_u.isin(ser_i)]
    

def task8():
    ser = pd.Series(np.random.normal(10, 5, 25))
    print(ser.median())
    print(ser.min())
    print(ser.max())
    print(ser.quantile(q=0.25))
    print(ser.quantile(q=0.75))
    # or 
    print(np.percentile(ser, q=[0, 25, 50, 75, 100]))
    

def task9():
    ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))
    print(ser.value_counts())


def task10():
    np.random.RandomState(100)
    ser = pd.Series(np.random.randint(1, 5, 12))

    # print(ser.value_counts().index)
    print(ser[ser.isin(ser.value_counts().index[:2])])
    print(ser[ser.isin(ser.value_counts().index[2:])])
    

def task11():
    ser = pd.Series(np.random.random(20))
    bins = pd.cut(ser, np.linspace(0,ser.max(),10), labels=[str(i)+'th' for i in range(1,10)])
    print(pd.concat([ser,bins], axis=1, keys=['data', 'bins']))

def task12():
    ser = pd.Series(np.random.randint(1, 10, 35))
    df = pd.DataFrame(data=ser.values.reshape((5,7)))
    print(df)


def task13():
    ser = pd.Series(np.random.randint(1, 10, 7))
    # print(ser[ser%3==0].keys())
    # OR
    # print(np.argwhere(ser%3==0)) # doesnt work


def task14():
    ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
    pos = [0, 4, 8, 14, 20]
    print(ser[pos])
    # or
    print(ser.take(pos))


def task15():
    ser1 = pd.Series(range(5))
    ser2 = pd.Series(list('abcde'))
    print(pd.concat([ser1,ser2]))
    print(pd.concat([ser1,ser2], axis=1))

def task16():
    ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
    ser2 = pd.Series([1, 3, 10, 13])
    print(ser1[ser1.isin(ser2)].keys())
    
    [np.where(i == ser1)[0].tolist()[0] for i in ser2]
    # Solution 2
    [pd.Index(ser1).get_loc(i) for i in ser2]


def task17():
    truth = pd.Series(range(10))
    pred = pd.Series(range(10)) + np.random.random(10)
    
    print(np.mean((truth-pred)**2))

def task18():
    ser = pd.Series(['how', 'to', 'kick', 'ass?'])
    print(ser.str.capitalize())
    # print(ser.map(lambda x: x.capitalize()))
    
def task19():
    ser = pd.Series(['how', 'to', 'kick', 'ass?'])
    lns = ser.map(lambda x: len(x))
    print(lns)


def task20():
    ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])
    

def task21():
    ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303',
                 '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
    
    ser = ser.astype('datetime64')
    #or
    # pd.to_datetime(ser)
    #or
    # from dateutil.parser import parse
    # ser.map(lambda x: parse(x))

def task22():
    ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303',
                 '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
    
        # Solution
    from dateutil.parser import parse
    ser_ts = ser.map(lambda x: parse(x))

    # day of month
    print("Date: ", ser_ts.dt.day.tolist())

    # week number
    print("Week number: ", ser_ts.dt.weekofyear.tolist())

    # day of year
    print("Day number of year: ", ser_ts.dt.dayofyear.tolist())

    # day of week
    print("Day of week: ", ser_ts.dt.weekday_name.tolist())

def task24():
    ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])
    z = ser.map(lambda x: len([i for i in x if i in "AEIOUaeiou"]))>=2
    print(ser[z])
    # OR
    # from collections import Counter
    # mask = ser.map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiou')]) >= 2)
    

def task25():
    emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])
    pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'

    print(emails[emails.str.match(pattern)])
    #OR
    # print([x[0] for x in [re.findall(pattern, email) for email in emails] if len(x) > 0])

def task26():
    fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
    weights = pd.Series(np.linspace(1, 10, 10))
    # print(weights.where(fruit==i for i in fruit.unique()))
    print([np.mean(weights[fruit==i])for i in fruit.unique()])
    # OR
    # weights.groupby(fruit).mean()

def task27():
    p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    # np.linalg.norm(p-q)

    # sum((p - q)**2)**.5
    
    # np.sqrt(np.sum(z**2))


def task28():
    ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
    
    dd = np.diff(np.sign(np.diff(ser)))
    peak_locs = np.where(dd == -2)[0] + 1

def task29():
    my_str = 'dbc deb abed gade'
    s = pd.Series(list(my_str))
    least_freq = s.value_counts().keys()[-1]
    s.replace(' ', least_freq, inplace=True)
    print(''.join(s))


def task30():
    ser = pd.Series(np.random.randint(1,10,10), pd.date_range('2000-01-01', periods=10, freq='W'))
    print(ser)

def task31():
    ser = pd.Series([1,10,3,np.nan], index=pd.to_datetime(['2000-01-01',
                                                         '2000-01-03', 
                                                         '2000-01-06', 
                                                         '2000-01-08']))
    idx = ser.index[[0,-1]]
    idx = pd.date_range(*idx)
    ser = ser.reindex(idx, fill_value=ser.mean())
    print(ser)
    #OR
    # ser.resample('D').ffill()
    #OR
    # ser.resample('D').bfill()
    # ser.resample('D').bfill().ffill()

def task32():
    ser = pd.Series(np.arange(20) + np.random.normal(1, 10, 20))
    autocorrelations = [ser.autocorr(i).round(2) for i in range(11)]


def task33():
    # Solution 1: Use chunks and for-loop
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)
    df2 = pd.DataFrame()
    for chunk in df:
        df2 = df2.append(chunk.iloc[0,:])


    # Solution 2: Use chunks and list comprehension
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)
    df2 = pd.concat([chunk.iloc[0] for chunk in df], axis=1)
    df2 = df2.transpose()

    # Solution 3: Use csv reader
    import csv          
    with open('BostonHousing.csv', 'r') as f:
        reader = csv.reader(f)
        out = []
        for i, row in enumerate(reader):
            if i%50 == 0:
                out.append(row)

    df2 = pd.DataFrame(out[1:], columns=out[0])
    print(df2.head())


def task34():
   pass


def task35():
    L = pd.Series(range(15))
    def gen_strides(a, stride_len=5, window_len=5):
        n_strides = ((a.size-window_len)//stride_len) + 1
        return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])

def task36():
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', usecols=['crim', 'medv'])


def task37():
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
    print(df[df.columns[:2]])
    print(type(df))
    print(df.describe())
    print(df.shape)

    df_arr = df.values

    df_list = df.values.tolist()

def task38():
    pass


def task39():
    pass




if __name__ == "__main__":
    task37()
