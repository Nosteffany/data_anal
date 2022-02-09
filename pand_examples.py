from numpy.core.numeric import count_nonzero
import pandas as pd
import numpy as np


def unit1():


    obj = pd.Series([4, 7, -5, 3])
    obj.values
    obj.index

    obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
    # we can apply elementwise functions like np.exp to our dataframe and series

    pd.isnull()
    pd.notnull()
    #there are such methods too obj.isnull()
    
    # obj4.name = 'population'
    # obj4.index.name = 'state'

    data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
    frame = pd.DataFrame(data)

def indexing_data():
    data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
    frame = pd.DataFrame(data)
    frame.index = [1,2,3,4,5,6]
    print(frame)
    print(frame.iat[0,2])


def mapping_functions():
    data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                        'Pastrami', 'corned beef', 'Bacon',
                        'pastrami', 'honey ham', 'nova lox'],
                        'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
    # Suppose you wanted to add a column indicating the type of animal that each food came from. 
    
    meat_to_animal = {'bacon': 'pig',
                    'pulled pork': 'pig',
                    'pastrami': 'cow',
                    'corned beef': 'cow',
                    'honey ham': 'pig',
                    'nova lox': 'salmon'}
    
    lowercased = data['food'].str.lower()
    data['animal'] = lowercased.map(meat_to_animal)
    
    # We could also have passed a function that does all the work:
    
    l = data['food'].map(lambda x: meat_to_animal[x.lower()])
    
    print(l)
    # Using map is a convenient way to perform element-wise transformations and other
    # data cleaning–related operations.

def binning_data():
    ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
    bins = [18, 25, 35, 60, 100]
    cats = pd.cut(ages, bins)
    print(cats)
    print(cats.codes) # return array of bins each element belongs to
    print(cats.categories)
    print(pd.value_counts(cats))

def detect_outliers():
    data = pd.DataFrame(np.random.randn(1000, 4))
    col = data[2]
    g3 = col[np.abs(col) > 3]
    result = data[(np.abs(data) > 3).any(1)]

def perm_random_sampling():
    df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
    sampler = np.random.permutation(5)

    # that array can then be used in iloc-based indexing or the equivalent take function
    df = df.take(sampler)
    print(df)
    # To select a random subset without replacement, you can use the sample method on
    # Series and DataFrame:
    print(df.sample(n=3))
    # To generate a sample with replacement (to allow repeat choices), pass replace=True to sample
    choices = pd.Series([5, 7, -1, 6, 4])
    draws = choices.sample(n=10, replace=True)
    print(draws)

def indicators_dummies():
    df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                       'data1': np.random.randn(6)})
    
    data = pd.get_dummies(df['key'])
    dummies = pd.get_dummies(df['key'], prefix='key')
    print(dummies)
    df_with_dummy = df[['data1']].join(dummies)
    print(df_with_dummy)

def hierarchical_indexing():
    #     Hierarchical indexing is an important feature of pandas that enables you to have mul‐
    # tiple (two or more) index levels on an axis.
    data = pd.Series(np.random.randn(9),index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                                                [1, 2, 3, 1, 3, 1, 2, 2, 3]])
    print(data)
    print(data.index.levels)
    # With a hierarchically indexed object, so-called partial indexing is possible
    # print(data['b'])
    # print(data['b':'c'])
    # print(data.loc[['b', 'd']])
    # print(data.loc[:, 2])
    # print(data)
    # print(data.unstack().stack())
    # With a DataFrame, either axis can have a hierarchical index:
    frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                        index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                        columns=[['Ohio', 'Ohio', 'Colorado'],
                        ['Green', 'Red', 'Green']])

    print(frame)
    # With partial column indexing you can similarly select groups of columns
    print(frame['Ohio'].loc['a'])

def reordering_sorting_levels():
    # at times you will need to rearrange the order of the levels on an axis or sort the data
    # by the values in one specific level. 
    frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                        index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                        columns=[['Ohio', 'Ohio', 'Colorado'],
                        ['Green', 'Red', 'Green']])
    

    frame.index.names = ['key1', 'key2']
    
    # frame = frame.swaplevel('key1', 'key2')
    # frame = frame.swaplevel(0,1) is the same as above
    print(frame)
    # frame = frame.sort_index(level=1)
    # print(frame)
    # frame = frame.swaplevel(0,1).sort_index(level=0)
    # print(frame)

def sum_stats_by_level():
    frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                        index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                        columns=[['Ohio', 'Ohio', 'Colorado'],
                        ['Green', 'Red', 'Green']])
    
    fsum = frame.sum(level=1, axis=1)
    print(fsum)
    print(frame)
    print(frame.sum(level='key2'))

def dbstyle_joins():
    # Merge or join operations combine datasets by linking rows using one or more keys
    df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                        'data1': range(7)})
    df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                        'data2': range(3)})

    print(df1)
    print(df2)
    # This is an example of a many-to-one join; the data in df1 has multiple rows labeled a
    # and b, whereas df2 has only one row for each value in the key column. Calling merge
    # with these objects we obtain:
    print(pd.merge(df1, df2))
    # merging uses the overlaping column
    # If the column names are different in each object, you can specify them separately:
    df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],'data1': range(7)})
    df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'],'data2': range(3)})
    print(pd.merge(df3,df4, left_on='lkey', right_on='rkey'))
    # You may notice that the 'c' and 'd' values and associated data are missing from the
    # result. By default merge does an 'inner' join; er possible options are 'left','right', and 'outer'. 
    print(pd.merge(df1, df2, how='outer'))
    # Many-to-many merges have well-defined, though not necessarily intuitive, behavior.
    # Here’s an example
    df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],'data1': range(7)})
    df2 = pd.DataFrame({'key': ['a', 'b', 'd'],'data2': range(3)})
    
def merge_index():
    #     n some cases, the merge key(s) in a DataFrame will be found in its index. In this
    # case, you can pass left_index=True or right_index=True (or both) to indicate that
    # the index should be used as the merge key
    left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],'value': range(6)})
    right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
    print(left1)
    print(right1)
    result = pd.merge(left1, right1, left_on='key', right_index=True)
    print(result)

def combine_with_overlap():
    # There is another data combination situation that can’t be expressed as either a merge
    # or concatenation operation. You may have two datasets whose indexes overlap in full
    # or part. As a motivating example, consider NumPy’s where function, which performs
    # the array-oriented equivalent of an if-else expression:
    a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=['f', 'e', 'd', 'c', 'b', 'a'])
    b = pd.Series(np.arange(len(a), dtype=np.float64), index=['f', 'e', 'd', 'c', 'b', 'a'])
    b[-1] = np.nan
    print(np.where(pd.isnull(a), b, a))
    # Series has a combine_first method, which performs the equivalent of this operation
    # along with pandas’s usual data alignment logic:
    print(b.combine_first(a))
    # With DataFrames, combine_first does the same thing column by column, so you
    # can think of it as “patching” missing data in the calling object with data from the
    # object you pass:
    
def reshaping_hierarchical_indexing():
    pass

def long_to_wide_pivot():
    pass

def wide_to_long_pivot():
    pass


# combine_with_overlap()
# merge_index()
# dbstyle_joins()
# sum_stats_by_level()
# reordering_sorting_levels()
# hierarchical_indexing()
# indicators_dummies()
# perm_random_sampling()
# detect_outliers()
# binning_data()
# mapping_functions()

#241