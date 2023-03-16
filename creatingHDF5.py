import pandas as pd
import numpy as np

csvfile = 'Raw Data walking jacket.csv'

df = pd.read_csv(csvfile)
hdf = pd.HDFStore("hdf_file.hdf5")

hdf.put('key1', df, format='table', data_columns=True) #put data in hdf file
df2 = pd.DataFrame(np.random.rand(5,3))
hdf.put('/testing/key2',df2)