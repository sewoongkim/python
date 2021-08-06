import numpy as np
import pandas as pd
import os

base_dir = 'G:\SampleGit01\python'
excel_file = 'data.xlsx'
excel_dir = os.path.join(base_dir, excel_file)

df_from_excel = pd.read_excel(excel_dir, # write your directory here
    sheet_name = 'F_뽑기' 
    # header = 1 
    # dtype = {'구분': str}
)

    # , 'sales_representative': np.int64, 'sales_amount': float dictionary type
    #names = ['region', 'sales_representative', 'sales_amount'], 
    # index_col = 'id', 
    # na_values = 'NaN', 
    # thousands = ',', 
    # nrows = 10, 
    # comment = '#')

# print(df_from_excel)

df_from_excel.head


class Message:
    def __init__(self,msg):
        self.msg = msg

    def __repr__(self):
        return 'Message: %s' % self.msg


x = Message('I have a secret.')
x
