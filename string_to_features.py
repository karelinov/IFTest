from typing import List

import numpy as np
import pandas as pd


#построение массивов численных параметров из текстовой строки
def str_to_feature(inputString: str, start: int, vectorCount: int) -> List[int]:

  outvectors = list()
  print(inputString)

  for i in range(1, vectorCount+1):
    print(i)
    strindex = int(start+i-1)
    # print(strindex)
    curstr = inputString[strindex:strindex+2]
    print(curstr)
    print(ord(curstr[0]))
    print(ord(curstr[1]))
    outvectors.append(['chars'+str(strindex)+str(strindex+1) ,32767*ord(curstr[0])+ord(curstr[1])])
    outvectors.append(32767*ord(curstr[0])+ord(curstr[1]))

  return outvectors

# лямбда функция для превращения элемента строки в столбце -> в численное значение (для обработки в алгоритме)
# (для DataFrame.apply)
def cl_str_to_feature(input: str, start: int, fillchar:str = ' ') -> str:
  # print('inputtype='+str(type(input)))
  # print(input)
  substr = input.ljust(30,fillchar)[start:start+2]
  # print(substr)
  return 32767*ord(substr[0])+ord(substr[1])


# Вспомогательная функция, генерирует список названий-feature на основе имени столбца с исходными данными
# name -> ['name_01','name_02'... ect ]
def string_to_featurename(input:str, numberoffeatures:int = 10) -> List[str]:
  result = list()

  for i in range(0,numberoffeatures):
    result.append(input+"_"+str(i).rjust(2,'0'))

  return result


# Функция добавляет в Dataframe для указанного (строкового) столбца указанное количество feature - столбцов
# feature для строки - пара букв, преобразованная в int32 число (код одной буквы "закидывается" в верхние байты
# второй - записывается в нижние)
def set_df_column_features(df:pd.DataFrame, columnname:str, numberoffeatures:int = 10, fillchar:str=' '):
  # получаем список названий features
  fnames:List[str] = string_to_featurename(columnname, numberoffeatures)

  for i in range(0,numberoffeatures):
    fname=fnames[i]
    df[fname] = df[columnname].apply(cl_str_to_feature, 0, [i,fillchar]) 


