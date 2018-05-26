# 调用impyute包中的数据补齐方法，主要有Multivariate Imputation，Expectation Maximization
import numpy as np
import time
import copy
from impyute.imputations.cs import mice
from impyute.imputations.cs import em

# 加载数据集
path = 'E:\\大论文相关\\LMVFM\\data\\compare';
x = np.loadtxt(path + '\\australian\\30%\\missing_other.txt');
x1 = copy.deepcopy(x)

# Multivariate Imputation by Chained Equations
time1 = time.clock();
data_mice = mice(x);
time_mice = time.clock() - time1;

#Expectation Maximization
time1 = time.clock();
data_em = em(x1);
time_em = time.clock() - time1;

np.savetxt('data_em.txt',data_em,'%.2f');
np.savetxt('data_mice.txt',data_mice,'%.2f');  
print(time_mice ,'----', time_em)