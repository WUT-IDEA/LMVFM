def autoNorm(data):         #传入一个矩阵  
    mins = data.min(0)      #返回data矩阵中每一列中最小的元素，返回一个列表  
    maxs = data.max(0)      #返回data矩阵中每一列中最大的元素，返回一个列表  
    ranges = maxs - mins    #最大值列表 - 最小值列表 = 差值列表  
    normData = np.zeros(np.shape(data))     #生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据  
    row = data.shape[0]                     #返回 data矩阵的行数  
    normData = data - np.tile(mins,(row,1)) #data矩阵每一列数据都减去每一列的最小值  
    normData = normData / np.tile(ranges,(row,1))   #data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）  
    return normData  