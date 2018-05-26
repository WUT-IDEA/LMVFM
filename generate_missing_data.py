import numpy
from impyute.datasets.corrupt import Corruptor
import copy

path = 'E:\\大论文相关\\大论文数据集\\credit1\\credit_all.txt';
#path = 'E:\\大论文相关\\大论文数据集\\australian\\crx_data_modified.txt';
x = numpy.loadtxt(path);
x_delete_label = numpy.delete(x,0,1);    # 删除第一列标签，第二个参数是索引，第三个参数为0表示行，1表示列

################# 注意  ################################
# 对于Jiangsu数据，删除前两列，因为差距过大
#x_delete_label = numpy.delete(x_delete_label,0,1);
#x_delete_label = numpy.delete(x_delete_label,0,1); 
# 发现credit1数据中有两列全是0，非常干扰实验，这里将这两列去掉，对应在原19属性中的位置是12，16，在x_delete_label是10,14
# 去掉后还有15个属性
#x_delete_label = numpy.delete(x_delete_label,10,1); 
#x_delete_label = numpy.delete(x_delete_label,13,1);  # 上面删掉了一列，所以这里是13
f_type = numpy.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);   # Jiangsu属性分类，0-离散，1-连
########################################################


#############################
#去掉Aurstrlian的最后一列
#x_delete_label = numpy.delete(x_delete_label,14,1);
#f_type = numpy.array([0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]);           # australian属性分类，去掉最后一列
#############################

[rows, cols] = x_delete_label.shape ;

# 归一化处理，只处理连续数据
min1 = x_delete_label.min(0);  # 每一列最小值
max1 = x_delete_label.max(0);  # 每一列最大值
distance = max1-min1;
print(min1);
print(distance);
for i in range(rows):
	for j in range(cols):
		if f_type[j] == 1:
			x_delete_label[i,j] = (x_delete_label[i,j]-min1[j])/distance[j];
corruptor = Corruptor(x_delete_label, thr=0.06);      # 2%的缺失率
raw_data = getattr(corruptor, 'mcar')();


# 遍历raw_data，创建缺失值索引矩阵，便于后边计算
raw_data_1 = copy.deepcopy(raw_data);   # 将缺失数据复制一份

missing_index_global = numpy.zeros([rows,cols]);   # 全局索引矩阵
for i in range(rows):
	for j in range(cols):
		if numpy.isnan(raw_data[i][j]):
			missing_index_global[i][j] = 1;    # 赋值索引矩阵


######## 用于LMVFM算法的数据集，需要将缺失数据单独提出来  #########
# 统计有多少行有缺失
count = 0;
for row in raw_data:
	for c in row:
		if numpy.isnan(c):
			count = count +1;
			break;

complete_x = numpy.zeros([rows-count,cols]);    # 完备矩阵x
complete_x_label = numpy.zeros([rows-count,1]); # 完备矩阵x的标签
groundtruth_y = numpy.zeros([count,cols]);        # 缺失之前的原始值
missing_y = numpy.zeros([count,cols]);          # 缺失值矩阵y
missing_y_label = numpy.zeros([count,1]);       # y的标签
is_missing = 0;
i = 0;
j = 0;
# 生成完备矩阵和缺失矩阵
for row in raw_data:
	for c in row:
		if numpy.isnan(c):
			is_missing = 1;
			break;
	if is_missing == 1:
		missing_y[i] = row;
		i = i+1;
	else:
		complete_x[j] = row;
		j = j+1;
	is_missing = 0;

# 生成完备矩阵和缺失矩阵的标签
is_missing = 0;
l = 0;
m = 0;
for i in range(rows):
	for j in range(cols):
		if numpy.isnan(raw_data[i][j]):
			is_missing = 1;
			break;
	if is_missing == 1:
		missing_y_label[l][0] = x[i][0];
		groundtruth_y[l] = x_delete_label[i];
		l = l+1;
	else:
		complete_x_label[m][0] = x[i][0];
		m = m+1;
	is_missing = 0;


[row_y, col_y] = missing_y.shape;
index_y = numpy.zeros([row_y,col_y]);
for i in range(row_y):
	for j in range(col_y):
		if numpy.isnan(missing_y[i][j]):
			index_y[i][j] = 1;

# 用于对比算法的数据集
numpy.savetxt('missing_other.txt',raw_data,'%.6f');                          # 用于对比算法的缺失数据
numpy.savetxt('missing_index_global_other.txt',missing_index_global,'%d');   # 用于对比算法的缺失数据索引，以整数的形式存索引文件
numpy.savetxt('missing_ground_truth_other',x_delete_label,'%.6f');           # 用于对比算法的ground truth

# LMVFM数据集
numpy.savetxt('groundtruth_y.txt',groundtruth_y,'%.6f');           # ground truth
numpy.savetxt('missing_y_label.txt',missing_y_label,'%d');       # 缺失数据标签
numpy.savetxt('index_y.txt',index_y,'%d');                       # 缺失数据索引
numpy.savetxt('complete_x_label.txt',complete_x_label,'%d');     # 完备数据集x的标签
numpy.savetxt('complete_x.txt',complete_x,'%.6f');                 # 完备数据集x
#numpy.savetxt('y.txt',missing_y,'%.2f');   #对于LMVFM来讲，不需要缺失矩阵，暂时不输出

