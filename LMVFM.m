function LMVFM(x_matrix, x_labels, y_matrix, y_labels, feature_types, index, K, V, m)

% LMVFM：LMVFM算法主函数
% x_matrix：完备矩阵，这里的处理是将缺失样本用均值补齐后依次添加到无缺失值矩阵末尾
% x_labels：完备矩阵标签集
% y_matrix：待补齐矩阵
% y_labels：待补齐矩阵标签集
% feature_types：特征属性类别，0-离散，1-连续
% index：缺失值索引矩阵。针对矩阵y_matrix，有缺失的位置为1，其他为0
% K：KNN最近邻个数
% V：属性关联度矩阵
% m：关联属性个数

tic

X = load(x_matrix);
X_L = load(x_labels);
Y = load(y_matrix);
Y1 = Y;
Y_L = load(y_labels);
F_T = load(feature_types);
Index = load(index);
v = load(V);
t1 = toc;
%% 缺失值初始化
tic;
init_feature = zeros(1,size(F_T,2));  % 记录每个属性的初始化值
for i=1:size(F_T,2)
	if(F_T(i) == 1)  % 连续值用均值初始化
		init_feature(i) = mean(X(:,i));   % 第i个属性的均值
	else  % 离散值用出现最多的值初始化
		init_feature(i) = mode(X(:,i));  % 第i个属性出现最多的值
	end
end
% matrix2txt(init_feature, 'init_feature.txt'); 
init_rep = repmat(init_feature, size(Y,1),1);  % 将init_feature扩展成与y_matrix同样规模的矩阵，每一列用均值填充
% matrix2txt(init_rep,'init_rep.txt');
Y(Index == 1) = init_rep(Index == 1);
time_mean = toc;
matrix2txt(Y, 'data_mean.txt');    % 将均值补充的结果输出

tic
X_all = zeros(size(X,1)+size(Y,1),size(X,2));  % 保存所有的样本，将Y添加到X矩阵之后
X_all_labels = zeros(size(X,1)+size(Y,1),1);  % 保存所有的样本标签
for i=1:size(X,1)
	X_all(i,:) = X(i,:);
	X_all_labels(i) = X_L(i);
end
for i=(size(X,1)+1):size(X_all,1)
	X_all(i,:) = Y(i-size(X,1),:);
	X_all_labels(i) = Y_L(i-size(X,1));
end
t2 = toc;

% KNN
tic;
result_knn = KNN(X_all, Y, X_all_labels, Y_L, K, 'cosine',Index);
time_knn = toc;

% 属性相似度
tic;
fea_sim = feature_similarity(X_all, 'cosine');
result_fea_sim = fea_sim((size(fea_sim,1)-size(Y,1)+1):size(fea_sim,1),:);  % 截取后面的待补齐样本
matrix2txt(result_fea_sim, 'result_fea_sim.txt');

% 属性关联度
if ~exist('m','var')
	m = size(X,2)-1;  % 默认为所有其他的属性个数
end
fea_rel = feature_relation(X_all, Y, v, m);
result_fea_rel = fea_rel((size(fea_rel,1)-size(Y,1)+1):size(fea_rel,1),:);  % 截取后面的待补齐样本
matrix2txt(result_fea_rel, 'result_fea_rel.txt');

% 确定参数
% MRES = zeros(1,3);
% miss_num = size(Y1(Index == 1),1);   % 缺失值数量
% A1 = Y1(Index == 1) - result_knn(Index == 1);  % 求差
% MRES(1) = sum(abs(A1)) / sum(Y1(Index == 1));
% A2 = Y1(Index == 1) - result_fea_sim(Index == 1);  % 求差
% MRES(2) = sum(abs(A2)) / sum(Y1(Index == 1));
% A3 = Y1(Index == 1) - result_fea_rel(Index == 1);  % 求差
% MRES(3) = sum(abs(A3)) / sum(Y1(Index == 1));
% tem = 1./MRES;
% sum1 = sum(tem);
% tem = tem/sum1;

% LMVFM = b0 + b1*result_knn + b2*result_fea_sim + b3*result_fea_rel;    % 参数暂时取经验值
% result = tem(1)*result_knn + tem(2)*result_fea_sim + tem(3)*result_fea_rel;    % 设置b0=0，其他参数设置为0.33
result = 0.33*result_knn + 0.33*result_fea_sim + 0.33*result_fea_rel;    % 设置b0=0，其他参数设置为0.33
Y(Index == 1) = result(Index == 1);
t3= toc;

matrix2txt(Y, 'data_lmvfm.txt');
matrix2txt([time_mean, time_knn, t1+t2+t3+time_knn+time_mean], 'time.txt');