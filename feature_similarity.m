function result = feature_similarity(X, method, M, deta)

% 函数功能：利用属性相似度算法计算待补齐样本集的补齐结果，后面根据缺失值位置索引对应获取补齐值即可
% X: 原始矩阵的完备矩阵，其中的缺失值用平均值（出现次数最多的值）替代
% method: 相似度（或距离）计算方法
% M: 相似属性个数，默认为所有属性的个数-1
% deta: 相似度阈值矩阵，大于等于这个值的相似度才会被考虑（该参数值需要谨慎选择）
%       本算法默认为中位数，实际上，该值的选择应该根据属性相似度的矩阵进行选择，这样可以做到每个样本满足条件的相似属性都不一样。

if ~exist('M','var')
	M = size(X,2)-1;  % 默认为size(X,2)-1
end

if ~exist('method','var')
	method = 'cosine';
end

% idx的每一行对应Y中对应样本的K个最近邻在X中的索引，dist表示对应距离矩阵
% 这里将矩阵转置，求出每一个属性对于其他所有属性（包括自己）的距离，后面根据相似度条件筛选
[idx, dist] = knnsearch(X',X','dist',method,'k',M+1);
idx = idx(:,2:size(idx,2));     % 去掉第一列
dist = dist(:,2:size(dist,2));
sim_mat = 1./dist;

[row, col] = size(X');
result = zeros(size(X'));  % 定义结果矩阵
matrix2txt(idx, 'idx.txt');
matrix2txt(dist, 'dist.txt');

% 一次将所有属性都计算KNN得到结果，后面根据需要取对应的索引位置即可
index_arr = 2:M;

if ~exist('deta','var')
	deta = median(sim_mat(:,1:M),2);   % 相似度阈值默认为每一行的中位数
end
matrix2txt(deta, 'deta.txt');

bigger_than_deta_index = zeros(1,M);  % 记录满足条件的索引位置

for i=1:row
	for j=1:M
		if(sim_mat(i,j) >= deta(j))
			bigger_than_deta_index(j) = 1;
		end
	end
	similarity_sum = sum(sim_mat(i,bigger_than_deta_index==1));  % 求相似度的和
	X1 = X';
	result(i,:) = sim_mat(i,bigger_than_deta_index==1)*X1(idx(i,bigger_than_deta_index==1),:)/similarity_sum;   % 计算待补齐的值
end
result = result';  % 转置称为正常的矩阵
