function result = KNN(X, Y, K, method)

% 函数功能：利用KNN算法计算待补齐样本集的补齐结果
% X: 无缺失值的矩阵
% Y: 待补齐的矩阵
% K: 最近邻个数，默认为10
% method: 相似度（或距离）计算方法

if ~exist('K','var')
	K = 10;  % 默认为10
end

if ~exist('method','var')
	method = 'cosine';
end

[idx, dist] = knnsearch(X,Y,'dist',method,'k',K);  % idx的每一行对应Y中对应样本的K个最近邻在X中的索引，dist表示对应距离矩阵
[row, col] = size(Y);
result = zeros(size(Y));  % 定义结果矩阵

% 一次将所有属性都计算KNN得到结果，后面根据需要取对应的索引位置即可
for i=1:row	
	similarity_sum = sum(1./dist(i,:));  % 将距离的倒数作为相似度，求相似度的和
	result(i,:) = 1./dist(i,:)*X(idx(i,:),:)/similarity_sum;   % 计算待补齐的值
end
