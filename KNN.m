function result = KNN(X, Y, label_X, label_Y, K, method, index)

% 函数功能：利用KNN算法计算待补齐样本集的补齐结果
% X: 缺失值初始化之后的矩阵
% Y: 待补齐的矩阵
% K: 最近邻个数，默认为10
% method: 相似度（或距离）计算方法
% index：缺失值索引

if ~exist('K','var')
	K = 10;  % 默认为10
end

if ~exist('method','var')
	method = 'cosine';
end

% idx的每一行对应Y中对应样本的K个最近邻在X中的索引，dist表示对应距离矩阵
% 由于要去掉本身，所有K+1
[idx, dist] = knnsearch(X,Y,'dist',method,'k',K+1);
idx = idx(:, 2:size(idx,2));      % 去掉第一列，该列表示缺失样本本身
dist = dist(:, 2:size(dist,2));   % 去掉第一列
matrix2txt(dist, 'dist_knn.txt'); 
[row, col] = size(Y);
result = zeros(size(Y));  % 定义结果矩阵

% 一次将所有属性都计算KNN得到结果，后面根据需要取对应的索引位置即可
for i=1:row
	label_Yi = label_Y(i);  % Y中第i个样本的标签
	label_Xi = label_X(idx(i,:));   % K个最近邻的标签矩阵
	label_equal_index = zeros(1,K); % 记录满足与样本Y中的标签相等的idx中的索引，用1表示相等
	for j=1:K
		if(label_Yi == label_Xi(j))
			label_equal_index(j) = 1;  % 若标签相等，则将对应位置索引置为1
		end
	end
	similarity_sum = sum(1./(dist(i,label_equal_index==1)+1));  % 将距离的倒数作为相似度，求相似度的和，+1避免分母为0
	if(similarity_sum == 0)
		result(i,:) = Y(i,:);  %如果K个邻居都与当前样本的标签不一样，则保持该样本的值不变 
	else 
		result(i,:) = 1./(dist(i,label_equal_index==1)+1)*X(idx(i,label_equal_index==1),:)/similarity_sum;   % 计算待补齐的值
	end
end
Y(index == 1) = result(index == 1);
result = Y;

matrix2txt(Y, 'data_knn.txt');   %将KNN补齐的结果输出
