function result = feature_relation(X, Y, V, m)

% 函数功能： 根据属性关联度补齐缺失数据
% X：均值补齐后的完备矩阵
% Y：待补齐矩阵
% V：属性关联度矩阵
% m：关联属性个数，默认为所有其他属性

% result：返回值，表示补齐之后的

if ~exist('m','var')
	m = size(X,2)-1;  % 默认为所有其他的属性个数
end

% 计算属性之间的关联度
feature_num = size(X,2);
v_f = ones(feature_num, feature_num)*-5;
for i=1:feature_num
	for j=(i+1):feature_num
		v_f(i,j) = V(i,:)*V(j,:)';
	end
end
matrix2txt(v_f, 'v_f.txt');

% 利用关联度补齐缺失数据
result = zeros(size(X'));
for i=1:feature_num
	[arr,idx] = sort(v_f(i,:));       % 递增排序
	arr = fliplr(arr);   % 将关联度倒序排序
	idx = fliplr(idx);
	X1 = X';
	result(i,:) = arr(1:m)*X1(idx(1:m),:)/sum(arr(1:m));
end

result = result';