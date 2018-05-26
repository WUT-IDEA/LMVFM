function significance = feature_significance(X, each_label, labels, feature_type)

% 计算属性重要度
% each_label: 每个元素表示一个类别
% X: 不包含缺失值的样本集
% labels: X的类别标签
% feature_type: 每一个特征属性是连续还是离散，需要提前指定，0-离散，1-连续

% 返回值：属性重要度数组，行向量

[row_X, col_X] = size(X);
significance = zeros(1,col_X);  % 记录每个属性的属性重要度

for i=1:col_X	
	%连续型属性
	if(feature_type(i) == 1)
		min_max = zeros(size(each_label,2),2); % 保存每个属性在每个类别上的最大值和最小值
		%对每一类别分别求属性的重要度，最后相加
		for c=1:size(each_label,2)
			index_c = labels == each_label(c);  % 所有属于第c类的索引
			X_c = X(index_c);      % 所有属于第c类的样本
			feature_i = X_c(:,i);  % 属性i在第c类上的取值集合
			max_ic = max(feature_i);
			min_ic = min(feature_i);
			min_max(c,:) = [min_ic, max_ic];
		end

		% 下面只处理二分类的情况，即min_max是2*2的矩阵
		tmp = sort([min_max(1,1), min_max(1,2), min_max(2,1), min_max(2,2)]);  % 将两个区间的端点进行排序
		% 如果最小的imn_max(1,1)
		if(tmp(1,1) == min_max(1,1))
			if(tmp(1,2) == min_max(1,2))   % 在两个类别上的区间不重叠
				significance(i) = 2;
			else
				% 第二个元素是另外一个最小值
				if(tmp(1,3) == min_max(2,2))   % 另外一个区间完全被包围
					significance(i) = (min_max(1,2)-min_max(2,2)+min_max(2,1)-min_max(1,1))/(min_max(1,2)-min_max(1,1))
				else
					% 两区间交叉，重要度为各类上的和
					significance(i) = (min_max(2,1)-min_max(1,1))/(min_max(1,2)-min_max(1,1)) + (min_max(2,2)-min_max(1,2))/(min_max(2,2)-min_max(2,1)); 
				end
			end
		else  % 与上面的情况正好反过来
			if(tmp(1,2) == min_max(2,2))   % 在两个类别上的区间不重叠
				significance(i) = 2;
			else
				% 第二个元素是另外一个最小值
				if(tmp(1,3) == min_max(1,2))   % 另外一个区间完全被包围
					significance(i) = (min_max(2,2)-min_max(1,2)+min_max(1,1)-min_max(2,1))/(min_max(2,2)-min_max(2,1))
				else
					% 两区间交叉，重要度为各类上的和
					significance(i) = (min_max(1,1)-min_max(2,1))/(min_max(2,2)-min_max(2,1)) + (min_max(1,2)-min_max(2,2))/(min_max(1,2)-min_max(1,1)); 
				end
			end
		end

	% 离散型属性
	% 对于离散型属性，其重要度定义为：属性f未出现在其他类别中的属性值样本数占该类别下总样本数的比例
	else		
		different_ele = unique(X(:,i));  % 获取属性所有不同的取值，此处为列向量
		dif_eles = ones(size(each_label,2),size(different_ele,1))*-1;  % 保存属性在每一个类上的不同取值，并赋予初始值-1（属性取值中没有-1）
		for c=1:size(each_label,2)
			index_c = labels == each_label(c);  % 所有属于第c类的索引
			X_c = X(index_c);      % 所有属于第c类的样本
			dif = unique(X_c(:,i));  % 属性i在第c类上的取值集合
			dif_eles(c,1:size(dif,1)) = dif;  % 对dif_eles进行赋值
		end

		% 以下只处理二分类
		if(dif_eles(1,:) == dif_eles(2,:))   % 属性在不同类别上的取值相同
			significance(i) = 0;
		else
			diff1 = setdiff(dif_else(1,:),dif_else(2,:));  % 在第一个类别下属性的唯一取值
			diff2 = setdiff(dif_else(2,:),dif_else(1,:));  % 在第二个类别下属性的唯一取值
			num1 = 0;
			num2 = 0;
			for j=1:size(diff1,2)
				num1 = num1 + size(X(X(:,i) == diff1(j)),1);
			end
			sig = num1 / size(labels(labels == each_label(1)));  % 属性在类别1上的重要度
			for j=1:size(diff2,2)
				num2 = num2 + size(X(X(:,i) == diff2(j)),1);
			end
			significance(i) = sig + num2 / size(labels(labels == each_label(2)));  %
		end
	end
end

