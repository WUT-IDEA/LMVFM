function change_zero_feature(path)
% 将矩阵中从0开始的离散属性变成从1开始

data = load(path);
f_type = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
for i=1:(size(data,2)-1)
	if(f_type(i) == 0)  % 如果属性是离散属性
		d_r = unique(data(:,i+1));
		if(d_r(1) == 0) %如果属性从0开始编号，让编号+1
			data(:,i+1) = data(:,i+1)+1;
		end
	end
end
matrix2txt(data,'credit1_miss_recode.txt');

