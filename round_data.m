function round_data()

%
path = './data/Jiangsu/all/mean_all.txt';
data = load(path);
feature_type = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
for i=1:size(feature_type,2)
	if(feature_type(i) == 0)
		data(:,i) = round(data(:,i));
	end
end
matrix2txt(data,'mean_all_new.txt');