%%%% LMVFM(x_matrix, x_labels, y_matrix, y_labels, feature_types, index, K, V, m)
path = './data/Jiangsu/all';
x_matrix = [path,'/complete_x.txt'];
x_labels = [path,'/complete_x_label.txt'];
y_matrix = [path,'/groundtruth_y.txt'];
y_labels = [path,'/missing_y_label.txt'];
feature_types = [path,'/feature_type.txt'];
index = [path,'/index_y.txt'];
V = [path,'/credit1_v.txt'];

LMVFM(x_matrix, x_labels, y_matrix, y_labels, feature_types, index, 20, V);