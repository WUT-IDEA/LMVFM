function calculate(Y,P,index)

% 计算数据补齐算法的MAE, MRE, RMSE
% Y：ground truth
% P：补齐之后的矩阵
% index：缺失值索引

Y = load(Y);
P = load(P);
index = load(index);

miss_num = size(Y(index == 1),1);   % 缺失值数量

A = Y(index == 1) - P(index == 1);  % 求差
MAE = sum(abs(A)) / miss_num;
RMSE = sqrt(A'*A / miss_num);
MRE = sum(abs(A)) / sum(Y(index == 1));

fprintf('MAE: %6.4f\n, MRE: %6.4f\n, RMSE: %6.4f\n', MAE, MRE, RMSE);