% read data
fname = '.\Iris_data.xls';
[Feat,Labels] = xlsread(fname)

Feat_N = size(Labels)

t = [];
for i = 2:Feat_N(1)
    label = Labels(i,5);
    label = label{1};
   
    if strcmp(label,'SETOSA')
        t = [t, 1];
    elseif strcmp(label,'VERSICOL')
        t = [t, 2];        
    elseif strcmp(label,'VIRGINIC')
        t = [t, 3];        
    end
end
%%   sub-problem1 - First 2 features with linear kernel

Two_Feat = Feat(:,1:2);

% Use one versus the rest, training on 3 classifiers
alpha_lst = [];
w_lst = [];
bias_lst = [];
supp_vec = false(1,150);

for cls = 1:3
    
    % Kernel-func
    Gram = zeros(length(t),length(t));
    for i = 1:length(t)
        for j = 1:length(t)
            Gram(i,j) = Two_Feat(i,:) * Two_Feat(j,:).';        
        end
    end

    % Create Labels
    YY = double(t == cls);
    YY(YY==0) = -1
    
    % Extract multipliers alphas
    [alpha,bias] = smo(Gram, YY, 1000, 0.001);

    
    % w using labels YY
    w = (alpha .* YY) * Two_Feat;
    
    % Support Vector
    supp_vec = supp_vec | alpha > 0;
    
    %append
    alpha_lst = [alpha_lst; alpha];
    bias_lst = [bias_lst; bias];
    w_lst = [w_lst; w];
end

% start testing
% augment data to draw more smooth image
[Feat_1,Feat_2] = meshgrid(min(Two_Feat(:,1)):0.01:max(Two_Feat(:,1)),min(Two_Feat(:,2)):0.01:max(Two_Feat(:,2)));
xTest = [ Feat_1(:),Feat_2(:) ];
yTest = w_lst * xTest.' + repmat(bias_lst,[1 length(xTest)]);
[~,Cls_result_test] = max(yTest);

% testing on the train set
yTrain = w_lst * Two_Feat.' + repmat(bias_lst,[1 150]);
[~,Cls_result_train] = max(yTrain);
c1Train = Two_Feat(Cls_result_train==1,:)
c2Train = Two_Feat(Cls_result_train==2,:)
c3Train = Two_Feat(Cls_result_train==3,:)
svTrain = Two_Feat(supp_vec,:);

% Plot
figure(1)
gscatter(xTest(:,1),xTest(:,2),Cls_result_test);
hold on;
scatter(c1Train(:,1),c1Train(:,2),'+');
scatter(c2Train(:,1),c2Train(:,2),'*');
scatter(c3Train(:,1),c3Train(:,2),'x');
scatter(svTrain(:,1),svTrain(:,2),'ko');
legend();
ylabel('W');
xlabel('L');
hold off;

%
%%   sub-problem2 - First 2 features with degree kernel
Two_Feat = Feat(:,1:2);
Phi_tr = [Two_Feat(:,1).^2, 2.^0.5 * Two_Feat(:,1).* Two_Feat(:,2), Two_Feat(:,2).^2];     % Create Phi

% Use one versus the rest, training on 3 classifiers
alpha_lst = [];
w_lst = [];
bias_lst = [];
supp_vec = false(1,150);

for cls = 1:3
    
    % Kernel

    Gram = zeros(length(t),length(t));
    for i = 1:length(t)
        for j = 1:length(t)
            Gram(i,j) = (Two_Feat(i,:) * Two_Feat(j,:).') .^2;        
        end
    end

    % Labels
    YY = double(t == cls);
    YY(YY==0) = -1
    
     % Extract multipliers alphas
    [alpha,bias] = smo(Gram, YY, 1000, 0.001);

    % w using labels YY
    w = (alpha .* YY) * Phi_tr;
    
    % Support Vector
    supp_vec = supp_vec | alpha > 0;
    
    %append
    alpha_lst = [alpha_lst; alpha];
    bias_lst = [bias_lst; bias];
    w_lst = [w_lst; w];
end

% start testing
% augment data to draw more smooth image
[Feat_1,Feat_2] = meshgrid(min(Two_Feat(:,1)):0.01:max(Two_Feat(:,1)),min(Two_Feat(:,2)):0.01:max(Two_Feat(:,2)));

xTest = [ Feat_1(:),Feat_2(:) ];
Phi_ts = [xTest(:,1).^2, 2.^0.5 * xTest(:,1).* xTest(:,2), xTest(:,2).^2]; % Create Phi 
yTest = w_lst * Phi_ts.' + repmat(bias_lst,[1 length(Phi_ts)]);

[~,Cls_result_test] = max(yTest);

% testing on the train set
yTrain = w_lst * Phi_tr.' + repmat(bias_lst,[1 150]);
[~,Cls_result_train] = max(yTrain);
c1Train = Two_Feat(Cls_result_train==1,:)
c2Train = Two_Feat(Cls_result_train==2,:)
c3Train = Two_Feat(Cls_result_train==3,:)
svTrain = Two_Feat(supp_vec,:);

% Plot
figure(2)
gscatter(xTest(:,1),xTest(:,2),Cls_result_test);
hold on;
scatter(c1Train(:,1),c1Train(:,2),'+');
scatter(c2Train(:,1),c2Train(:,2),'*');
scatter(c3Train(:,1),c3Train(:,2),'x');
scatter(svTrain(:,1),svTrain(:,2),'ko');
legend();
ylabel('W');
xlabel('L');
hold off;
%%  LDA
mu = mean(Feat);   
Sw_mat = zeros(4,4);
Sb_mat = zeros(4,4);
for cls = 1:3
    
    xk = Feat(t==cls,:);
    mk = mean(xk);
    
    Sk = zeros(4,4);
    for i = 1:length(xk)
        Sk = Sk + (xk(i,:)-mk).' * (xk(i,:)-mk) ;
    end    
    
    Sw_mat = Sw_mat + Sk;
    
    % Calculate Sb_mat
    
    Sb_mat = Sb_mat + length(xk) * ((mk-mu).' * (mk-mu) );
    
end    

% Find LDA matrix

 [ev,lambda] = eig(inv(Sw_mat) * Sb_mat);
 IDA_mat = ev(:,1:2);
 
 %%  sub-problem3 - First 2 features with linear kernel
 
Two_Feat = Feat * IDA_mat;

% Use one versus the rest, training on 3 classifiers
alpha_lst = [];
w_lst = [];
bias_lst = [];
supp_vec = false(1,150);

for cls = 1:3
    
    % Kernel

    Gram = zeros(length(t),length(t));
    for i = 1:length(t)
        for j = 1:length(t)
            Gram(i,j) = Two_Feat(i,:) * Two_Feat(j,:).';        
        end
    end

    % Labels
    YY = double(t == cls);
    YY(YY==0) = -1
    
    
     % Extract multipliers alphas
    [alpha,bias] = smo(Gram, YY, 1000, 0.001);

    % w using labels YY
    w = (alpha .* YY) * Two_Feat;
    
    % Support Vector
    supp_vec = supp_vec | alpha > 0;
    
    %append
    alpha_lst = [alpha_lst; alpha];
    bias_lst = [bias_lst; bias];
    w_lst = [w_lst; w];
    
end

% start testing
% augment data to draw more smooth image
[Feat_1,Feat_2] = meshgrid(min(Two_Feat(:,1)):0.01:max(Two_Feat(:,1)),min(Two_Feat(:,2)):0.01:max(Two_Feat(:,2)));
xTest = [ Feat_1(:),Feat_2(:) ];
yTest = w_lst * xTest.' + repmat(bias_lst,[1 length(xTest)]);
[~,Cls_result_test] = max(yTest);

% testing on the train set
yTrain = w_lst * Two_Feat.' + repmat(bias_lst,[1 150]);
[~,Cls_result_train] = max(yTrain);
c1Train = Two_Feat(Cls_result_train==1,:)
c2Train = Two_Feat(Cls_result_train==2,:)
c3Train = Two_Feat(Cls_result_train==3,:)
svTrain = Two_Feat(supp_vec,:);

% Plot
figure(3)
gscatter(xTest(:,1),xTest(:,2),Cls_result_test);
hold on;
scatter(c1Train(:,1),c1Train(:,2),'+');
scatter(c2Train(:,1),c2Train(:,2),'*');
scatter(c3Train(:,1),c3Train(:,2),'x');
scatter(svTrain(:,1),svTrain(:,2),'ko');
legend();
ylabel('W');
xlabel('L');
hold off;

 %%  sub-problem4 - First 2 features with degree kernel
 
Two_Feat = Feat * IDA_mat;
Phi_tr = [Two_Feat(:,1).^2, 2.^0.5 * Two_Feat(:,1).* Two_Feat(:,2), Two_Feat(:,2).^2];     % Create Phi

% Use one versus the rest, training on 3 classifiers
alpha_lst = [];
w_lst = [];
bias_lst = [];
supp_vec = false(1,150);

for cls = 1:3
    
    % Kernel
    Gram = zeros(length(t),length(t));
    for i = 1:length(t)
        for j = 1:length(t)
            Gram(i,j) = (Two_Feat(i,:) * Two_Feat(j,:).') .^2;        
        end
    end
    
    
    % Labels
    YY = double(t == cls);
    YY(YY==0) = -1
    
     % Extract multipliers alphas
    [alpha,bias] = smo(Gram, YY, 1000, 0.001);

    % w using labels YY
    w = (alpha .* YY) * Phi_tr;
    
    % Support Vector
    supp_vec = supp_vec | alpha > 0;
    
    %append
    alpha_lst = [alpha_lst; alpha];
    bias_lst = [bias_lst; bias];
    w_lst = [w_lst; w];

end

% start testing
% augment data to draw more smooth image
[Feat_1,Feat_2] = meshgrid(min(Two_Feat(:,1)):0.01:max(Two_Feat(:,1)),min(Two_Feat(:,2)):0.01:max(Two_Feat(:,2)));

xTest = [ Feat_1(:),Feat_2(:) ];
Phi_ts = [xTest(:,1).^2, 2.^0.5 * xTest(:,1).* xTest(:,2), xTest(:,2).^2]; % Create Phi 
yTest = w_lst * Phi_ts.' + repmat(bias_lst,[1 length(Phi_ts)]);

[~,Cls_result_test] = max(yTest);

% testing on the train set
yTrain = w_lst * Phi_tr.' + repmat(bias_lst,[1 150]);
[~,Cls_result_train] = max(yTrain);
c1Train = Two_Feat(Cls_result_train==1,:)
c2Train = Two_Feat(Cls_result_train==2,:)
c3Train = Two_Feat(Cls_result_train==3,:)
svTrain = Two_Feat(supp_vec,:);

% Plot
figure(4)
gscatter(xTest(:,1),xTest(:,2),Cls_result_test);
hold on;
scatter(c1Train(:,1),c1Train(:,2),'+');
scatter(c2Train(:,1),c2Train(:,2),'*');
scatter(c3Train(:,1),c3Train(:,2),'x');
scatter(svTrain(:,1),svTrain(:,2),'ko');
legend();
ylabel('W');
xlabel('L');
hold off;