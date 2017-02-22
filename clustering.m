function clustering(n1, n2, n3, filename)
% Uses kmeans clustering to group 3 bots into different data sets. Trains 3
% different transition matrices on the 3 data sets using Baum Welch.
% Outputs csv file with filename containing guesses of 100th step of each
% run.
% Inputs - n1 = # rows bot 1 trains on, etc.

% Use clustering(30, 10, 10, 'kaggle.csv')
% 	  This is how we got our best kaggle submission, but we ran it a few
% 	  times and selected what we thought was best based on labels


% Read in files
obs = csvread('observations.csv');
labels = csvread('labels.csv');

% Find all possible distance emissions
dist = [];
fullDist = []; % contains duplicates
for r=0:4
    for c=1:5
        d = round(sqrt(r^2 + c^2),4); % Round to match csv files exactly
        fullDist = [fullDist d];
        if ~ismember(d, dist)
            dist = [dist d];
        end
    end
end

% ------------------------ k means clustering ---------------------
X = zeros(3000,length(dist));
for i=1:3000
    for j=1:100
        temp = find(dist==obs(i,j));% distance to index mapping
        X(i,temp) = X(i,temp) + 1;
    end
end

centroids = [0.2973 0.5814 0.7716 0.5714 18.9859 0.2051 18.7359  0.8563  0.6678 0 18.7243 19.9477  0.8488 0 18.6246 0 0.1819 0;
             0.0973 0.2064 0.2534 0.2223 12.2458 0.1074 12.1871 12.3733 12.0612 0 12.3045 13.1854 12.3372 0 12.3012 0 0.1174 0;
             0.2036 0.3924 0.4934 0.3659  0.6623 0.2268  0.4967 31.5579  0.3659 0  0.2599 32.5199  0.5232 0 31.7202 0 0.2119 0];
[idx, C] = kmeans(X,3,'Start',centroids);

% Group bots into 3 data sets, store original index in obs
bot1 = [];
bot1idx = []; 
bot2 = [];
bot2idx = [];
bot3 = [];
bot3idx = [];
for k=1:length(idx)
    if idx(k) == 1
        bot1 = [bot1; obs(k,:)];
        bot1idx = [bot1idx k];
    elseif idx(k) == 2
        bot2 = [bot2; obs(k,:)];
        bot2idx = [bot2idx k];
    else
        bot3 = [bot3; obs(k,:)];
        bot3idx = [bot3idx k];
    end
end

% --------- Initializing transition and emission matrices ---------
% Initialize transition matrix. More likely to jump to neighbors:
transition = zeros(25,25);
for i = 1:25
    if rem(i,5) ~= 0 % not last column (5,10,15,20,25)
        transition(i,i+1) = .25;
    elseif i~=5 && i~=25
        transition(i,i-1) = 1/3;
        transition(i,i-5) = 1/3;
        transition(i,i+5) = 1/3;
        continue
    end
    if rem(i,5) ~= 1% not in 1st column (1,6,11,16,21)
        transition(i,i-1) = .25;
    elseif i~=1 && i~=21
        transition(i,i+1) = 1/3;
        transition(i,i-5) = 1/3;
        transition(i,i+5) = 1/3;
        continue
    end
    if ~ismember(i,[1 2 3 4 5]) % not in 1st row (1,2,3,4,5)
        transition(i,i-5) = .25;
    elseif i~=1 && i~=5
        transition(i,i-1) = 1/3;
        transition(i,i+1) = 1/3;
        transition(i,i+5) = 1/3;
        continue
    end
    if ~ismember(i,[21 22 23 24 25]) % not in last row (21,22,23,24,25)
        transition(i,i+5) = .25;
    elseif i~=21 && i~=25
        transition(i,i-1) = 1/3;
        transition(i,i+1) = 1/3;
        transition(i,i-5) = 1/3;
        continue
    end
end
%Set corners
transition(1,2)=.5;
transition(1,6)=.5;
transition(21,16)=.5;
transition(21,22)=.5;
transition(25,24)=.5;
transition(25,20)=.5;
transition(5,4)=.5;
transition(5,10)=.5;

% Initialize EMITGUESS matrix. States that will emit given 
%     1.0000    2.0000    3.0000    4.0000    5.0000
%     1.4142    2.2361    3.1623    4.1231    5.0990
%     2.2361    2.8284    3.6056    4.4721    5.3852
%     3.1623    3.6056    4.2426    5.0000    5.8310
%     4.1231    4.4721    5.0000    5.6569    6.4031
emission = zeros(25,length(dist));
for i = 1:25
    d = fullDist(i);
    k = find(dist==d,1);
    emission(i,k) = 1;
end


% --------- Train transition and emission matrices ---------
n1 = min(n1, length(bot1));
n2 = min(n2, length(bot2));
n3 = min(n3, length(bot3));

% bot1
seq = bot1(1:n1,:);
[ESTTR,ESTEMIT] = hmmtrain(seq,transition,rand(25,18),'Symbols',dist);
ESTTR = ESTTR + .0001;

for k=1:length(bot1)
    STATES = hmmviterbi(bot1(k,:),ESTTR,emission,'Symbols',dist);
    temp = bot1idx(k); %index into observation
    OUTPUT(temp,1) = temp; 
    OUTPUT(temp,2) = STATES(100);
    if temp<=200
        fprintf('%d %d, bot#%d\n',labels(temp), OUTPUT(temp,2), 1)
    end
end

% bot2
seq = bot2(1:n2,:);
[ESTTR,ESTEMIT] = hmmtrain(seq,transition,rand(25,18),'Symbols',dist);
ESTTR = ESTTR + .0001;

for k=1:length(bot2)
    STATES = hmmviterbi(bot2(k,:),ESTTR,emission,'Symbols',dist);
    temp = bot2idx(k); %index into observation
    OUTPUT(temp,1) = temp; 
    OUTPUT(temp,2) = STATES(100);
    if temp<=200
        fprintf('%d %d, bot#%d\n',labels(temp), OUTPUT(temp,2), 2)
    end
end

% bot3
seq = bot3(1:n3,:);
[ESTTR,ESTEMIT] = hmmtrain(seq,transition,rand(25,18),'Symbols',dist);
ESTTR = ESTTR + .0001;

for k=1:length(bot3)
    STATES = hmmviterbi(bot3(k,:),ESTTR,emission,'Symbols',dist);
    temp = bot3idx(k); %index into observation
    OUTPUT(temp,1) = temp;
    OUTPUT(temp,2) = STATES(100);
    if temp<=200
        fprintf('%d %d, bot#%d\n',labels(temp), OUTPUT(temp,2), 3)
    end
end

M = OUTPUT(201:3000,1:2);
M(:,1) = M(:,1) - 200;

csvwrite(filename,M);
