function compPerformance(Tasks,PGPol,learningRate,trajLengthTest,numRolloutsTest,numIterationsTest,modelCrossDomainPGELLA,modelPGELLA,nGroupDomain)
disp([' ']);
disp(['Generating the plots of the average rewards of Cross-Domain-PG-ELLA, PG-ELLA and PG. There are the following options']);
disp(['1 Loading previously calculated data']);
disp([' ']);
disp(['2 Executing the code to evaluate the average reward of all three algorithms.']);
disp(['(This option can take several minutes to be completed)']);
disp([' ']);
prompt = 'Please type the number of one of the options indicated above ';
inp = input(prompt);
if (inp~=1 && inp~=2)
    disp(['Invalid option, proceeding to load data from .mat file to generate plot...'])
    load('Data.mat','Toplot_rCrossDomainPGELLA','Toplot_rPGELLA','Toplot_rPG')
elseif inp == 2
    for i = 1:size(Tasks,2) % loop over all tasks
        % Take the task and determine to which group it belongs 
        G_Task = Tasks(i).param.Group;
        Groups(i) = G_Task;
        % Building testing policies for Cross-Domain PG-ELLA
        Psi = modelCrossDomainPGELLA(G_Task).Proj.Psi;
        theta_Task = Psi*modelCrossDomainPGELLA(1).L*modelCrossDomainPGELLA(1).S(:,i);
        policy.theta = theta_Task;
        policy.sigma = PGPol(i).policy.sigma;
        PolicyCrossDomainPGELLA(i).policy = policy;

        % Building testing policies for PG-ELLA
        theta_PG_ELLA = modelPGELLA(G_Task).L*modelPGELLA(1).S(:,i);
        policyPGELLA.theta = theta_PG_ELLA;
        policyPGELLA.sigma = PGPol(i).policy.sigma;
        PolicyPGELLAGroup(i).policy = policyPGELLA;
    end

    nGroups = max(Groups);

    % Test Cross-Domain PG-ELLA
    %--------------------------------------------------------------------------

    for k = 1:size(Tasks,2) % Test over all tasks 

        disp(['Testing task ',num2str(k),': ',Tasks(k).param.ProblemID,'...',])
        h = waitbar(0,['Testing task ',num2str(k),': ',Tasks(k).param.ProblemID]);

        for m = 1:numIterationsTest % Loop over Iterations
            %------------------------------------------------
            % PG
            [data] = obtainData(PGPol(k).policy,trajLengthTest,numRolloutsTest,Tasks(k));

            dJdTheta = episodicNAC(PGPol(k).policy,data,Tasks(k));

            % Update Policy Parameters
            PGPol(k).policy.theta = PGPol(k).policy.theta ...
            +learningRate*dJdTheta;
            %------------------------------------------------

            %------------------------------------------------
            % PG-ELLA
            [dataPGELLA] = obtainData(PolicyPGELLAGroup(k).policy,trajLengthTest,numRolloutsTest,Tasks(k));

            dJdThetaPGELLA = episodicNAC(PolicyPGELLAGroup(k).policy,dataPGELLA,Tasks(k));

            % Update Policy Parameters
            PolicyPGELLAGroup(k).policy.theta = PolicyPGELLAGroup(k).policy.theta ...
            +learningRate*dJdThetaPGELLA;
            %------------------------------------------------

            %------------------------------------------------
            % Cross-Domain PG ELLA
            [dataCrossDomainPGELLA] = obtainData(PolicyCrossDomainPGELLA(k).policy,trajLengthTest,numRolloutsTest,Tasks(k));

            dJdThetaCrossDomainPGELLA = episodicNAC(PolicyCrossDomainPGELLA(k).policy,dataCrossDomainPGELLA,Tasks(k));

            % Update Policy Parameters
            PolicyCrossDomainPGELLA(k).policy.theta = PolicyCrossDomainPGELLA(k).policy.theta ...
            + learningRate*dJdThetaCrossDomainPGELLA; 
            %------------------------------------------------


            % Computing average reward in one system per iteration

            for z = 1:size(dataCrossDomainPGELLA,2)
                Sum_rCrossDomainPGELLA(z,:) = sum(dataCrossDomainPGELLA(z).r);
            end

            for z = 1:size(dataPGELLA,2)
                Sum_rPGELLA(z,:) = sum(dataPGELLA(z).r);
            end

            for z = 1:size(data,2)
                Sum_rPG(z,:) = sum(data(z).r);
            end

            Avg_rCrossDomainPGELLA(m,k) = mean(Sum_rCrossDomainPGELLA);
            Avg_rPGELLA(m,k) = mean(Sum_rPGELLA);
            Avg_rPG(m,k) = mean(Sum_rPG);

            waitbar(m / numIterationsTest)
        end
        close(h) 
    end

    % Plotting Phase
    n = 1;
    for i = 1:nGroups
        Toplot_rCrossDomainPGELLA(:,i) = mean(Avg_rCrossDomainPGELLA(:,n:n + nGroupDomain(i)-1),2); % Average over each column
        n = n + nGroupDomain(i);
    end

    n = 1;
    for i = 1:nGroups
        Toplot_rPGELLA(:,i) = mean(Avg_rPGELLA(:,n:n + nGroupDomain(i)-1),2); % Average over each column
        n = n + nGroupDomain(i);
    end

    n = 1;
    for i = 1:nGroups
        Toplot_rPG(:,i) = mean(Avg_rPG(:,n:n + nGroupDomain(i)-1),2); % Average over each column
        n = n + nGroupDomain(i);
    end

elseif inp == 1
    load('Data.mat','Toplot_rCrossDomainPGELLA','Toplot_rPGELLA','Toplot_rPG')
end

rCrossDomainPGELLAvec1 = Toplot_rCrossDomainPGELLA(:,1);
rPGELLAvec1 = Toplot_rPGELLA(:,1);
rPGvec1 = Toplot_rPG(:,1);

rCrossDomainPGELLAvec2 = Toplot_rCrossDomainPGELLA(:,2);
rPGELLAvec2 = Toplot_rPGELLA(:,2);
rPGvec2 = Toplot_rPG(:,2);

barnumELLA = numIterationsTest-1;
barnumPGELLA = numIterationsTest-1;
barnumPG = numIterationsTest-1;

% Plotting average reward for domain SM

% Create figure
figure1 = figure;
set(figure1,'Position',[30 30 900 600])
% Create axes
axes1 = axes('Parent',figure1,'FontSize',27);
xlim(axes1,[0 numIterationsTest]);
box(axes1,'on');
hold(axes1,'on');
plot(1:round((size(rCrossDomainPGELLAvec1,1)-1)/barnumELLA):size(rCrossDomainPGELLAvec1,1),...
    mean(rCrossDomainPGELLAvec1(1:round((size(rCrossDomainPGELLAvec1,1)-1)/barnumELLA):size(rCrossDomainPGELLAvec1,1),:),2),...
    'DisplayName','Cross-Domain','MarkerSize',10,...
    'Marker','square',...
    'LineWidth',2,...
    'Color',[1 0 0])
hold on
plot(1:round((size(rPGELLAvec1,1)-1)/barnumPGELLA):size(rPGELLAvec1,1),...
    mean(rPGELLAvec1(1:round((size(rPGELLAvec1,1)-1)/barnumPGELLA):size(rPGELLAvec1,1),:),2),...
    'DisplayName','PG-ELLA','MarkerSize',8,...
    'Marker','diamond',...
    'LineWidth',2,...
    'Color',[0 .7 0])
plot(1:round((size(rPGvec1,1)-1)/barnumPG):size(rPGvec1,1),...
    mean(rPGvec1(1:round((size(rPGvec1,1)-1)/barnumPG):size(rPGvec1,1),:),2),...
    'DisplayName','Standard PG','MarkerSize',8,...
    'Marker','o',...
    'LineWidth',2,...
    'Color',[0 0 1])

% Create ylabel
ylabel('Average Reward','FontSize',27);

% Create xlabel
xlabel('Iterations','FontSize',27);

% Creaye title
title('Average reward for domain SM')

% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.603802148841547 0.486726660930808 0.289427076562101 0.221776583168347],...
    'FontSize',28);


% Plotting average reward for domain DM

% Create figure
figure2 = figure;
set(figure2,'Position',[30 30 900 600])
% Create axes
axes1 = axes('Parent',figure2,'FontSize',27);
xlim(axes1,[0 numIterationsTest]);
box(axes1,'on');
hold(axes1,'on');
plot(1:round((size(rCrossDomainPGELLAvec2,1)-1)/barnumELLA):size(rCrossDomainPGELLAvec2,1),...
    mean(rCrossDomainPGELLAvec2(1:round((size(rCrossDomainPGELLAvec2,1)-1)/barnumELLA):size(rCrossDomainPGELLAvec2,1),:),2),...
    'DisplayName','Cross-Domain','MarkerSize',10,...
    'Marker','square',...
    'LineWidth',2,...
    'Color',[1 0 0])
hold on
plot(1:round((size(rPGELLAvec2,1)-1)/barnumPGELLA):size(rPGELLAvec2,1),...
    mean(rPGELLAvec2(1:round((size(rPGELLAvec2,1)-1)/barnumPGELLA):size(rPGELLAvec2,1),:),2),...
    'DisplayName','PG-ELLA','MarkerSize',8,...
    'Marker','diamond',...
    'LineWidth',2,...
    'Color',[0 .7 0])
plot(1:round((size(rPGvec2,1)-1)/barnumPG):size(rPGvec2,1),...
    mean(rPGvec2(1:round((size(rPGvec2,1)-1)/barnumPG):size(rPGvec2,1),:),2),...
    'DisplayName','Standard PG','MarkerSize',8,...
    'Marker','o',...
    'LineWidth',2,...
    'Color',[0 0 1])

% Create ylabel
ylabel('Average Reward','FontSize',27);

% Create xlabel
xlabel('Iterations','FontSize',27);

% Creaye title
title('Average reward for domain DM')

% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.603802148841547 0.486726660930808 0.289427076562101 0.221776583168347],...
    'FontSize',28);