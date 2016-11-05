function trials = extractTrials(stats,pokeHistory)
% trials = extractTrials(stats,pokeHistory)
% This function takes as its inputs: stats, pokeHistory
% And returns the matrix: trials
% 'trials' is a matrix where each row represents a decision poke in a
% succesful trial. It has 4 columns which contain the following
% information: 
    % 1: time
    % 2: port (1 = right, 2 = left)
    % 3: right port reward probability
    % 4: left port reward probability
    % 5: reward given (1 / 0)
%Note: extractTrials was changed on 08/03/16 to include the probabilities
%of both ports for each trial. Previously there was only 4 columns, with
%the 3rd being the reward probability of the port poked and the 4th the
%reward boolean. 

% create a vector 'timePoked' where each entry represents a poke, and the
% value is the number of seconds since the first poke. This is also
% calculated in pokeAnalysis_v1, but I wanted to add it here so this can
% stand as an independent function. 
numPokes = length(pokeHistory);
timePoked = zeros(1,numPokes);
firstPoke = datevec(pokeHistory(1).timeStamp);
for i = 1:numPokes
    timePoked(i) = etime(datevec(pokeHistory(i).timeStamp),firstPoke);
end

%% calculate total number of trials
numLeftTrials = sum(stats.trials.left == 2);
numRightTrials = sum(stats.trials.right == 2);
numTrials = numLeftTrials + numRightTrials;


%% determine which actual pokes were trials
decisionPokes = find((stats.trials.left == 2) + (stats.trials.right == 2));
trialStartPokes = decisionPokes-1;
decisionTimes = timePoked(decisionPokes);
trialStartTimes = timePoked(trialStartPokes);

leftTrialIndices = find(stats.trials.left == 2);
rightTrialIndices = find(stats.trials.right == 2);

leftTrialTimes = timePoked(leftTrialIndices);
rightTrialTimes = timePoked(rightTrialIndices);
allTrialIndices = sort([leftTrialIndices,rightTrialIndices]);


%% create trials matrix
%trials is going to be a numTrials by 4 matrix where the columns are as
%following:
    % 1. decision poke time (since start of session)
    % 2: time since last trial
    % 3. time between trial initiation and decision
    % 4: port (1 = right, 2 = left)
    % 5: right port reward probability
    % 6: left port reward probability
    % 7: reward given (1 / 0)
trials = zeros(numTrials,4);

% time since last trial
trials(:,1) = decisionTimes;

% time since last trial
if (trialStartPokes(1) == 1)
    trials(1,2) = 0;
    trials(2:end,2) = trialStartTimes(2:end) - timePoked(trialStartPokes(2:end)-1);
else
    trials(:,2) = trialStartTimes - timePoked(trialStartPokes-1);
end

% trial length (between center poke and decision poke)
trials(:,3) = decisionTimes - trialStartTimes;


% trial port
lefttrials = ismember(decisionTimes,leftTrialTimes);
righttrials = ismember(decisionTimes,rightTrialTimes);
trials(:,4) = righttrials' + (lefttrials.*2)';

% reward probabilities
% don't see a way to vectorize this :( will loop through:

%all trials
allTrialPokes = pokeHistory(allTrialIndices);

for i = 1:numTrials
    trials(i,5) = allTrialPokes(i).rightPortStats.prob;
    trials(i,6) = allTrialPokes(i).leftPortStats.prob;
end

%finally, the rewards:
leftrewards = stats.rewards.left(leftTrialIndices);
rightrewards = stats.rewards.right(rightTrialIndices);

trials(righttrials,7) = rightrewards;
trials(lefttrials,7) = leftrewards;
