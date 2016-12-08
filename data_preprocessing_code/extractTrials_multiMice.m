%% Extract trials from all mice for 1 day
% This script simply cycles through all mouse folders in
% a fiven day, loads in pokeHistory and
% stats for each mouse, on each day, extracts
% the trials, and saves it  as 'trials.mat' in each folder.

%% Specify directory
%sets dataDir as the directory where the function will be looked as
%specified by the user
rootDir = uigetdir;
%change the directory to the one chosen by the user
cd(rootDir);
%total number of files (including unwanted files) in dataDir
dirStruct = dir(rootDir);
numFolders = size(dirStruct,1);

%% cycle through the folders
for currFolder = 1:numFolders
    %only cd into a folder that doesn't start with '.' otherwise it brings
    %you out of the root folder, at which point you won't be able to cd
    %into the mouse folders.
    if ~strcmpi(dirStruct(currFolder).name(1),'.')
        cd(dirStruct(currFolder).name)
        matFiles = dir('*.mat');
        %if the directory has this type of '.mat' file(s) the function will
        %execute the following code
        if size(matFiles,1) ~= 0
            %loads the stats and pokeHistory
            for i = 1:size(matFiles,1)
                load(matFiles(i).name);
            end
            %extracts from the stats and pokeHistory
            trials = extractTrials(stats,pokeHistory);
            filename = strcat(dirStruct(currFolder).name,'_',num2str(currFolder),'_','.csv');
            csvwrite(filename,trials);
            cd(rootDir);
        end
    end
end