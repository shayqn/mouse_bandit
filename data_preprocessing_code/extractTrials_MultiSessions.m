d
%%
dates = {'11222016','11302016','12022016','12052016','12062016'};
mouseName = 'q45';
saveDir = uigetdir;

rootDir = uigetdir;
cd(rootDir)

dirStruct = dir(rootDir);
numFolders = size(dirStruct,1);

%% cycle through the folders
for date = 1:5
    cd(dates{date})
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
        trial_name = strcat(saveDir,'/',mouseName,'_',dates{date},'.csv');
        csvwrite(trial_name,trials)
        cd(rootDir);
    end
end