function prepro_csv(mouse_name,date,folder)

%%
% 1. Loads in stats, pokeHistory, parameters
% 2. Extract trials
% 3. Saves csv files of trials and parameters


cd(folder)
matFiles = dir('*.mat');
%if the directory has this type of '.mat' file(s) the function will
%execute the following code
if size(matFiles,1) ~= 0
    %loads the stats and pokeHistory
    for i = 1:size(matFiles,1)
        load(matFiles(i).name);
    end
else
    error('No mat files in folder')
end


%extracts from the stats and pokeHistory
trials = extractTrials(stats,pokeHistory);


trials_filename = strcat(date,'_',mouse_name,'_trials','.csv');
p_filename = strcat(date,'_',mouse_name,'_parameters','.csv');

csvwrite(trials_filename,trials);
writetable(struct2table(p),p_filename);

end
