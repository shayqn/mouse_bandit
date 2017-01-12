%% prepro_csv across days for 1 mouse

mouse_name = input('Mouse Name: ','s');
root_dir = uigetdir;
file_names = dir(root_dir);
num_files = size(dir,1);


for i = 1:num_files
    
    file_name = file_names(i).name;
    
    if file_name(1) ~= '.'
        cd(file_name);
        
        date = file_name;
        
        try
            prepro_csv(mouse_name,date,cd);
        catch ME
            warning(strcat('Error on day: ',date))
            disp(ME)
        end
    end
    
    cd(root_dir);
end