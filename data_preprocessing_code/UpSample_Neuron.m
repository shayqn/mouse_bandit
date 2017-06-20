%% Upsample
up_rate = 5;

n = size(neuron.C,1); % number of neurons
t_up = size(neuron.C,2)*up_rate;

% Copy into temp variables
C = neuron.C;
S = neuron.S;
C_raw = neuron.C_raw;

% Initialize upsampled variables
C_up = zeros(n,t_up);
S_up = zeros(n,t_up);
C_raw_up = zeros(n,t_up);


for i = 1:n
    C_up(i,:) = interp(C(i,:),up_rate);d
    S_up(i,:) = interp(S(i,:),up_rate);
    C_raw_up(i,:) = interp(C_raw(i,:),up_rate);
end

%% assign upsampled variables to neuron class

neuron_upsampled = neuron.copy();

neuron_upsampled.C = C_up;
neuron_upsampled.C_raw = C_raw_up;
neuron_upsampled.S = S_up;

%% save upsampled variable

save_dir = uigetdir;
cd(save_dir);
save('neuron_upsampled.mat','neuron_upsampled');
