# ac209a_project
Class project for ac209a

The goal of our model is to predict the future decision of the mouse, i.e., left port or right port poke, on a given trial, based on previously observed events and trials.

### Description of features:

Trial data refers here to the outcome of a previously observed trial or trials. For example, did the animal go left or right and did it get a reward or not?

Events data refers to the inter-trial interval (ITI), which is the time it takes the mouse to self-initiate a new trial from when it last received a reward.  A second event type is the decision time, which refers to the time in which the mouse self-initiated a trial to when it reported its decision, i.e., poked to the right or to the left.

The data from the previously observed trials and events will be pre processed in a way to allow them to fit into one row of predictors. These predictors are separated into 4 data types.

Data type 1 and 2 account for the same number of previous trials but relate to different granularities of that data. More specifically, the first data type is the summary data from i previous trials and contain 4 variables: How many times did the mouse go to the left, How many times did the mouse go to the right, and how many times was it rewarded on each side. Data type 2 is the streak data from i previous trials. This data will contain 1 variable and is the most recent streak of either successes or failures in the i previous trials.

Data type 3 accounts for fewer previous trials then data type 1 and 2 but adds event information to each trial. For each trial out of p trials (as noted p<i) the data will be: side of decision, rewarded or not, ITI, and decision time, i.e, 4 variables.

Data type 4 is the events from trial we wish to predict its outcome, thus contains 2 variables.
