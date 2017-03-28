# Mouse Bandit

# Summary
This is an ongoing attempt to model the behavior of a mouse as it performs a version of the two-armed bandit task. 


# Background
This behavioral task is inspired from the 'multi-armed bandit problem', a scenario first formalized in probability theory to illustrate the concept of balancing the attempt to acquire new information about the world with the desire to optimize your actions based on existing knowledge. 

For example, you are at a restaurant, staring at the menu. How do you decide what to get? Do you pick your favorite dish – one you know will bring satisfaction? Or do you decide to take a risk and try something new? After all, maybe you will discover your new favorite food! 

Whether or not you try that new dish may not be so important, but the general ability to strategize when ‘taking a chance’ is better than trusting your previous experiences – that is absolutely critical to successfully navigating life. This decision process is known as the ‘explore vs exploit’ problem. In the 1950s, (some) mathematicians became obsessed with trying to find the objectively optimal strategy – how often should you explore? When should you exploit? Is there an exact solution? To a large extent, the field of mathematics and machine learning have converged on several theorems that promise to optimize this decision process in a wide variety of settings. In fact, computer scientists often use these theorems to instruct artificial intelligence algorithms how to learn and behave in various complex situations. 

The most popular formalization of the explore-exploit problem is the multi-armed bandit task, where a gambler in front of a row of slot machines (often aptly referred to as 'one-armed bandits') must decide which machines to play, and in what order, in an effort to maximize earnings over a set period of time. 

We and others have adapted this task for mice, where a mouse must choose to poke a 'left' or 'right' port for a water reward. The two ports have different probabilites of paying out water, and these probabilities change unpredictably over time. Therefore, based on the current outcomes of its actions, the mouse must flexibly decide how to bias its behavior to maximize the amount of water earned. 


# The Model
We are currently trying a few different models. First, we are trying to predict when the mouse 'switches' - that is, when the mouse chooses a different port than its immediate preceeding choice. Due to the structure of the behavior, where the reward probabilities are switched ~4-6 times per behavior session, a mouse switches its behavior only ~10% of the time. 

Currently, we are exploring using logistic regression, linear discriminant analysis, and decision trees to model when the mouse switches its decision.

### Description of features:

Trial data refers here to the outcome of a previously observed trial or trials. For example, did the animal go left or right and did it get a reward or not?

Events data refers to the inter-trial interval (ITI), which is the time it takes the mouse to self-initiate a new trial from when it last received a reward.  A second event type is the decision time, which refers to the time in which the mouse self-initiated a trial to when it reported its decision, i.e., poked to the right or to the left.

The data from the previously observed trials and events will be pre processed in a way to allow them to fit into one row of predictors. These predictors are separated into 4 data types.

Data type 1 and 2 account for the same number of previous trials but relate to different granularities of that data. More specifically, the first data type is the summary data from i previous trials and contain 4 variables: How many times did the mouse go to the left, How many times did the mouse go to the right, and how many times was it rewarded on each side. Data type 2 is the streak data from i previous trials. This data will contain 1 variable and is the most recent streak of either successes or failures in the i previous trials.

Data type 3 accounts for fewer previous trials then data type 1 and 2 but adds event information to each trial. For each trial out of p trials (as noted p<i) the data will be: side of decision, rewarded or not, ITI, and decision time, i.e, 4 variables.

Data type 4 is the events from trial we wish to predict its outcome, thus contains 2 variables.
