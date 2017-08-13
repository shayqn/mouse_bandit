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
We modeled the optimal player as a hidden Markov model doing online inference to make decisions based on previous reward outcomes. The hidden Markov model has a single latent variable representing which state the system is in (i.e. which port has the higher reward probability). 

When provided the correct task parameters (the emission and transition probabilities) and acting according to a greedy decision policy, the model outperforms mice in all three conditions (90-10, 80-20, and 70-30 reward probabilities). However, we can not assume that the mice, if they are acting as hidden Markov models, have estimated the correct model parameters. If we instead infer the most likely model parameters from the mouse behavior, the resulting models behave much more similarly to the mice. 



