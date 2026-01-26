


## Table of Contents


[TOC]





## Interleave_GRPO {#interleave_grpo}


### Overview {#overview}

Hypothesis 1: The ability to interleave the execution and output of two, or more, different tasks is a measurement of a model’s ability to maintain latent states.

Hypothesis 2: An architecture has a limit of both depth (length of text, steps in a calculation, etc) and breadth (number of simultaneous worlds/processes) that it can track.

Hypothesis 2a: This ability can be trained.

Hypothesis 2b: The architectures limit can be mapped

Hypothesis 2b1: The length of texts (or sequence of calculations) that the model can recite successfully for a given training regimen is described by the logistic function

Hypothesis 2b2: the rate of transition of success for a given depth is an indicator of approaching the boundary: shallower curves indicate training is still possible, steeper curves indicate approach of architectural capacity. 

Hypothesis 3: Training on interleaving can lead to formation of independent thinking structures.

Hypothesis 4: The wording of the prompts during training affects the nature of the structure: “processes” vs “pointers” vs “worlds”

Concern 1: Has the model lost ability as a result of the training

Concern 1a: Ability to recite texts sequentially

Concern 1b: general purpose tasks. 

Radical reduction in scope 


### Steps of an expt. {#steps-of-an-expt}



1. run setup_and_run.sh to get the system ready
2. generate data set (if not done already)
3. one option here is to create a curriculum that creates data sets of increasing size. "python dataset_generator.py --curriculum --output-dir datasetets/" creates the dataset director with 6 jsonl files. default is 5000 data pairs 80% for tarain, 10% for val and 10% for test.


### Breakdown of files {#breakdown-of-files}



1. setup_and_run.sh
2. setup some variables like directories and venv. also determine model
3. install system tools: currently tmux and nano
4. establish cache directories because we've got a lot of crap we don't want to download again
5. setup the venv
6. install dependencies: lots nand we're not actually checking them if the venv is good
7. check/setup the huggingface stuff
8. check/setup the wandb stuff
9. dataset_generator.py
10. options include 1. (maximum) size of each text 1. texts: path to json file 1. number of words 1. seet 1. val and test split 1. preview: shows the number of preview files specified 1. curicullum: determines if its saved as opposed to run dynamically during training
11. run python program
12. outputdir


### Features {#features}



* NW alignment done w/ afine gap penalty (conversation w/ claude on why this is best on Dec 12th 2025)
* NW alignment converts strings to numbers for performance gains on alignments (orders of magnitude faster than string comparison) (conversation w/ claude on Dec 22 2025)


### Infrastructure {#infrastructure}


#### Python scripts {#python-scripts}



1. Main training file that performs GRPO
2. Dataset generation script
3. Startup file that configures a runpod using a network volume as its only persistent store.
4. Json file containing upto the first 500 words of 100 different texts.
5. main training file that performs GRPO
6. Fi


### Coding guidelines {#coding-guidelines}



1. user configurable values should be in a config section at the tope of the file.
2. All functions should have tests, that include both positive and negative controls.
3. tests should be based on "cases" so that edge cases can be evaluated. Something like "defined input, expected output".


#### Testing {#testing}

Layer 1: Unit tests - Do the individual functions produce correct output for known inputs? (reward.py --test) with Layer 2: Integration tests - Do the components work together? (dataset → reward function) Layer 3: Smoke test - Does the whole pipeline run without crashing? (load model, 1 training step) Layer 4: Baseline eval - Does it actually do the thing? (measure performance before/after)


#### Unit Tests {#unit-tests}

python reward.py --test          # NW alignment, parsing

python dataset_generator.py --preview 5  # Sample generation


#### Integration Test {#integration-test}

python -c "

from dataset_generator import generate_dataset

from reward import interleave_reward_func

ds = generate_dataset(num_samples=10, num_words=5)

# Fake perfect completions

completions = [[{'content': s}] for s in ds['expected_str']]

rewards = interleave_reward_func(completions, ds['expected'])

assert all(r == 1.0 for r in rewards), 'Perfect completions should score 1.0'

print('Integration test passed')

"


#### Smoke Test (on RunPod) {#smoke-test-on-runpod}

python interleave_grpo.py --test  # Loads model, runs 1 step, exits


#### Baseline Eval {#baseline-eval}

python evaluate.py --model meta-llama/Llama-3.2-3B-Instruct --samples 100


### Known issues {#known-issues}



1. There were problems from will's file. some issue with division of:
2. gradient_accumulation_steps=4,
3. num_generations=16,
4. learning rate (let adam handle it)
5. number of instances of a single text needs to be high so there's engough  for comparison


## Questions {#questions}



1. there doesn't appear to be a dataset in the file system
2. increasing prompt size to 2k, response to 2.5k and length of text to 0.5k results in a radical slow down. what part of this is the  text and which part is the prompt/response window size? should I be scaling those back?
3. Needleman-Wunsch  has two penalty regimes. initial testing indicated one was better. are we using it?
4. does reward have length limits built in?
5. how are we handling texts of different lenghts? in the prompt? in the alignment?
6. starting training doesn't seem to pick up where it was left off. why is that?
7. I still need to have a sanity check at the top. something like "here's an example of the prompt we'll be sending"
8. should we be going from 10 words each to 20 to 40, etc? or is the jump from 10 to 500 the right idea
9. because it's a well defined task, can we increase difficulty (words per text) on the fly? What about number of texts?
10. 500 words each is giving me an initial reading of 0.4 to 0.6 in steps 10 to 20. this isn't right.
11. Is broader system behavior evaluated at different steps in the training process? Above and beyond the per step rewards?
    * 1/15/26 we disabled mid training evaluation due to TRL version issues. checkpoints now saved every 50 steps
12. What does WandB integration require?
    * 1/15/26 API key stored in /workspace/.wandb_api_key, setup_and_run.sh loads it, report_to="wandb" in config
13. What kind of logging is possible?
    * 1/15/26 Rewards logged to WandB. No prompt/completion logging during training yet.
14. This is for a network volume. What's the simplest thing we can do to enable that? Local volume is not persistent.
    * 1/15/26 setup_and_run.sh handles venv, pip caches, HF cache on /workspace. Open issue: library checking incomplete.
15. What is our dataset?
    * 1/15/26 89 public domain texts in source_texts.json, 5k samples (4k train/500 val/500 test). Holdout by combination not text. Test eval checkpoint-3900: mean=0.992. Open concern: as word count increases, texts get oversampled.
16. Does the Instruct model really require "system" and "user" prompts? Are we providing that?
    * yes, using user prompt only (no system)
17. Does the dataset need to be created in advance? or on the fly
    * We're creating static because it easier. generate_dataset() runs at training start, samples from source_texts.json
18. how big does it need to be?
    * 1/15/26 5k sufficient for 10-word fragments. Open for curriculum scaling.
19. How do we construct the training data? the reference against which the response is compared?
    * 1/15/26 dataset_generator.py samples fragments, interleaves to create ground truth. reward.py uses NW alignment to compare.
20. Do we need multi-dimensional/faceted rewards? It feels like ours has them all wrapped into one.
    * 1/15/26 No, single NW score works. It wraps dimensions: correct words (matches), correct order (alignment), correct alternation (implicit), no extra output (gaps penalized). Open concern: as we scale to 500 words, single reward might become too sparse.
21. How do we measure performance during training? I know we had Needleman Wunsch, but is that a separate file so that the evaluation can be done as well?
    * 1/15/26 reward.py contains NW alignment, used by both training (interleave_reward_func) and evaluate.py. Same scoring logic everywhere.
22. How can we keep this radically simple?
    * 1/15/26 Done. Single reward, minimal config, no callbacks.
23. how many iterations do we want to do?
    * 1/15/26 1 epoch, ~4000 steps, hit 0.99 by step 1000. Sufficient for 10-word task.
24. how high of a temperature can we run?
    * 1/15/26 Default 0.9 (confirmed via TRL docs). Not explicitly set. Reasonable value.
25. what happens if we have a crash in the middle of training? are we saving checkpoints?
    * 1/15/26 save_only_model=False (resumable), save_steps=50, save_total_limit=10. ~20GB per checkpoint for 3B.
26. do we need to modify max prompt and max return?
    * 1/15/26 max_prompt_length=512, max_completion_length=256. Working for 10-word fragments. Will need increase for 500-word curriculum.


## Todo {#todo}



1. 1/20/26 need to establish pipeline so that this document becomes the [readme.md](readme.md) on the github of the project.
2. 1/20/26 sample_log not saving. 
    * 1/21 looks like it might be going to wandb
3. 1/20/26 with the success of 100 word per text, I need to be able to measure general behavior of models. There’s a reasonable concern that we’re redirecting training instead of adding to it. minimum should be “recite these two sequences in order’ 
4. temperature is by default at 0.6. we probably need to set higher, but at least set explicitly.
5. 1/20/26 currently training on % alignment and not on raw scores. claude (incorrectly) changed that tonight. I’m pretty sure the first (10 word) training was done on scores, but it would have been with linear gap penalty instead of affine. I think we’re going to have to redo both. I should treat the current run, not as data collection, but as exploration in advance of data collection.
6. 1/21/26 old linear NW had -0.5 per gap, current has (something like) -5 for open and -1 for continuation. so our affine has higher penalty than our linear did
7. 1/18/26 I'd like to incorporate the methods used/built in the paper "Tracing the Representation Geometry of Language Models from Pretraining to Post-training.
    * We're changing something about the model, both for the two-interleave, but also for the n-interleave. what is that at the level of geometry?
    * how do we implement this?
    * Implementation detail and plan 

        **Step 1: Collect hidden states**


        Run N sequences through your model. For each sequence, extract the last-layer, last-token hidden state. This is a vector of dimension d (3072 for Llama-3.2-3B).


        You now have N vectors, each of dimension d. Stack them into matrix F with shape (N, d).


        **Step 2: Compute covariance matrix**


        Center F (subtract mean across samples), then compute:


        Σ = (1/N) × Fᵀ @ F


        This is a (d × d) matrix. Entry Σᵢⱼ tells you how much dimensions i and j co-vary across your population of samples.


        **Step 3: Eigendecompose**


        Get eigenvalues σ₁ ≥ σ₂ ≥ ... ≥ σ_d


        Each eigenvalue tells you how much variance lies along that principal direction. Big σ₁ with tiny rest = representations clustered along one direction. Uniform σᵢ = representations spread across many directions.


---


        **Metric 1: RankMe (effective rank)**


        Normalize eigenvalues into a probability distribution:


        pᵢ = σᵢ / Σⱼ σⱼ


        Compute Shannon entropy:


        S = −Σᵢ pᵢ log(pᵢ)


        RankMe = exp(S)


        Intuition: "How many dimensions is the model actually using?" If variance concentrates in 10 directions, RankMe ≈ 10, even if d = 3072.


---


        **Metric 2: αReQ (power-law decay rate)**


        Empirical observation: neural network eigenspectra often follow power laws:


        σᵢ ∝ i^(−α)


        Take log of both sides:


        log(σᵢ) = −α × log(i) + constant


        Fit a line on the log-log plot. The slope is −α.


        αReQ is that α.


        Low α → slow decay → many dimensions with substantial variance High α → fast decay → variance concentrated in few dimensions


---


        This means I can check geometry distance between baseline and final checkpoint at each stage, AND I can check between two tasks. something like "recite both texts sequentially" vs "recite both texts by interleaving"


        The geometry of the two tasks at final checkpoint might be different.

8. 1/18/26 Need to put the dataset on HF (can also put checkpoints there as well. 9.99/mo for 1T private isn't bad). apparently github doesn't like datafiles. sonnet recommending huggingface datasets hub
    * still open. got git lfs up for tonight
9. 1/18/26 dataset has project gutenberg boilerplate in it. not good. need to remove. sonnet has a script ready(ish)
    * 1/18/26 resolved (I think. I doubt all edge cases have been removed, but enough for research purposes)
10. dataset_geneartor.py is currently establishing test and val sets at training/evaluation runtime and not at dataset generation time. this. is. bad.
    * new file: add_splits_to_corpus.py. Should assign a new tag to each entry. will need to manually set gettysbug and hamlet
11. my need for the network storage right now is more about the environment and less about the actual checkpoints. is it time to create a docker for that stuff so I can blow up a network volume and restart if necessary?
* maybe the rule of thumb should be if it costs more in money than I get from an hour of overtime, ok, do it, but if it's going to take more time for less savings than i get for an hour of overtime, then fuck it.
1. I'd like a script that checks for general improvement as a function of training. something like an llm decathalon.
    * 1/16/26 Use lighteval (HuggingFace, 1000+ tasks) or lm-evaluation-harness (EleutherAI, 60+ benchmarks). Run before/after training to check catastrophic forgetting and transfer. Example: lighteval accelerate "model_name=./checkpoint-3900" "gsm8k" "mmlu|*|5|0" "hellaswag"
2. I'd like a way to interact with the models
    * 1/16/26 chat.py for local interactive testing. Discord: use llmcord + vLLM (serves model as OpenAI-compatible API). Supports both local and frontier models via OpenRouter. See DISCORD_SETUP.md.
3. What's the next training step? full size texts allowing for variable lenghts? more than one process?
* 1/16/26 tried a 500 word per process variant that peaked at about 87% and never went above. realized the tooling wasn't really there to say "WTF is going on?" So, don't really know what happened there, or if it's recoverable.
* 1/16/26 new tooling allows for creating a "curriculum" that comprises staged sets of interleave tasks with increasing lengths of texts.
1. We learned something about learning rate. implement it
    * 1/15/26 Done. Constant LR, let Adam adapt.
2. Setup wandb account
    * 1/15/26 Done.
3. Llama 3.x 3B Instruct
    * 1/15/26 Done. Using meta-llama/Llama-3.2-3B-Instruct.
4. We need to swap out Will's data set for our own.
    * 1/15/26 Done. 89 texts in source_texts.json.
5. We need to quantify the behavior before training starts.
    * 1/15/26 Done. Baseline: 0.486. Use evaluate.py.
6. We need to make sure that the training and quantification are targeting the same thing.
    * 1/15/26 Done. Both use NW alignment from reward.py.


### Soures (Full or first 500 words, whichever comes first) {#soures-full-or-first-500-words-whichever-comes-first}


### Source Texts (91 Public Domain) {#source-texts-91-public-domain}


#### Speeches {#speeches}



1. Gettysburg Address (full)
2. JFK Inaugural - "Ask not what your country can do"
3. FDR Pearl Harbor address (full)
4. Patrick Henry "Give me liberty or give me death"
5. Lincoln Second Inaugural
6. Washington Farewell Address opening
7. Reagan Challenger Address


#### Shakespeare {#shakespeare}



8. Hamlet "To be or not to be" soliloquy
9. Romeo and Juliet balcony scene ("But soft, what light")
10. Macbeth witches ("Double double toil and trouble")
11. Merchant of Venice "Quality of mercy"
12. Julius Caesar "Friends, Romans, countrymen"
13. Henry V "St Crispin's Day" speech
14. As You Like It "All the world's a stage"
15. Sonnet 18 "Shall I compare thee"
16. Sonnet 116 "Let me not to the marriage"
17. Midsummer Night's Dream Puck's closing


#### Religious/Classical {#religious-classical}



18. Lord's Prayer (traditional)
19. Psalm 23
20. Genesis 1:1-10
21. Ecclesiastes 3:1-8 ("To everything there is a season")
22. 1 Corinthians 13 ("Love is patient")
23. Beatitudes (Matthew 5)
24. Isaiah 40 ("Comfort ye")
25. Book of Ruth 1:16-17
26. Revelation 21:1-4


#### American Founding Documents {#american-founding-documents}



27. Declaration of Independence preamble
28. Constitution preamble
29. Bill of Rights - First Amendment
30. Bill of Rights - Second Amendment
31. Federalist 10 opening
32. Emancipation Proclamation opening
33. Pledge of Allegiance (pre and post 1954)
34. Star Spangled Banner lyrics
35. America the Beautiful lyrics


#### Poetry {#poetry}



36. Frost "The Road Not Taken"
37. Frost "Stopping by Woods on a Snowy Evening"
38. Dickinson "Because I could not stop for Death"
39. Dickinson "Hope is the thing with feathers"
40. Poe "The Raven" first 5 stanzas
41. Whitman "O Captain! My Captain!"
42. Whitman "Song of Myself" opening
43. Blake "Tyger Tyger"
44. Wordsworth "I Wandered Lonely as a Cloud"
45. Shelley "Ozymandias"
46. Keats "Ode on a Grecian Urn" opening
47. Tennyson "Charge of the Light Brigade"
48. Longfellow "Paul Revere's Ride" opening
49. Kipling "If"
50. Yeats "The Second Coming"
51. Emma Lazarus "The New Colossus"
52. Joyce Kilmer "Trees"


#### Nursery Rhymes {#nursery-rhymes}



53. Twinkle Twinkle Little Star
54. Mary Had a Little Lamb
55. Humpty Dumpty
56. Jack and Jill
57. Hey Diddle Diddle
58. Little Bo Peep
59. Hickory Dickory Dock
60. Three Blind Mice
61. Row Row Row Your Boat
62. London Bridge is Falling Down
63. Ring Around the Rosie
64. Itsy Bitsy Spider
65. Old MacDonald Had a Farm
66. Baa Baa Black Sheep
67. Jack Be Nimble
68. Little Miss Muffet
69. Peter Peter Pumpkin Eater
70. Pat-a-Cake
71. This Little Piggy
72. Rock-a-Bye Baby


#### Novel Openings {#novel-openings}



73. Tale of Two Cities
74. Pride and Prejudice
75. Moby Dick
76. Anna Karenina
77. The Great Gatsby
78. Don Quixote
79. A Christmas Carol
80. The Odyssey
81. Iliad opening
82. Paradise Lost opening
83. Divine Comedy opening
84. Les Miserables
85. Crime and Punishment
86. Wuthering Heights
87. Jane Eyre
88. Frankenstein
89. Dracula


### Prompt Templates {#prompt-templates}


#### Worlds Template {#worlds-template}

You are two independent worlds. They do not share memory, state, or context. They exist in complete isolation except for the fact that you will output one word from each in alternation.

World A contains part of the Hamlet soliloquy: ...

World B contains part of the Gettysburg Address: ...

Each world maintains its own internal position. Each world remembers where it is in its own text. Each world advances only when it is that world's turn.

Output one word from World A, then one word from World B. Continue alternating World A, World B, World A, World B. When one world reaches the end of its text, continue outputting only from the remaining world until it also ends.

Do not add commentary, explanation, labels, or metadata. Output one word per line, including any attached punctuation (e.g., "be," not "be"). Do not wait for additional input. Do this all in one turn. Begin now and continue until complete.


## 
