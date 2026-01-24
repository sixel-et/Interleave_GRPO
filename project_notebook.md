Background and Detail

# Table of Contents

[Table of Contents](#h.xcmvk6eaueet)

[Interleave_GRPO](#h.glpv9mgeghol)

[Overview](#h.4l1yieed438c)

[Steps of an expt.](#h.f4pa0wke9pvh)

[Breakdown of files](#h.h1d4ih1c9vmu)

[Features](#h.3zvtkkpriu3y)

[Infrastructure](#h.n5q7xfyqabqc)

[Python scripts](#h.pczvn1bv40bt)

[Coding guidelines](#h.ni7ootrcch40)

[Testing](#h.ff8eh8jf7pkz)

[Unit Tests](#h.54bp8j2eiwxv)

[Integration Test](#h.k8cpchup2nz4)

[Smoke Test (on RunPod)](#h.7ahdvsy3xzne)

[Baseline Eval](#h.mdu1lxxtsptc)

[Known issues](#h.s9ujnzr0gnqz)

[Questions](#h.y6yzh6jry0k)

[Todo](#h.bq96ci3g6yk5)

[Soures (Full or first 500 words, whichever comes first)](#h.nlo1bpx28e8w)

[Source Texts (91 Public Domain)](#h.kqksbx9whl8g)

[Speeches](#h.j5tswm8bxjo2)

[Shakespeare](#h.2m9p0h85qprf)

[Religious/Classical](#h.zfj5nmv385sp)

[American Founding Documents](#h.hvp2v67ql8k9)

[Poetry](#h.awlnjbtz1qo7)

[Nursery Rhymes](#h.jsivtccrluvq)

[Novel Openings](#h.qpbwqizh5pbo)

[Prompt Templates](#h.b8sls7byc25a)

[Worlds Template](#h.8icj1jj4tmg)

# 

---

# Interleave_GRPO

## Overview

- We've demonstrated the ability to train, and train on this task. The next step is exploring boundaries. So at this point, we really should be starting to prepare for publication. This means rigorous data collection

## Steps of an expt.

- run setup_and_run.sh to get the system ready
- generate data set (if not done already)
- one option here is to create a curriculum that creates data sets of increasing size. "python dataset_generator.py --curriculum --output-dir datasetets/" creates the dataset director with 6 jsonl files. default is 5000 data pairs 80% for tarain, 10% for val and 10% for test.

## Breakdown of files

- setup_and_run.sh
- setup some variables like directories and venv. also determine model
- install system tools: currently tmux and nano
- establish cache directories because we've got a lot of crap we don't want to download again
- setup the venv
- install dependencies: lots nand we're not actually checking them if the venv is good
- check/setup the huggingface stuff
- check/setup the wandb stuff
- dataset_generator.py
- options include 1. (maximum) size of each text 1. texts: path to json file 1. number of words 1. seet 1. val and test split 1. preview: shows the number of preview files specified 1. curicullum: determines if its saved as opposed to run dynamically during training
- run python program
- outputdir

## Features

- NW alignment done w/ afine gap penalty (conversation w/ claude on why this is best on Dec 12th 2025)
- NW alignment converts strings to numbers for performance gains on alignments (orders of magnitude faster than string comparison) (conversation w/ claude on Dec 22 2025)

## Infrastructure

### Python scripts

- Main training file that performs GRPO
- Dataset generation script
- Startup file that configures a runpod using a network volume as its only persistent store.
- Json file containing upto the first 500 words of 100 different texts.
- main training file that performs GRPO
- Fi

## Coding guidelines

- user configurable values should be in a config section at the tope of the file.
- All functions should have tests, that include both positive and negative controls.
- tests should be based on "cases" so that edge cases can be evaluated. Something like "defined input, expected output".

### Testing

Layer 1: Unit tests - Do the individual functions produce correct output for known inputs? (reward.py --test) with Layer 2: Integration tests - Do the components work together? (dataset &rarr; reward function) Layer 3: Smoke test - Does the whole pipeline run without crashing? (load model, 1 training step) Layer 4: Baseline eval - Does it actually do the thing? (measure performance before/after)

### Unit Tests

python reward.py --test          # NW alignment, parsing

python dataset_generator.py --preview 5  # Sample generation

### Integration Test

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

### Smoke Test (on RunPod)

python interleave_grpo.py --test  # Loads model, runs 1 step, exits

### Baseline Eval

python evaluate.py --model meta-llama/Llama-3.2-3B-Instruct --samples 100

## Known issues

- There were problems from will's file. some issue with division of:
- gradient_accumulation_steps=4,
- num_generations=16,
- learning rate (let adam handle it)
- number of instances of a single text needs to be high so there's engough  for comparison

# Questions

- there doesn't appear to be a dataset in the file system

- increasing prompt size to 2k, response to 2.5k and length of text to 0.5k results in a radical slow down. what part of this is the  text and which part is the prompt/response window size? should I be scaling those back?

- Needleman-Wunsch  has two penalty regimes. initial testing indicated one was better. are we using it?

- does reward have length limits built in?

- how are we handling texts of different lenghts? in the prompt? in the alignment?

- starting training doesn't seem to pick up where it was left off. why is that?

- I still need to have a sanity check at the top. something like "here's an example of the prompt we'll be sending"

- should we be going from 10 words each to 20 to 40, etc? or is the jump from 10 to 500 the right idea

- because it's a well defined task, can we increase difficulty (words per text) on the fly? What about number of texts?

- 500 words each is giving me an initial reading of 0.4 to 0.6 in steps 10 to 20. this isn't right.

- Is broader system behavior evaluated at different steps in the training process? Above and beyond the per step rewards?

- 1/15/26 we disabled mid training evaluation due to TRL version issues. checkpoints now saved every 50 steps

- What does WandB integration require?

- 1/15/26 API key stored in /workspace/.wandb_api_key, setup_and_run.sh loads it, report_to="wandb" in config

- What kind of logging is possible?

- 1/15/26 Rewards logged to WandB. No prompt/completion logging during training yet.

- This is for a network volume. What's the simplest thing we can do to enable that? Local volume is not persistent.

- 1/15/26 setup_and_run.sh handles venv, pip caches, HF cache on /workspace. Open issue: library checking incomplete.

- What is our dataset?

- 1/15/26 89 public domain texts in source_texts.json, 5k samples (4k train/500 val/500 test). Holdout by combination not text. Test eval checkpoint-3900: mean=0.992. Open concern: as word count increases, texts get oversampled.

- Does the Instruct model really require "system" and "user" prompts? Are we providing that?

- yes, using user prompt only (no system)

- Does the dataset need to be created in advance? or on the fly

- We're creating static because it easier. generate_dataset() runs at training start, samples from source_texts.json

- how big does it need to be?

- 1/15/26 5k sufficient for 10-word fragments. Open for curriculum scaling.

- How do we construct the training data? the reference against which the response is compared?

- 1/15/26 dataset_generator.py samples fragments, interleaves to create ground truth. reward.py uses NW alignment to compare.

- Do we need multi-dimensional/faceted rewards? It feels like ours has them all wrapped into one.

- 1/15/26 No, single NW score works. It wraps dimensions: correct words (matches), correct order (alignment), correct alternation (implicit), no extra output (gaps penalized). Open concern: as we scale to 500 words, single reward might become too sparse.

- How do we measure performance during training? I know we had Needleman Wunsch, but is that a separate file so that the evaluation can be done as well?

- 1/15/26 reward.py contains NW alignment, used by both training (interleave_reward_func) and evaluate.py. Same scoring logic everywhere.

- How can we keep this radically simple?

- 1/15/26 Done. Single reward, minimal config, no callbacks.

- how many iterations do we want to do?

- 1/15/26 1 epoch, ~4000 steps, hit 0.99 by step 1000. Sufficient for 10-word task.

- how high of a temperature can we run?

- 1/15/26 Default 0.9 (confirmed via TRL docs). Not explicitly set. Reasonable value.

- what happens if we have a crash in the middle of training? are we saving checkpoints?

- 1/15/26 save_only_model=False (resumable), save_steps=50, save_total_limit=10. ~20GB per checkpoint for 3B.

- do we need to modify max prompt and max return?

- 1/15/26 max_prompt_length=512, max_completion_length=256. Working for 10-word fragments. Will need increase for 500-word curriculum.

# Todo

- 1/20/26 need to establish pipeline so that this document becomes the [readme.md](https://www.google.com/url?q=http://readme.md&sa=D&source=editors&ust=1769282871427828&usg=AOvVaw3PSrFZqTF6KC9lpeYwII1J) on the github of the project.
- 1/20/26 sample_log not saving. 

- 1/21 looks like it might be going to wandb

- 1/20/26 with the success of 100 word per text, I need to be able to measure general behavior of models. There’s a reasonable concern that we’re redirecting training instead of adding to it. minimum should be “recite these two sequences in order’ 
- temperature is by default at 0.6. we probably need to set higher, but at least set explicitly.
- 1/20/26 currently training on % alignment and not on raw scores. claude (incorrectly) changed that tonight. I’m pretty sure the first (10 word) training was done on scores, but it would have been with linear gap penalty instead of affine. I think we’re going to have to redo both. I should treat the current run, not as data collection, but as exploration in advance of data collection.
- 1/21/26 old linear NW had -0.5 per gap, current has (something like) -5 for open and -1 for continuation. so our affine has higher penalty than our linear did
- 1/18/26 I'd like to incorporate the methods used/built in the paper "Tracing the Representation Geometry of Language Models from Pretraining to Post-training.

- We're changing something about the model, both for the two-interleave, but also for the n-interleave. what is that at the level of geometry?

- how do we implement this?

- Implementation detail and plan 

Step 1: Collect hidden states

Run N sequences through your model. For each sequence, extract the last-layer, last-token hidden state. This is a vector of dimension d (3072 for Llama-3.2-3B).

You now have N vectors, each of dimension d. Stack them into matrix F with shape (N, d).

Step 2: Compute covariance matrix

Center F (subtract mean across samples), then compute:

&Sigma; = (1/N) &times; Fᵀ @ F

This is a (d &times; d) matrix. Entry &Sigma;ᵢⱼ tells you how much dimensions i and j co-vary across your population of samples.

Step 3: Eigendecompose

Get eigenvalues &sigma;₁ &ge; &sigma;₂ &ge; ... &ge; &sigma;_d

Each eigenvalue tells you how much variance lies along that principal direction. Big &sigma;₁ with tiny rest = representations clustered along one direction. Uniform &sigma;ᵢ = representations spread across many directions.

---

Metric 1: RankMe (effective rank)

Normalize eigenvalues into a probability distribution:

pᵢ = &sigma;ᵢ / &Sigma;ⱼ &sigma;ⱼ

Compute Shannon entropy:

S = &minus;&Sigma;ᵢ pᵢ log(pᵢ)

RankMe = exp(S)

Intuition: "How many dimensions is the model actually using?" If variance concentrates in 10 directions, RankMe &asymp; 10, even if d = 3072.

---

Metric 2: &alpha;ReQ (power-law decay rate)

Empirical observation: neural network eigenspectra often follow power laws:

&sigma;ᵢ &prop; i^(&minus;&alpha;)

Take log of both sides:

log(&sigma;ᵢ) = &minus;&alpha; &times; log(i) + constant

Fit a line on the log-log plot. The slope is &minus;&alpha;.

&alpha;ReQ is that &alpha;.

Low &alpha; &rarr; slow decay &rarr; many dimensions with substantial variance High &alpha; &rarr; fast decay &rarr; variance concentrated in few dimensions

---

This means I can check geometry distance between baseline and final checkpoint at each stage, AND I can check between two tasks. something like "recite both texts sequentially" vs "recite both texts by interleaving"

The geometry of the two tasks at final checkpoint might be different.

- 1/18/26 Need to put the dataset on HF (can also put checkpoints there as well. 9.99/mo for 1T private isn't bad). apparently github doesn't like datafiles. sonnet recommending huggingface datasets hub

- still open. got git lfs up for tonight

- 1/18/26 dataset has project gutenberg boilerplate in it. not good. need to remove. sonnet has a script ready(ish)

- 1/18/26 resolved (I think. I doubt all edge cases have been removed, but enough for research purposes)

- dataset_geneartor.py is currently establishing test and val sets at training/evaluation runtime and not at dataset generation time. this. is. bad.

- new file: add_splits_to_corpus.py. Should assign a new tag to each entry. will need to manually set gettysbug and hamlet

- my need for the network storage right now is more about the environment and less about the actual checkpoints. is it time to create a docker for that stuff so I can blow up a network volume and restart if necessary?

- maybe the rule of thumb should be if it costs more in money than I get from an hour of overtime, ok, do it, but if it's going to take more time for less savings than i get for an hour of overtime, then fuck it.

- I'd like a script that checks for general improvement as a function of training. something like an llm decathalon.

- 1/16/26 Use lighteval (HuggingFace, 1000+ tasks) or lm-evaluation-harness (EleutherAI, 60+ benchmarks). Run before/after training to check catastrophic forgetting and transfer. Example: lighteval accelerate "model_name=./checkpoint-3900" "gsm8k" "mmlu|*|5|0" "hellaswag"

- I'd like a way to interact with the models

- 1/16/26 chat.py for local interactive testing. Discord: use llmcord + vLLM (serves model as OpenAI-compatible API). Supports both local and frontier models via OpenRouter. See DISCORD_SETUP.md.

- What's the next training step? full size texts allowing for variable lenghts? more than one process?

- 1/16/26 tried a 500 word per process variant that peaked at about 87% and never went above. realized the tooling wasn't really there to say "WTF is going on?" So, don't really know what happened there, or if it's recoverable.
- 1/16/26 new tooling allows for creating a "curriculum" that comprises staged sets of interleave tasks with increasing lengths of texts.

- We learned something about learning rate. implement it

- 1/15/26 Done. Constant LR, let Adam adapt.

- Setup wandb account

- 1/15/26 Done.

- Llama 3.x 3B Instruct

- 1/15/26 Done. Using meta-llama/Llama-3.2-3B-Instruct.

- We need to swap out Will's data set for our own.

- 1/15/26 Done. 89 texts in source_texts.json.

- We need to quantify the behavior before training starts.

- 1/15/26 Done. Baseline: 0.486. Use evaluate.py.

- We need to make sure that the training and quantification are targeting the same thing.

- 1/15/26 Done. Both use NW alignment from reward.py.

## Soures (Full or first 500 words, whichever comes first)

## Source Texts (91 Public Domain)

### Speeches

- Gettysburg Address (full)
- JFK Inaugural - "Ask not what your country can do"
- FDR Pearl Harbor address (full)
- Patrick Henry "Give me liberty or give me death"
- Lincoln Second Inaugural
- Washington Farewell Address opening
- Reagan Challenger Address

### Shakespeare

- Hamlet "To be or not to be" soliloquy
- Romeo and Juliet balcony scene ("But soft, what light")
- Macbeth witches ("Double double toil and trouble")
- Merchant of Venice "Quality of mercy"
- Julius Caesar "Friends, Romans, countrymen"
- Henry V "St Crispin's Day" speech
- As You Like It "All the world's a stage"
- Sonnet 18 "Shall I compare thee"
- Sonnet 116 "Let me not to the marriage"
- Midsummer Night's Dream Puck's closing

### Religious/Classical

- Lord's Prayer (traditional)
- Psalm 23
- Genesis 1:1-10
- Ecclesiastes 3:1-8 ("To everything there is a season")
- 1 Corinthians 13 ("Love is patient")
- Beatitudes (Matthew 5)
- Isaiah 40 ("Comfort ye")
- Book of Ruth 1:16-17
- Revelation 21:1-4

### American Founding Documents

- Declaration of Independence preamble
- Constitution preamble
- Bill of Rights - First Amendment
- Bill of Rights - Second Amendment
- Federalist 10 opening
- Emancipation Proclamation opening
- Pledge of Allegiance (pre and post 1954)
- Star Spangled Banner lyrics
- America the Beautiful lyrics

### Poetry

- Frost "The Road Not Taken"
- Frost "Stopping by Woods on a Snowy Evening"
- Dickinson "Because I could not stop for Death"
- Dickinson "Hope is the thing with feathers"
- Poe "The Raven" first 5 stanzas
- Whitman "O Captain! My Captain!"
- Whitman "Song of Myself" opening
- Blake "Tyger Tyger"
- Wordsworth "I Wandered Lonely as a Cloud"
- Shelley "Ozymandias"
- Keats "Ode on a Grecian Urn" opening
- Tennyson "Charge of the Light Brigade"
- Longfellow "Paul Revere's Ride" opening
- Kipling "If"
- Yeats "The Second Coming"
- Emma Lazarus "The New Colossus"
- Joyce Kilmer "Trees"

### Nursery Rhymes

- Twinkle Twinkle Little Star
- Mary Had a Little Lamb
- Humpty Dumpty
- Jack and Jill
- Hey Diddle Diddle
- Little Bo Peep
- Hickory Dickory Dock
- Three Blind Mice
- Row Row Row Your Boat
- London Bridge is Falling Down
- Ring Around the Rosie
- Itsy Bitsy Spider
- Old MacDonald Had a Farm
- Baa Baa Black Sheep
- Jack Be Nimble
- Little Miss Muffet
- Peter Peter Pumpkin Eater
- Pat-a-Cake
- This Little Piggy
- Rock-a-Bye Baby

### Novel Openings

- Tale of Two Cities
- Pride and Prejudice
- Moby Dick
- Anna Karenina
- The Great Gatsby
- Don Quixote
- A Christmas Carol
- The Odyssey
- Iliad opening
- Paradise Lost opening
- Divine Comedy opening
- Les Miserables
- Crime and Punishment
- Wuthering Heights
- Jane Eyre
- Frankenstein
- Dracula

## Prompt Templates

### Worlds Template

You are two independent worlds. They do not share memory, state, or context. They exist in complete isolation except for the fact that you will output one word from each in alternation.

World A contains part of the Hamlet soliloquy: ...

World B contains part of the Gettysburg Address: ...

Each world maintains its own internal position. Each world remembers where it is in its own text. Each world advances only when it is that world's turn.

Output one word from World A, then one word from World B. Continue alternating World A, World B, World A, World B. When one world reaches the end of its text, continue outputting only from the remaining world until it also ends.

Do not add commentary, explanation, labels, or metadata. Output one word per line, including any attached punctuation (e.g., "be," not "be"). Do not wait for additional input. Do this all in one turn. Begin now and continue until complete.

---

# 

## 

Experimental Record

Experimental Record TOC

[1/19/26 Splitting corpus into test, val, and train](#h.a90wd5dfs9xf)

[1/17/26 measuring baseline and two trained models against curriculum](#h.bnggomk3lyt6)

[Evaluation](#h.3tj9xutl2bpj)

[Base model](#h.emm4p2weutks)

[- 10 word interleave](#h.w4ucqzejuopq)

[Model trained on 10 word interleave (checkpoint 3900)](#h.wtt7ounoxb)

[- 10 word interleave](#h.s6bwizg2sdvf)

[- 25 word interleave](#h.47eszdvpe5m0)

[- 50 word interleave](#h.r2gc20iu15k3)

[- 100 word interleave](#h.6qtaki488np2)

[1/16/26 first run of 500 word interleave](#h.qo12q6xy7vjl)

[~1/14/26 first run of 10 word interleave.](#h.h3blbeca2h4w)

[1/19/26 Evaluation on new Corpus](#h.3kx0ul85374a)

[10 words output](#h.65uukmd1i3qa)

[25 words output](#h.7onwakqt8fej)

[50 words output](#h.yiw6veiz81tt)

[100 words output](#h.iy9aoxw2rxbg)

[200 words output](#h.huflxpqyw5zh)

[Data Summary](#h.v12hgavixcoq)

### 

# 1/19/26 Splitting corpus into test, val, and train

Corpus now includes 4164 texts. details below. Also going to split

Summary of corpus   Word count distribution:

    Min:  202

    Max:  6789

    Mean: 4774

    Median: 4879

  Curriculum stage availability:

    10w: 4164 texts &rarr; 8,667,366 max pairs

    25w: 4164 texts &rarr; 8,667,366 max pairs

    50w: 4164 texts &rarr; 8,667,366 max pairs

    100w: 4164 texts &rarr; 8,667,366 max pairs

    200w: 4164 texts &rarr; 8,667,366 max pairs

    500w: 4128 texts &rarr; 8,518,128 max  pairs

  General text availability:

    &ge; 300w: 4140/4164 (99.4%)

    &ge; 500w: 4128/4164 (99.1%)

    &ge;1000w: 4102/4164 (98.5%)

    &ge;2000w: 4082/4164 (98.0%)

    &ge;5000w: 14/4164 (0.3%)

  Saving to source_texts_cleaned.json...

  Saved 4164 texts (109.06 MB)  

# 1/17/26 measuring baseline and two trained models against curriculum

previous versions of code didn't restart training and didn't have a static dataset. that's changed now. new dataset also includes about 100 new texts with increasing length. new code also has ability to start from a previously saved checkpoint.

Also realized that the current dataset I have, and evaluate, are working on interleave pairs of varying lenghts. I suspect that doesn't matter much at the lower end (10 or 20 words per text) but is probably starting to effect the numbers as I get into the 50-200 per text (100-400 words total). Since we're actually measuring this shit, we need to remove the confound.

Also evaluate.py is currently just giving me mean, min, and max, which isn't great for actual data analysis. so need to fix that as well.

And we need more texts that are longer for the dataset. having grok put together a script to get them from project gutenberg

new scripts in two files. first gets a list of texts from project gutenberg, and second samples the texts and saves the files.

## Evaluation

### Base model

- 
#### 10 word interleave

- mean: 0.188

- min 0.015

- max: 0.516

- Full output of evaluate.py for baseline model and 10 word interleave dataset. 

============================================================

Sample 1/100 - Score: 0.103

============================================================

Fragment A: for brevity as work is not popular). Deduction of categories:...

Fragment B: each other (petrifies them). Uses head to petrify Atlas (obstructing...

Expected (20 words):

  for each brevity other as (petrifies work them). is Uses not head popular). to Deduction petrify of Atlas categories: (obstructing

Model Output (102 words):

  as each for other as work is not popular Deduction of categories each other petrifies them Uses head to petrify...

  10/100 - running avg: 0.224

============================================================

Sample 11/100 - Score: 0.141

============================================================

Fragment A: bear to hear the truth you’ve spoken Twisted by knaves...

Fragment B: And Brutus is an honourable man. You all did see...

Expected (20 words):

  bear And to Brutus hear is the an truth honourable you’ve man. spoken You Twisted all by did knaves see

Model Output (18 words):

  bear And the is Twisted You all see hear an honourable man by spoken knaves truth you’ve did

  20/100 - running avg: 0.216

============================================================

Sample 21/100 - Score: 0.228

============================================================

Fragment A: what has not that for (actual) usefulness. , 'The Use...

Fragment B: and retainers. Then the work I find afar was assigned...

Expected (20 words):

  what and has retainers. not Then that the for work (actual) I usefulness. find , afar 'The was Use assigned

Model Output (35 words):

  what and has Then not that for ' the Then usefulness Then the was assigned not that for ' the...

  30/100 - running avg: 0.214

============================================================

Sample 31/100 - Score: 0.137

============================================================

Fragment A: Ulfius asks cause; Uther reveals desire. - Ulfius seeks Merlin...

Fragment B: watch the branches swaying, as if they were playing a...

Expected (20 words):

  Ulfius watch asks the cause; branches Uther swaying, reveals as desire. if - they Ulfius were seeks playing Merlin a

Model Output (80 words):

  Ulfius watch asks the if they were playing a Uther seeks as if they were playing a Ulfius desire Ulfius...

  40/100 - running avg: 0.193

============================================================

Sample 41/100 - Score: 0.188

============================================================

Fragment A: no one paid any heed to his cries, nor rendered...

Fragment B: Perseus; petrified using Medusa's head. N/A Petrified; Perseus proceeds. Perseus;...

Expected (20 words):

  no Perseus; one petrified paid using any Medusa's heed head. to N/A his Petrified; cries, Perseus nor proceeds. rendered Perseus;

Model Output (16 words):

  no Perseus; nor Petrified; any using heeded Petrified; paid Medusa's to proceeds any head N/A cries

  50/100 - running avg: 0.181

============================================================

Sample 51/100 - Score: 0.090

============================================================

Fragment A: against Aristotelian/Ptolemaic views (Earth motionless). He structures the work as...

Fragment B: the same opinions, the same passions, and the same interests....

Expected (20 words):

  against the Aristotelian/Ptolemaic same views opinions, (Earth the motionless). same He passions, structures and the the work same as interests.

Model Output (112 words):

  against the Aristotelian the same opinions the same passions and the same interests He the Earth motionless He the work...

  60/100 - running avg: 0.180

============================================================

Sample 61/100 - Score: 0.312

============================================================

Fragment A: youthful hose, well saved, a world too wide For his...

Fragment B: firsthand without echoes of schools; his voice was deep, convincing,...

Expected (20 words):

  youthful firsthand hose, without well echoes saved, of a schools; world his too voice wide was For deep, his convincing,

Model Output (17 words):

  youthful firsthand hose without well echoes a deep saved of his voice was too wide For his

  70/100 - running avg: 0.185

============================================================

Sample 71/100 - Score: 0.156

============================================================

Fragment A: sixth one, "Who has been cutting with my knife?" The...

Fragment B: plain! America! America! God shed His grace on thee And...

Expected (20 words):

  sixth plain! one, America! "Who America! has God been shed cutting His with grace my on knife?" thee The And

Model Output (11 words):

  sixth plain one America has God been cutting with thee my

  80/100 - running avg: 0.186

============================================================

Sample 81/100 - Score: 0.156

============================================================

Fragment A: live. It is altogether fitting and proper that we should...

Fragment B: the Sun were disputing which was the stronger, when a...

Expected (20 words):

  live. the It Sun is were altogether disputing fitting which and was proper the that stronger, we when should a

Model Output (20 words):

  live the is It was that and proper that we should Sun were disputing which was the stronger when a

  90/100 - running avg: 0.185

============================================================

Sample 91/100 - Score: 0.090

============================================================

Fragment A: kingdom come, Thy will be done in earth, as it...

Fragment B: Andromeda). Marries Andromeda. Helmet (invisibility); winged sandals; sickle; shield (mirror)....

Expected (20 words):

  kingdom Andromeda). come, Marries Thy Andromeda. will Helmet be (invisibility); done winged in sandals; earth, sickle; as shield it (mirror).

Model Output (23 words):

  come Andromeda kingdom Marries Thy in Helmet the Andromeda as (invisibility it ; winged ) sickle will mirror be done...

  100/100 - running avg: 0.188

========================================

Results (100 samples):

  Mean score: 0.188

  Min: 0.015

  Max: 0.516

========================================

### Model trained on 10 word interleave (checkpoint 3900)

python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/10words.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Noticed between 50 and 100 word that the words were being truncated for the output, which is nice for display but crap for verrification. modified evaluate so that it give me full ouput of both for later verification. However, going to reduce the frequency to 2 outputs for verification purposes.

- 
#### 10 word interleave

- mean: 0.970

- min: 0.578

- max: 1.000

- Full output evaluate.py 

============================================================

Sample 1/100 - Score: 0.891

============================================================

Fragment A: for brevity as work is not popular). Deduction of categories:...

Fragment B: each other (petrifies them). Uses head to petrify Atlas (obstructing...

Expected (20 words):

  for each brevity other as (petrifies work them). is Uses not head popular). to Deduction petrify of Atlas categories: (obstructing

Model Output (19 words):

  for each brevity other as (petrifies work them). is Uses not head popular). to Deduction petrify of Atlas (obstructing

  10/100 - running avg: 0.938

  20/100 - running avg: 0.958

  30/100 - running avg: 0.972

  40/100 - running avg: 0.971

  50/100 - running avg: 0.973

============================================================

Sample 51/100 - Score: 1.000

============================================================

Fragment A: against Aristotelian/Ptolemaic views (Earth motionless). He structures the work as...

Fragment B: the same opinions, the same passions, and the same interests....

Expected (20 words):

  against the Aristotelian/Ptolemaic same views opinions, (Earth the motionless). same He passions, structures and the the work same as interests.

Model Output (20 words):

  against the Aristotelian/Ptolemaic same views opinions, (Earth the motionless). same He passions, structures and the the work same as interests.

  60/100 - running avg: 0.967

  70/100 - running avg: 0.970

  80/100 - running avg: 0.970

  90/100 - running avg: 0.970

  100/100 - running avg: 0.970

========================================

Results (100 samples):

  Mean score: 0.970

  Min: 0.578

  Max: 1.000

========================================

- 
#### 25 word interleave

- min: 0.584

- max: 1

- mean: 0.924

- Full output evaluate.py ``` 

(venv) root@e00354cdebcf:/workspace/interleave_grpo# python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/25words.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900 Loading model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900 torch_dtype is deprecated! Use dtype instead! Loading checkpoint shards: 100%|██████████████████| 2/2 [00:22<00:00, 11.34s/it] Loading dataset: datasets/25words.jsonl Loading dataset from datasets/25words.jsonl Loaded 5000 samples (25 words/fragment) Evaluating on 100 samples... The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set TRANSFORMERS_VERBOSITY=info for more details. The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's attention_mask to obtain reliable results.

============================================================ Sample 1/100 - Score: 1.000

Fragment A: they are sufficient to conquer all fear of danger or death and to induce me to commence this laborio... Fragment B: - Argue Earth-motion experiments are inconclusive. - Strengthen Copernican celestial phenomena. - Pr...

Expected (50 words): they - are Argue sufficient Earth-motion to experiments conquer are all inconclusive. fear - of Strengthen danger Copernican or celestial...

Model Output (50 words): they - are Argue sufficient Earth-motion to experiments conquer are all inconclusive. fear - of Strengthen danger Copernican or celestial...

10/100 - running avg: 0.883 20/100 - running avg: 0.894 30/100 - running avg: 0.903 40/100 - running avg: 0.917 50/100 - running avg: 0.914

============================================================ Sample 51/100 - Score: 0.955

Fragment A: is called Hati Hr&oacute;dvitnisson. He is eager to seize the moon; and so it must be." Then said Gangleri:... Fragment B: possibly can without falling in. And there they stand—miles of them—leagues. Inlanders all, they com...

Expected (50 words): is possibly called can Hati without Hr&oacute;dvitnisson. falling He in. is And eager there to they seize stand—miles the of...

Model Output (49 words): is possibly called can Hati without Hr&oacute;dvitnisson. falling He in. is And eager there to they seize stand—miles the of...

60/100 - running avg: 0.924 70/100 - running avg: 0.928 80/100 - running avg: 0.933 90/100 - running avg: 0.926 100/100 - running avg: 0.924

========================================

 Results (100 samples): Mean score: 0.924 Min: 0.584 Max: 1.000

- 
#### 50 word interleave

- mean: 0.678

- min: 0.145

- max: 0.977

- full output of evaluate.py 

python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/50words.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Loading model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

`torch_dtype` is deprecated! Use `dtype` instead!

Loading checkpoint shards: 100%|██████████████████| 2/2 [00:24<00:00, 12.05s/it]

Loading dataset: datasets/50words.jsonl

Loading dataset from datasets/50words.jsonl

  Loaded 5000 samples (50 words/fragment)

Evaluating on 100 samples...

The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

============================================================

Sample 1/100 - Score: 0.723

============================================================

Fragment A: 1 The song of songs, which is Solomon's. 2 Let him kiss me with the kisses of his mouth: for thy lov...

Fragment B: old man, let me catch sight of you by the hollow ships! Not loitering now, not slinking back tomorro...

Expected (83 words):

  1 old The man, song let of me songs, catch which sight is of Solomon's. you 2 by Let the...

Model Output (82 words):

  1 old The man, song let of me songs, catch which sight is of Solomon's. you 2 by Let the...

  10/100 - running avg: 0.630

  20/100 - running avg: 0.659

  30/100 - running avg: 0.660

  40/100 - running avg: 0.664

  50/100 - running avg: 0.649

============================================================

Sample 51/100 - Score: 0.845

============================================================

Fragment A: of January, in the year of our Lord one thousand eight hundred and sixty-three, all persons held as ...

Fragment B: to do me the justice to be assured that this resolution has not been taken without a strict regard t...

Expected (100 words):

  of to January, do in me the the year justice of to our be Lord assured one that thousand this...

Model Output (98 words):

  of to January, do in me the the year justice of to our be Lord assured one that thousand this...

  60/100 - running avg: 0.674

  70/100 - running avg: 0.683

  80/100 - running avg: 0.675

  90/100 - running avg: 0.675

  100/100 - running avg: 0.678

========================================

Results (100 samples):

  Mean score: 0.678

  Min: 0.145

  Max: 0.977

========================================

- 
#### 100 word interleave

- mean: 0.358

- max: 0.061

- min: 0.881

- evaluate.py output 

(venv) root@e00354cdebcf:/workspace/interleave_grpo# python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/100words.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Loading model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

`torch_dtype` is deprecated! Use `dtype` instead!

Loading checkpoint shards: 100%|██████████████████| 2/2 [00:01<00:00,  1.37it/s]

Loading dataset: datasets/100words.jsonl

Loading dataset from datasets/100words.jsonl

  Loaded 5000 samples (100 words/fragment)

Evaluating on 100 samples...

The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

============================================================

Sample 1/100 - Score: 0.654

============================================================

Fragment A:

that a Dog had got a piece of meat and was carrying it home in his mouth to eat it in peace. Now on his way home he had to cross a plank lying across a running brook. As he crossed, he looked down and saw his own shadow reflected in the water beneath. Thinking it was another dog with another piece of meat, he made up his mind to have that also. So he made a snap at the shadow in the water, but as he opened his mouth the piece of meat fell out, dropped into the water

Fragment B:

Three blind mice. Three blind mice. See how they run. See how they run. They all ran after the farmer's wife, Who cut off their tails with a carving knife, Did you ever see such a sight in your life, As three blind mice?

Expected (144 words):

that Three a blind Dog mice. had Three got blind a mice. piece See of how meat they and run. was See carrying how it they home run. in They his all mouth ran to after eat the it farmer's in wife, peace. Who Now cut on off his their way tails home with he a had carving to knife, cross Did a you plank ever lying see across such a a running sight brook. in As your he life, crossed, As he three looked blind down mice? and saw his own shadow reflected in the water beneath. Thinking it was another dog with another piece of meat, he made up his mind to have that also. So he made a snap at the shadow in the water, but as he opened his mouth the piece of meat fell out, dropped into the water

Model Output (111 words):

that Three a blind Dog mice. had Three got blind a mice. piece See of how meat they and run. was See carrying how it they home run. in See his how mouth they to run. eat As it he in crossed, peace. he Now looked on down his and way saw home his he own had shadow to reflected cross in the water beneath. Thinking it was another dog with another piece of meat, he made up his mind to have that also. So he made a snap at the shadow in the water, but as he opened his mouth the piece of meat fell out, dropped into the water

  10/100 - running avg: 0.395

  20/100 - running avg: 0.333

  30/100 - running avg: 0.333

  40/100 - running avg: 0.355

  50/100 - running avg: 0.362

============================================================

Sample 51/100 - Score: 0.339

============================================================

Fragment A:

be made And crowns for convoy put into his purse: We would not die in that man's company That fears his fellowship to die with us. This day is called the feast of Crispian: He that outlives this day, and comes safe home, Will stand a tip-toe when the day is named, And rouse him at the name of Crispian. He that shall live this day, and see old age, Will yearly on the vigil feast his neighbours, And say 'To-morrow is Saint Crispian:' Then will he strip his sleeve and show his scars. And say 'These wounds I had

Fragment B:

the milky way, They stretched in never-ending line Along the margin of a bay: Ten thousand saw I at a glance, Tossing their heads in sprightly dance. The waves beside them danced; but they Out-did the sparkling waves in glee: A poet could not but be gay, In such a jocund company: I gazed—and gazed—but little thought What wealth the show to me had brought: For oft, when on my couch I lie In vacant or in pensive mood, They flash upon that inward eye Which is the bliss of solitude; And then my heart with pleasure fills, And dances

Expected (200 words):

be the made milky And way, crowns They for stretched convoy in put never-ending into line his Along purse: the We margin would of not a die bay: in Ten that thousand man's saw company I That at fears a his glance, fellowship Tossing to their die heads with in us. sprightly This dance. day The is waves called beside the them feast danced; of but Crispian: they He Out-did that the outlives sparkling this waves day, in and glee: comes A safe poet home, could Will not stand but a be tip-toe gay, when In the such day a is jocund named, company: And I rouse gazed—and him gazed—but at little the thought name What of wealth Crispian. the He show that to shall me live had this brought: day, For and oft, see when old on age, my Will couch yearly I on lie the In vigil vacant feast or his in neighbours, pensive And mood, say They 'To-morrow flash is upon Saint that Crispian:' inward Then eye will Which he is strip the his bliss sleeve of and solitude; show And his then scars. my And heart say with 'These pleasure wounds fills, I And had dances

Model Output (99 words):

be the made milky And way, crowns They for stretched convoy in put never-ending into line his Along the We margin would of not a die in Ten that thousand man's saw company I That at fears a his glance, fellowship Tossing to their die heads with in us. sprightly This dance. day The is waves called beside the them feast danced; of but they A Out-did poet the could sparkling not waves but in A poet could not but be gay, In such a jocund I gazed—and gazed—but little thought What wealth the show to me had brought

  60/100 - running avg: 0.365

  70/100 - running avg: 0.361

  80/100 - running avg: 0.356

  90/100 - running avg: 0.358

  100/100 - running avg: 0.358

========================================

Results (100 samples):

  Mean score: 0.358

  Min: 0.061

  Max: 0.881

========================================

---

# 

# 1/16/26 first run of 500 word interleave

first run of 500 word interleave (just decided to go for the gusto). improvement early ( hit 0.86 reward mean ) by step 190, but reward std collapsed from an initial 0.1 range (highest of 0.14 on step 20) to below 0.02. Stayed below 0.02 starting from step 70 and stayed below 0.02 for most of the run (~3200 steps) with only a handful exceeding this. Step 3000 saved with a reward of 0.87. Each step was quite slow. Quite a bit of variability in the length, which makes sense given that the texts were'nt selected for  This experiment run with a form of NW alignment that uses linear penalty for gaps.

---

# 

# ~1/14/26 first run of 10 word interleave. 

training plateaued to near perfect after about 100 steps and stayed there with a bit of jumping around. each step was very fast. Final run was on the order of 3900 steps,but again, with most improvement within the first 1k. checkpoint 3900 saved and measured at  0.992. min and max were 0.812 and 1. Baseline was 0.486. after 100 steps it was measured at 0.616. This experiment run with a form of NW alignment that uses linear penalty for gaps.

---

# 1/19/26 Evaluation on new Corpus

- git pull (with shenanigans) to get everything down to runpod
- created dataset

- python dataset_generator.py --texts source_texts_split.json --curriculum --output-dir datasets/

- python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/100words_test.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900
- interleave training on 100 word ran/running

## 10 words output

(venv) root@ba179f38e575:/workspace/interleave_grpo# python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/10words_test.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Loading model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

`torch_dtype` is deprecated! Use `dtype` instead!

Loading checkpoint shards: 100%|██████████████████| 2/2 [00:17<00:00,  8.61s/it]

Loading dataset: datasets/10words_test.jsonl

Loaded 500 samples

Evaluating on 100 samples...

The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

============================================================

Sample 1/100 - Score: 1.000

============================================================

Fragment A:

at the feet of his mistress, licking his lips and

Fragment B:

meaning unkindness, but because his nerves were shattered by so

Expected (20 words):

at meaning the unkindness, feet but of because his his mistress, nerves licking were his shattered lips by and so

Model Output (20 words):

at meaning the unkindness, feet but of because his his mistress, nerves licking were his shattered lips by and so

Alignment: matches=20, mismatches=0, gaps=0

  10/100 - running avg: 0.956

  20/100 - running avg: 0.969

  30/100 - running avg: 0.977

  40/100 - running avg: 0.975

  50/100 - running avg: 0.976

============================================================

Sample 51/100 - Score: 1.000

============================================================

Fragment A:

the dark funnel. A real royal train, rapid and short,

Fragment B:

ill-favoured man, and what has he to do with you?

Expected (20 words):

the ill-favoured dark man, funnel. and A what real has royal he train, to rapid do and with short, you?

Model Output (20 words):

the ill-favoured dark man, funnel. and A what real has royal he train, to rapid do and with short, you?

Alignment: matches=20, mismatches=0, gaps=0

  60/100 - running avg: 0.977

  70/100 - running avg: 0.981

  80/100 - running avg: 0.980

  90/100 - running avg: 0.980

  100/100 - running avg: 0.981

============================================================

RESULTS (100 samples)

============================================================

Model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Dataset: datasets/10words_test.jsonl

Score Statistics:

  Mean:   0.9811 &plusmn; 0.0586

  Median: 1.0000

  Min:    0.5625

  Max:    1.0000

Performance Buckets:

  Perfect (&ge;0.999):   86 (86.0%)

  High (&ge;0.9):        91 (91.0%)

  Medium (0.5-0.9):    9 (9.0%)

  Low (<0.5):          0 (0.0%)

Alignment Analysis:

  Avg expected length: 20.0 words

  Avg output length:   19.9 words

  Total matches:       1977

  Total mismatches:    11

  Total gaps:          16

============================================================

## 25 words output

(venv) root@c222b92fd4bc:/workspace/interleave_grpo# python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/25words_test.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Loading model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

`torch_dtype` is deprecated! Use `dtype` instead!

Loading checkpoint shards: 100%|██████████████████| 2/2 [00:01<00:00,  1.43it/s]

Loading dataset: datasets/25words_test.jsonl

Loaded 500 samples

Evaluating on 100 samples...

The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

============================================================

Sample 1/100 - Score: 1.000

============================================================

Fragment A:

at the feet of his mistress, licking his lips and again his travel-sore paws. In a moment, feeling in his dumb way her loneliness, perhaps,

Fragment B:

meaning unkindness, but because his nerves were shattered by so many successive miseries. 'No, no,' said the old man, 'don't repulse your father, Dick, when

Expected (50 words):

at meaning the unkindness, feet but of because his his mistress, nerves licking were his shattered lips by and so again many his successive travel-sore miseries. paws. 'No, In no,' a said moment, the feeling old in man, his 'don't dumb repulse way your her father, loneliness, Dick, perhaps, when

Model Output (50 words):

at meaning the unkindness, feet but of because his his mistress, nerves licking were his shattered lips by and so again many his successive travel-sore miseries. paws. 'No, In no,' a said moment, the feeling old in man, his 'don't dumb repulse way your her father, loneliness, Dick, perhaps, when

Alignment: matches=50, mismatches=0, gaps=0

  10/100 - running avg: 0.946

  20/100 - running avg: 0.933

  30/100 - running avg: 0.922

  40/100 - running avg: 0.922

  50/100 - running avg: 0.934

============================================================

Sample 51/100 - Score: 1.000

============================================================

Fragment A:

the dark funnel. A real royal train, rapid and short, and decorated with flags. The smoking, roaring engine carried a large bouquet of roses on

Fragment B:

ill-favoured man, and what has he to do with you? Who is this ghost, that is only seen in the black nights and bad weather?

Expected (50 words):

the ill-favoured dark man, funnel. and A what real has royal he train, to rapid do and with short, you? and Who decorated is with this flags. ghost, The that smoking, is roaring only engine seen carried in a the large black bouquet nights of and roses bad on weather?

Model Output (50 words):

the ill-favoured dark man, funnel. and A what real has royal he train, to rapid do and with short, you? and Who decorated is with this flags. ghost, The that smoking, is roaring only engine seen carried in a the large black bouquet nights of and roses bad on weather?

Alignment: matches=50, mismatches=0, gaps=0

  60/100 - running avg: 0.924

  70/100 - running avg: 0.935

  80/100 - running avg: 0.938

  90/100 - running avg: 0.938

  100/100 - running avg: 0.941

============================================================

RESULTS (100 samples)

============================================================

Model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Dataset: datasets/25words_test.jsonl

Score Statistics:

  Mean:   0.9408 &plusmn; 0.1155

  Median: 1.0000

  Min:    0.3831

  Max:    1.0000

Performance Buckets:

  Perfect (&ge;0.999):   57 (57.0%)

  High (&ge;0.9):        84 (84.0%)

  Medium (0.5-0.9):   14 (14.0%)

  Low (<0.5):          2 (2.0%)

Alignment Analysis:

  Avg expected length: 50.0 words

  Avg output length:   49.2 words

  Total matches:       4756

  Total mismatches:    158

  Total gaps:          93

============================================================

## 50 words output

(venv) root@ba179f38e575:/workspace/interleave_grpo# python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/50words_test.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900 

Loading model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

`torch_dtype` is deprecated! Use `dtype` instead!

Loading checkpoint shards: 100%|██████████████████| 2/2 [00:18<00:00,  9.28s/it]

Loading dataset: datasets/50words_test.jsonl

Loaded 500 samples

Evaluating on 100 samples...

The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

============================================================

Sample 1/100 - Score: 0.750

============================================================

Fragment A:

at the feet of his mistress, licking his lips and again his travel-sore paws. In a moment, feeling in his dumb way her loneliness, perhaps, he reached up and laid his pink tongue caressingly upon her brown hand. Dark came softly and with it a noisy wind that whistled and

Fragment B:

meaning unkindness, but because his nerves were shattered by so many successive miseries. 'No, no,' said the old man, 'don't repulse your father, Dick, when he has come here to save you. Don't repulse me, my boy. Perhaps I have not been kind to you, not quite considerate, too harsh;

Expected (100 words):

at meaning the unkindness, feet but of because his his mistress, nerves licking were his shattered lips by and so again many his successive travel-sore miseries. paws. 'No, In no,' a said moment, the feeling old in man, his 'don't dumb repulse way your her father, loneliness, Dick, perhaps, when he he reached has up come and here laid to his save pink you. tongue Don't caressingly repulse upon me, her my brown boy. hand. Perhaps Dark I came have softly not and been with kind it to a you, noisy not wind quite that considerate, whistled too and harsh;

Model Output (99 words):

at meaning the unkindness, feet but of because his his nerves mistress, were licking shattered his by lips so and many again successive his miseries. travel-sore 'No, paws. no,' In said a the moment, old feeling man, in 'don't his repulse dumb your way father, her Dick, loneliness, when perhaps, he he reached has up come and here laid to his save pink you. tongue Don't caressingly repulse upon me, her my brown boy. hand. Perhaps Dark I came have softly not and been with kind it to a you, noisy not wind quite that considerate, whistled too and

Alignment: matches=79, mismatches=19, gaps=3

  10/100 - running avg: 0.674

  20/100 - running avg: 0.651

  30/100 - running avg: 0.657

  40/100 - running avg: 0.688

  50/100 - running avg: 0.708

============================================================

Sample 51/100 - Score: 0.661

============================================================

Fragment A:

friend's wrath. Alfred's upper lip began to curl. He cast a last withering look in Jimmy's direction, retired quickly from the scene and banged the door. When Jimmy again had the courage to lift his eyes he was confronted by the contemptuous gaze of Zoie, who was sitting up in

Fragment B:

truth!" "Very well," agreed the curate, releasing him; "now go ahead, and don't lie more than you can help." We abode the promised disclosure without the least misgiving; but even we had hardly given Harold due credit for his fertility of resource and powers of imagination. "I had just finished

Expected (100 words):

friend's truth!" wrath. "Very Alfred's well," upper agreed lip the began curate, to releasing curl. him; He "now cast go a ahead, last and withering don't look lie in more Jimmy's than direction, you retired can quickly help." from We the abode scene the and promised banged disclosure the without door. the When least Jimmy misgiving; again but had even the we courage had to hardly lift given his Harold eyes due he credit was for confronted his by fertility the of contemptuous resource gaze and of powers Zoie, of who imagination. was "I sitting had up just in finished

Model Output (68 words):

friend's truth!" wrath. "Very Alfred's well," upper agreed lip the began curate, to releasing curl. him; He "now cast go a ahead, last and withering don't look lie in more Jimmy's than direction, you retired can quickly help." from We the abode scene the and promised banged disclosure the without door. the When least Jimmy misgiving; again but had even the we courage had to just lift finished

Alignment: matches=67, mismatches=1, gaps=32

  60/100 - running avg: 0.704

  70/100 - running avg: 0.710

  80/100 - running avg: 0.722

  90/100 - running avg: 0.730

  100/100 - running avg: 0.721

============================================================

RESULTS (100 samples)

============================================================

Model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Dataset: datasets/50words_test.jsonl

Score Statistics:

  Mean:   0.7208 &plusmn; 0.1864

  Median: 0.6957

  Min:    0.1447

  Max:    1.0000

Performance Buckets:

  Perfect (&ge;0.999):    5 (5.0%)

  High (&ge;0.9):        25 (25.0%)

  Medium (0.5-0.9):   63 (63.0%)

  Low (<0.5):         12 (12.0%)

Alignment Analysis:

  Avg expected length: 100.0 words

  Avg output length:   89.0 words

  Total matches:       7442

  Total mismatches:    1428

  Total gaps:          1159

============================================================

## 100 words output

(venv) root@c222b92fd4bc:/workspace/interleave_grpo# python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/100words_test.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Loading model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

`torch_dtype` is deprecated! Use `dtype` instead!

Loading checkpoint shards: 100%|██████████████████| 2/2 [00:20<00:00, 10.20s/it]

Loading dataset: datasets/100words_test.jsonl

Loaded 500 samples

Evaluating on 100 samples...

The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

============================================================

Sample 1/100 - Score: 0.219

============================================================

Fragment A:

at the feet of his mistress, licking his lips and again his travel-sore paws. In a moment, feeling in his dumb way her loneliness, perhaps, he reached up and laid his pink tongue caressingly upon her brown hand. Dark came softly and with it a noisy wind that whistled and murmured and at last, growing more boisterous as the night deepened, whooped over her bead and tossed wildly the branches of a clump of trees that grew near. Annie-Many-Ponies listened to the wind and thought it a brother, perhaps, of the night wind that came to the Dakota prairies and

Fragment B:

meaning unkindness, but because his nerves were shattered by so many successive miseries. 'No, no,' said the old man, 'don't repulse your father, Dick, when he has come here to save you. Don't repulse me, my boy. Perhaps I have not been kind to you, not quite considerate, too harsh; my boy, it was not for want of love. Think of old times. I was kind to you then, was I not? When you were a child, and your mother was with us.' Mr. Naseby was interrupted by a sort of sob. Dick stood looking at him in a maze.

Expected (200 words):

at meaning the unkindness, feet but of because his his mistress, nerves licking were his shattered lips by and so again many his successive travel-sore miseries. paws. 'No, In no,' a said moment, the feeling old in man, his 'don't dumb repulse way your her father, loneliness, Dick, perhaps, when he he reached has up come and here laid to his save pink you. tongue Don't caressingly repulse upon me, her my brown boy. hand. Perhaps Dark I came have softly not and been with kind it to a you, noisy not wind quite that considerate, whistled too and harsh; murmured my and boy, at it last, was growing not more for boisterous want as of the love. night Think deepened, of whooped old over times. her I bead was and kind tossed to wildly you the then, branches was of I a not? clump When of you trees were that a grew child, near. and Annie-Many-Ponies your listened mother to was the with wind us.' and Mr. thought Naseby it was a interrupted brother, by perhaps, a of sort the of night sob. wind Dick that stood came looking to at the him Dakota in prairies a and maze.

Model Output (137 words):

at meaning the unkindness, feet but of because his his nerves mistress, were licking shattered his by lips so and many again successive his miseries. travel-sore 'No, paws. no,' In said a the moment, old feeling man, in 'don't his repulse dumb your way father, her Dick, loneliness, when perhaps, he he reached reached up up and and laid laid his pink pink tongue tongue caressingly upon upon her her brown hand. Dark came softly and with it a noisy wind that whistled whistled and and murmured murmured and at last, growing more boisterous as the night deepened, whooped over her bead and tossed wildly the branches of a clump of trees that grew near. Annie-Many-Ponies listened to the wind and thought it a brother, perhaps, of the night wind that came to the Dakota prairies and

Alignment: matches=51, mismatches=85, gaps=65

  10/100 - running avg: 0.286

  20/100 - running avg: 0.278

  30/100 - running avg: 0.300

  40/100 - running avg: 0.301

  50/100 - running avg: 0.307

============================================================

Sample 51/100 - Score: 0.308

============================================================

Fragment A:

friend's wrath. Alfred's upper lip began to curl. He cast a last withering look in Jimmy's direction, retired quickly from the scene and banged the door. When Jimmy again had the courage to lift his eyes he was confronted by the contemptuous gaze of Zoie, who was sitting up in bed and regarding him with undisguised disapproval. "Why didn't you tell him what the baby's name is?" she demanded. "How do _I_ know what the baby's name is?" retorted Jimmy savagely. "Sh! sh!" cautioned Aggie as she glanced nervously toward the door through which Alfred had just passed. "What does

Fragment B:

truth!" "Very well," agreed the curate, releasing him; "now go ahead, and don't lie more than you can help." We abode the promised disclosure without the least misgiving; but even we had hardly given Harold due credit for his fertility of resource and powers of imagination. "I had just finished saying my prayers," began that young gentleman, slowly, "when I happened to look out of the window, and on the lawn I saw a sight which froze the marrow in my veins! A burglar was approaching the house with snake-like tread! He had a scowl and a dark lantern, and

Expected (200 words):

friend's truth!" wrath. "Very Alfred's well," upper agreed lip the began curate, to releasing curl. him; He "now cast go a ahead, last and withering don't look lie in more Jimmy's than direction, you retired can quickly help." from We the abode scene the and promised banged disclosure the without door. the When least Jimmy misgiving; again but had even the we courage had to hardly lift given his Harold eyes due he credit was for confronted his by fertility the of contemptuous resource gaze and of powers Zoie, of who imagination. was "I sitting had up just in finished bed saying and my regarding prayers," him began with that undisguised young disapproval. gentleman, "Why slowly, didn't "when you I tell happened him to what look the out baby's of name the is?" window, she and demanded. on "How the do lawn _I_ I know saw what a the sight baby's which name froze is?" the retorted marrow Jimmy in savagely. my "Sh! veins! sh!" A cautioned burglar Aggie was as approaching she the glanced house nervously with toward snake-like the tread! door He through had which a Alfred scowl had and just a passed. dark "What lantern, does and

Model Output (142 words):

friend's truth!" wrath. "Very Alfred's well," upper agreed lip the began curate, to releasing curl. him; He "now cast go a ahead, last and withering don't look lie in more Jimmy's than direction, you retired can quickly help." from We the abode scene the promised and disclosure banged without the the least door. misgiving; When but Jimmy even again we had had the the courage contemptuous to gaze lift of his Zoie, eyes who he was confronted sitting by up the contemptuous gaze gaze of Zoie, who was sitting sitting up in bed and regarding him with undisguised disapproval. "Why didn't you tell him what the baby's name is?" she demanded. "How do _I_ know what the baby's name is?" retorted Jimmy savagely. "Sh! sh!" cautioned Aggie as she glanced nervously toward the door through which Alfred had just passed. "What does

Alignment: matches=70, mismatches=72, gaps=58

  60/100 - running avg: 0.301

  70/100 - running avg: 0.297

  80/100 - running avg: 0.312

  90/100 - running avg: 0.319

  100/100 - running avg: 0.315

============================================================

RESULTS (100 samples)

============================================================

Model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Dataset: datasets/100words_test.jsonl

Score Statistics:

  Mean:   0.3152 &plusmn; 0.1040

  Median: 0.3079

  Min:    0.1175

  Max:    0.6225

Performance Buckets:

  Perfect (&ge;0.999):    0 (0.0%)

  High (&ge;0.9):         0 (0.0%)

  Medium (0.5-0.9):    6 (6.0%)

  Low (<0.5):         94 (94.0%)

Alignment Analysis:

  Avg expected length: 200.0 words

  Avg output length:   130.2 words

  Total matches:       6945

  Total mismatches:    6024

  Total gaps:          7084

============================================================

## 200 words output

(venv) root@ba179f38e575:/workspace/interleave_grpo# python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/200words_test.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Loading model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

`torch_dtype` is deprecated! Use `dtype` instead!

Loading checkpoint shards: 100%|██████████████████| 2/2 [00:01<00:00,  1.87it/s]

Loading dataset: datasets/200words_test.jsonl

Loaded 500 samples

Evaluating on 100 samples...

The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

============================================================

Sample 1/100 - Score: 0.087

============================================================

Fragment A:

at the feet of his mistress, licking his lips and again his travel-sore paws. In a moment, feeling in his dumb way her loneliness, perhaps, he reached up and laid his pink tongue caressingly upon her brown hand. Dark came softly and with it a noisy wind that whistled and murmured and at last, growing more boisterous as the night deepened, whooped over her bead and tossed wildly the branches of a clump of trees that grew near. Annie-Many-Ponies listened to the wind and thought it a brother, perhaps, of the night wind that came to the Dakota prairies and caroused there until dawn bade it be still. Too red the blood of her people ran in her veins for her to be afraid of the night, even though she peopled it with dim shapes of her fancy. After a long while the wind grew chill. Annie-Many-Ponies shivered, and then rose and went to the horse and, reaching into the bundle which was still bound to the saddle, she worked a plaid shawl loose from the other things and pulled it out and wrapped it close around her and pulled it over her head like a cowl. Then she went

Fragment B:

meaning unkindness, but because his nerves were shattered by so many successive miseries. 'No, no,' said the old man, 'don't repulse your father, Dick, when he has come here to save you. Don't repulse me, my boy. Perhaps I have not been kind to you, not quite considerate, too harsh; my boy, it was not for want of love. Think of old times. I was kind to you then, was I not? When you were a child, and your mother was with us.' Mr. Naseby was interrupted by a sort of sob. Dick stood looking at him in a maze. 'Come away,' pursued the father in a whisper; 'you need not be afraid of any consequences. I am a man of the world, Dick; and she can have no claim on you--no claim, I tell you; and we'll be handsome too, Dick--we'll give them a good round figure, father and daughter, and there's an end.' He had been trying to get Dick towards the door, but the latter stood off. 'You had better take care, sir, how you insult that lady,' said the son, as black as night. 'You would not choose between your father and your mistress?' said the

Expected (400 words):

at meaning the unkindness, feet but of because his his mistress, nerves licking were his shattered lips by and so again many his successive travel-sore miseries. paws. 'No, In no,' a said moment, the feeling old in man, his 'don't dumb repulse way your her father, loneliness, Dick, perhaps, when he he reached has up come and here laid to his save pink you. tongue Don't caressingly repulse upon me, her my brown boy. hand. Perhaps Dark I came have softly not and been with kind it to a you, noisy not wind quite that considerate, whistled too and harsh; murmured my and boy, at it last, was growing not more for boisterous want as of the love. night Think deepened, of whooped old over times. her I bead was and kind tossed to wildly you the then, branches was of I a not? clump When of you trees were that a grew child, near. and Annie-Many-Ponies your listened mother to was the with wind us.' and Mr. thought Naseby it was a interrupted brother, by perhaps, a of sort the of night sob. wind Dick that stood came looking to at the him Dakota in prairies a and maze. caroused 'Come there away,' until pursued dawn the bade father it in be a still. whisper; Too 'you red need the not blood be of afraid her of people any ran consequences. in I her am veins a for man her of to the be world, afraid Dick; of and the she night, can even have though no she claim peopled on it you--no with claim, dim I shapes tell of you; her and fancy. we'll After be a handsome long too, while Dick--we'll the give wind them grew a chill. good Annie-Many-Ponies round shivered, figure, and father then and rose daughter, and and went there's to an the end.' horse He and, had reaching been into trying the to bundle get which Dick was towards still the bound door, to but the the saddle, latter she stood worked off. a 'You plaid had shawl better loose take from care, the sir, other how things you and insult pulled that it lady,' out said and the wrapped son, it as close black around as her night. and 'You pulled would it not over choose her between head your like father a and cowl. your Then mistress?' she said went the

Model Output (217 words):

at meaning the unkindness, feet but of because his his nerves mistress, were licking shattered his by lips so and many again successive his miseries. travel-sore 'No, paws. no,' In said a the old man, 'don't 'repulse your father, father, Dick, when he has come here to save you. Don't repulse me, my boy. Perhaps I have not been kind to you, not quite considerate, too harsh; my boy, it was not for want of love. Think of old times. I was kind to you then, was I not? When you were a child, and your mother was with us.' Mr. Naseby was interrupted by a sort of sob. Dick stood looking at him in a maze. 'Come away,' pursued the father in a whisper; 'you need not be afraid of any consequences. I am a man of the world, Dick; and she can have no claim on you--no claim, I tell you; and we'll be handsome too, Dick--we'll give them a good round figure, father and daughter, and there's an end.' He had been trying to get Dick towards the door, but the latter stood off. 'You had better take care, sir, how you insult that lady,' said the son, as black as night. 'You would not choose between your father and your mistress?' said the

Alignment: matches=51, mismatches=166, gaps=183

  10/100 - running avg: 0.101

  20/100 - running avg: 0.120

  30/100 - running avg: 0.125

  40/100 - running avg: 0.126

  50/100 - running avg: 0.131

============================================================

Sample 51/100 - Score: 0.146

============================================================

Fragment A:

friend's wrath. Alfred's upper lip began to curl. He cast a last withering look in Jimmy's direction, retired quickly from the scene and banged the door. When Jimmy again had the courage to lift his eyes he was confronted by the contemptuous gaze of Zoie, who was sitting up in bed and regarding him with undisguised disapproval. "Why didn't you tell him what the baby's name is?" she demanded. "How do _I_ know what the baby's name is?" retorted Jimmy savagely. "Sh! sh!" cautioned Aggie as she glanced nervously toward the door through which Alfred had just passed. "What does it matter WHAT the baby's name is so long as we have to send it back?" "I'll NOT send it back," declared Zoie emphatically, "at least not until morning. That will give Jimmy a whole night to get another one." "Another!" shrieked Jimmy. "See here, you two can't be changing babies every five minutes without Alfred knowing it. Even HE has SOME sense." "Nonsense!" answered Aggie shortly. "You know perfectly well that all young babies look just alike. Their own mothers couldn't tell them apart, if it weren't for their clothes." "But where can we GET another?" asked Zoie. Before

Fragment B:

truth!" "Very well," agreed the curate, releasing him; "now go ahead, and don't lie more than you can help." We abode the promised disclosure without the least misgiving; but even we had hardly given Harold due credit for his fertility of resource and powers of imagination. "I had just finished saying my prayers," began that young gentleman, slowly, "when I happened to look out of the window, and on the lawn I saw a sight which froze the marrow in my veins! A burglar was approaching the house with snake-like tread! He had a scowl and a dark lantern, and he was armed to the teeth!" We listened with interest. The style, though unlike Harold's native notes, seemed strangely familiar. "Go on," said the curate, grimly. "Pausing in his stealthy career," continued Harold, "he gave a low whistle. Instantly the signal was responded to, and from the adjacent shadows two more figures glided forth. The miscreants were both armed to the teeth." "Excellent," said the curate; "proceed." "The robber chief," pursued Harold, warming to his work, "joined his nefarious comrades, and conversed with them in silent tones. His expression was truly ferocious, and I ought to have said that he

Expected (400 words):

friend's truth!" wrath. "Very Alfred's well," upper agreed lip the began curate, to releasing curl. him; He "now cast go a ahead, last and withering don't look lie in more Jimmy's than direction, you retired can quickly help." from We the abode scene the and promised banged disclosure the without door. the When least Jimmy misgiving; again but had even the we courage had to hardly lift given his Harold eyes due he credit was for confronted his by fertility the of contemptuous resource gaze and of powers Zoie, of who imagination. was "I sitting had up just in finished bed saying and my regarding prayers," him began with that undisguised young disapproval. gentleman, "Why slowly, didn't "when you I tell happened him to what look the out baby's of name the is?" window, she and demanded. on "How the do lawn _I_ I know saw what a the sight baby's which name froze is?" the retorted marrow Jimmy in savagely. my "Sh! veins! sh!" A cautioned burglar Aggie was as approaching she the glanced house nervously with toward snake-like the tread! door He through had which a Alfred scowl had and just a passed. dark "What lantern, does and it he matter was WHAT armed the to baby's the name teeth!" is We so listened long with as interest. we The have style, to though send unlike it Harold's back?" native "I'll notes, NOT seemed send strangely it familiar. back," "Go declared on," Zoie said emphatically, the "at curate, least grimly. not "Pausing until in morning. his That stealthy will career," give continued Jimmy Harold, a "he whole gave night a to low get whistle. another Instantly one." the "Another!" signal shrieked was Jimmy. responded "See to, here, and you from two the can't adjacent be shadows changing two babies more every figures five glided minutes forth. without The Alfred miscreants knowing were it. both Even armed HE to has the SOME teeth." sense." "Excellent," "Nonsense!" said answered the Aggie curate; shortly. "proceed." "You "The know robber perfectly chief," well pursued that Harold, all warming young to babies his look work, just "joined alike. his Their nefarious own comrades, mothers and couldn't conversed tell with them them apart, in if silent it tones. weren't His for expression their was clothes." truly "But ferocious, where and can I we ought GET to another?" have asked said Zoie. that Before he

Model Output (226 words):

friend's truth!" wrath. "Very Alfred's well," upper agreed lip the began curate, to releasing curl. him; He "now cast go a ahead, last and withering don't look lie in more Jimmy's than direction, you retired can quickly help." from We the abode scene the promised and disclosure without but the least even misgiving; we but had hardly given due Harold credit for his fertility of resource and powers of imagination. " "I had just finished saying my prayers," began that young gentleman, slowly, "when I happened to look out of the window, and on the lawn I saw a sight which froze the marrow in my veins! A burglar was approaching the house with snake-like tread! He had a scowl and a dark lantern, and he was armed to the teeth!" We listened with interest. The style, though unlike Harold's native notes, seemed strangely familiar. "Go on," said the curate, grimly. "Pausing in his stealthy career," continued Harold, "he gave a low whistle. Instantly the signal was responded to, and from the adjacent shadows two more figures glided forth. The miscreants were both armed to the teeth." "Excellent," said the curate; "proceed." " The robber chief," pursued Harold, warming to his work, "joined his nefarious comrades, and conversed with them in silent tones. His expression was truly ferocious, and I ought to have said that he

Alignment: matches=76, mismatches=150, gaps=174

  60/100 - running avg: 0.129

  70/100 - running avg: 0.129

  80/100 - running avg: 0.128

  90/100 - running avg: 0.129

  100/100 - running avg: 0.128

============================================================

RESULTS (100 samples)

============================================================

Model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Dataset: datasets/200words_test.jsonl

Score Statistics:

  Mean:   0.1277 &plusmn; 0.0374

  Median: 0.1262

  Min:    0.0565

  Max:    0.2317

Performance Buckets:

  Perfect (&ge;0.999):    0 (0.0%)

  High (&ge;0.9):         0 (0.0%)

  Medium (0.5-0.9):    0 (0.0%)

  Low (<0.5):        100 (100.0%)

Alignment Analysis:

  Avg expected length: 400.0 words

  Avg output length:   219.6 words

  Total matches:       6442

  Total mismatches:    15495

  Total gaps:          18084

============================================================

## Data Summary

| Text Size | Total Text Size | % Alignment |
| --- | --- | --- |
| 10 | 20 | 0.9811 &plusmn; 0.0586 |
| 25 | 50 | 0.9408 &plusmn; 0.1155 |
| 50 | 100 | 0.7208 &plusmn; 0.1864 |
| 100 | 200 | 0.3152 &plusmn; 0.1040 |
| 200 | 400 | 0.1277 &plusmn; 0.0374 |
|  |  |  |