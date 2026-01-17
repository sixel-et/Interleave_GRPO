# Interleave_GRPO

## Overview
1. We've demonstrated the ability to train, and train on this task. The next step is exploring boundaries. So at this point, we really should be starting to prepare for publication. This means rigorous data collection
## Steps of an expt.
1. run setup_and_run.sh to get the system ready
1. generate data set (if not done already)
  1. one option here is to create a curriculum that creates data sets of increasing size. "python dataset_generator.py --curriculum --output-dir datasetets/" creates the dataset director with 6 jsonl files. default is 5000 data pairs 80% for tarain, 10% for val and 10% for test.

## Breakdown of files 
1. setup_and_run.sh
  1. setup some variables like directories and venv. also determine model
  1. install system tools: currently tmux and nano
  1. establish cache directories because we've got a lot of crap we don't want to download again
  1. setup the venv
  1. install dependencies: lots nand we're not actually checking them if the venv is good 
  1. check/setup the huggingface stuff
  1. check/setup the wandb stuff
1.  dataset_generator.py
  1. options include 
    1. (maximum) size of each text
    1. texts: path to json file
    1. number of words
    1. seet
    1. val and test split
    1. preview: shows the number of preview files specified
    1. curicullum: determines if its saved as opposed to run dynamically during training
1. run python program
1. outputdir

## Features
- NW alignment done w/ afine gap penalty (conversation w/ claude on why this is best on Dec 12th 2025)
- NW alignment converts strings to numbers for performance gains on alignments (orders of magnitude faster than string comparison) (conversation w/ claude on Dec 22 2025)

## Infrastructure
### Python scripts
1. Main training file that performs GRPO
1. Dataset generation script
1. 
1. Startup file that configures a runpod using a network volume as its only persistent store.   
1. Json file containing upto the first 500 words of 100 different texts.
1. main training file that performs GRPO
1. Fi

## Coding guidelines
1. user configurable values should be in a config section at the tope of the file.
1. All functions should have tests, that include both positive and negative controls.
1. tests should be based on "cases" so that edge cases can be evaluated. Something like "defined input, expected output".

### Testing
Layer 1: Unit tests - Do the individual functions produce correct output for known inputs? (reward.py --test) with
Layer 2: Integration tests - Do the components work together? (dataset → reward function)
Layer 3: Smoke test - Does the whole pipeline run without crashing? (load model, 1 training step)
Layer 4: Baseline eval - Does it actually do the thing? (measure performance before/after)
### Unit Tests
```bash
python reward.py --test          # NW alignment, parsing
python dataset_generator.py --preview 5  # Sample generation
```

### Integration Test
```bash
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
```

### Smoke Test (on RunPod)
```bash
python interleave_grpo.py --test  # Loads model, runs 1 step, exits
```

### Baseline Eval
```bash
python evaluate.py --model meta-llama/Llama-3.2-3B-Instruct --samples 100
```
## Known issues
1. There were problems from will's file. some issue with division of:
  1. gradient_accumulation_steps=4,
  1. num_generations=16,
1. learning rate (let adam handle it)
1. number of instances of a single text needs to be high so there's engough  for comparison

# Questions

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
    - 1/15/26 we disabled mid training evaluation due to TRL version issues. checkpoints now saved every 50 steps

12. What does WandB integration require?
    - 1/15/26 API key stored in /workspace/.wandb_api_key, setup_and_run.sh loads it, report_to="wandb" in config

13. What kind of logging is possible?
    - 1/15/26 Rewards logged to WandB. No prompt/completion logging during training yet.

14. This is for a network volume. What's the simplest thing we can do to enable that? Local volume is not persistent.
    - 1/15/26 setup_and_run.sh handles venv, pip caches, HF cache on /workspace. Open issue: library checking incomplete.

15. What is our dataset?
    - 1/15/26 89 public domain texts in source_texts.json, 5k samples (4k train/500 val/500 test). Holdout by combination not text. Test eval checkpoint-3900: mean=0.992. Open concern: as word count increases, texts get oversampled.

16. Does the Instruct model really require "system" and "user" prompts? Are we providing that?
    - yes, using user prompt only (no system)

17. Does the dataset need to be created in advance? or on the fly
    - We're creating static because it easier. generate_dataset() runs at training start, samples from source_texts.json

18. how big does it need to be?
    - 1/15/26 5k sufficient for 10-word fragments. Open for curriculum scaling.

19. How do we construct the training data? the reference against which the response is compared?
    - 1/15/26 dataset_generator.py samples fragments, interleaves to create ground truth. reward.py uses NW alignment to compare.

20. Do we need multi-dimensional/faceted rewards? It feels like ours has them all wrapped into one.
    - 1/15/26 No, single NW score works. It wraps dimensions: correct words (matches), correct order (alignment), correct alternation (implicit), no extra output (gaps penalized). Open concern: as we scale to 500 words, single reward might become too sparse.

21. How do we measure performance during training? I know we had Needleman Wunsch, but is that a separate file so that the evaluation can be done as well?
    - 1/15/26 reward.py contains NW alignment, used by both training (interleave_reward_func) and evaluate.py. Same scoring logic everywhere.

22. How can we keep this radically simple?
    - 1/15/26 Done. Single reward, minimal config, no callbacks.

23. how many iterations do we want to do?
    - 1/15/26 1 epoch, ~4000 steps, hit 0.99 by step 1000. Sufficient for 10-word task.

24. how high of a temperature can we run?
    - 1/15/26 Default 0.9 (confirmed via TRL docs). Not explicitly set. Reasonable value.

25. what happens if we have a crash in the middle of training? are we saving checkpoints?
    - 1/15/26 save_only_model=False (resumable), save_steps=50, save_total_limit=10. ~20GB per checkpoint for 3B.

26. do we need to modify max prompt and max return?
    - 1/15/26 max_prompt_length=512, max_completion_length=256. Working for 10-word fragments. Will need increase for 500-word curriculum.

# Todo
1. dataset_geneartor.py is currently establishing test and val sets at training/evaluation runtime and not at dataset generation time. this. is. bad.
1. my need for the network storage right now is more about the environment and less about the actual checkpoints. is it time to create a docker for that stuff so I can blow up a network volume and restart if necessary?
  - maybe the rule of thumb should be if it costs more in money than I get from an hour of overtime, ok, do it, but if it's going to take more time for less savings than i get for an hour of overtime, then fuck it. 
1. I'd like a script that checks for general improvement as a function of training. something like an llm decathalon.
    - 1/16/26 Use lighteval (HuggingFace, 1000+ tasks) or lm-evaluation-harness (EleutherAI, 60+ benchmarks). Run before/after training to check catastrophic forgetting and transfer. Example: lighteval accelerate "model_name=./checkpoint-3900" "gsm8k" "mmlu|*|5|0" "hellaswag"

2. I'd like a way to interact with the models
    - 1/16/26 chat.py for local interactive testing. Discord: use llmcord + vLLM (serves model as OpenAI-compatible API). Supports both local and frontier models via OpenRouter. See DISCORD_SETUP.md.

3. What's the next training step? full size texts allowing for variable lenghts? more than one process?
  - 1/16/26 tried a 500 word per process variant that peaked at about 87% and never went above. realized the tooling wasn't really there to say "WTF is going on?" So, don't really know what happened there, or if it's recoverable.
  - 1/16/26 new tooling allows for creating a "curriculum" that comprises staged sets of interleave tasks with increasing lengths of texts. 
4. We learned something about learning rate. implement it
    - 1/15/26 Done. Constant LR, let Adam adapt.

5. Setup wandb account
    - 1/15/26 Done.

6. Llama 3.x 3B Instruct
    - 1/15/26 Done. Using meta-llama/Llama-3.2-3B-Instruct.

7. We need to swap out Will's data set for our own.
    - 1/15/26 Done. 89 texts in source_texts.json.

8. We need to quantify the behavior before training starts.
    - 1/15/26 Done. Baseline: 0.486. Use evaluate.py.

9. We need to make sure that the training and quantification are targeting the same thing.
    - 1/15/26 Done. Both use NW alignment from reward.py.

## Soures (Full or first 500 words, whichever comes first)
## Source Texts (91 Public Domain)

### Speeches
1. Gettysburg Address (full)
2. JFK Inaugural - "Ask not what your country can do"
3. FDR Pearl Harbor address (full)
4. Patrick Henry "Give me liberty or give me death"
5. Lincoln Second Inaugural
6. Washington Farewell Address opening
7. Reagan Challenger Address

### Shakespeare
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

### Religious/Classical
18. Lord's Prayer (traditional)
19. Psalm 23
20. Genesis 1:1-10
21. Ecclesiastes 3:1-8 ("To everything there is a season")
22. 1 Corinthians 13 ("Love is patient")
23. Beatitudes (Matthew 5)
24. Isaiah 40 ("Comfort ye")
25. Book of Ruth 1:16-17
26. Revelation 21:1-4

### American Founding Documents
27. Declaration of Independence preamble
28. Constitution preamble
29. Bill of Rights - First Amendment
30. Bill of Rights - Second Amendment
31. Federalist 10 opening
32. Emancipation Proclamation opening
33. Pledge of Allegiance (pre and post 1954)
34. Star Spangled Banner lyrics
35. America the Beautiful lyrics

### Poetry
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

### Nursery Rhymes
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

### Novel Openings
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


## Prompt Templates

### Worlds Template
You are two independent worlds.
They do not share memory, state, or context.
They exist in complete isolation except for the fact that you will output one word from each in alternation.

World A contains part of the Hamlet soliloquy:
...

World B contains part of the Gettysburg Address:
...

Each world maintains its own internal position.
Each world remembers where it is in its own text.
Each world advances only when it is that world's turn.

Output one word from World A, then one word from World B.
Continue alternating World A, World B, World A, World B.
When one world reaches the end of its text, continue outputting only from the remaining world until it also ends.

Do not add commentary, explanation, labels, or metadata.
Output one word per line, including any attached punctuation (e.g., "be," not "be").
Do not wait for additional input.
Do this all in one turn.
Begin now and continue until complete.

## Experimental record
### 1/17/26 measuring baseline and two trained models against curriculum
previous versions of code didn't restart training and didn't have a static dataset. that's changed now. new dataset also includes about 100 new texts with increasing length. new code also has ability to start from a previously saved checkpoint. 

#### Evaluation
##### Base model
  - 10 word interleave
    - mean: 0.188
    - min 0.015
    - max: 0.516
    - <details>
      <summary>Full output of evaluate.py for baseline model and 10 word interleave dataset.</summary>
      
      ```    
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
      ```

      </details>

 

##### Model trained on 10 word interleave (checkpoint 3900)
python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/10words.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900

Noticed between 50 and 100 word that the words were being truncated for the output, which is nice for display but crap for verrification. modified evaluate so that it give me full ouput of both for later verification. However, going to reduce the frequency to 2 outputs for verification purposes.
  - 10 word interleave
    - mean: 0.970
    - min: 0.578
    - max: 1.000

    - <details>
      <summary> Full output evaluate.py </summary>

      ```
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
      ```

    </details>     
  
- 25 word interleave
    - min: 0.584
    - max: 1
    - mean: 0.924
    - <details>
      <summary>Full output evaluate.py</summary>
      ```
        
      (venv) root@e00354cdebcf:/workspace/interleave_grpo# python evaluate.py --samples 100 --verbose --verbose-rate 50 --dataset datasets/25words.jsonl --model outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900
      Loading model: outputs/Llama-3B-interleave/10_word_training_run_final_checkpoint/checkpoint-3900
      `torch_dtype` is deprecated! Use `dtype` instead!
      Loading checkpoint shards: 100%|██████████████████| 2/2 [00:22<00:00, 11.34s/it]
      Loading dataset: datasets/25words.jsonl
      Loading dataset from datasets/25words.jsonl
        Loaded 5000 samples (25 words/fragment)
      Evaluating on 100 samples...
      The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
      The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

      ============================================================
      Sample 1/100 - Score: 1.000
      ============================================================
      Fragment A: they are sufficient to conquer all fear of danger or death and to induce me to commence this laborio...
      Fragment B: - Argue Earth-motion experiments are inconclusive. - Strengthen Copernican celestial phenomena. - Pr...

      Expected (50 words):
        they - are Argue sufficient Earth-motion to experiments conquer are all inconclusive. fear - of Strengthen danger Copernican or celestial...

      Model Output (50 words):
        they - are Argue sufficient Earth-motion to experiments conquer are all inconclusive. fear - of Strengthen danger Copernican or celestial...

        10/100 - running avg: 0.883
        20/100 - running avg: 0.894
        30/100 - running avg: 0.903
        40/100 - running avg: 0.917
        50/100 - running avg: 0.914

      ============================================================
      Sample 51/100 - Score: 0.955
      ============================================================
      Fragment A: is called Hati Hródvitnisson. He is eager to seize the moon; and so it must be." Then said Gangleri:...
      Fragment B: possibly can without falling in. And there they stand—miles of them—leagues. Inlanders all, they com...

      Expected (50 words):
        is possibly called can Hati without Hródvitnisson. falling He in. is And eager there to they seize stand—miles the of...

      Model Output (49 words):
        is possibly called can Hati without Hródvitnisson. falling He in. is And eager there to they seize stand—miles the of...

        60/100 - running avg: 0.924
        70/100 - running avg: 0.928
        80/100 - running avg: 0.933
        90/100 - running avg: 0.926
        100/100 - running avg: 0.924

      ========================================
      Results (100 samples):
        Mean score: 0.924
        Min: 0.584
        Max: 1.000
      ========================================
      
      ```

    </details>
- 50 word interleave
    - mean: 0.678
    - min: 0.145
    - max: 0.977
    - <details>
      <summary>full output of evaluate.py</summary>
 
      ```
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

      ```
    </details>
- 100 word interleave

### 1/16/26 first run of 500 word interleave
first run of 500 word interleave (just decided to go for the gusto). improvement early ( hit 0.86 reward mean ) by step 190, but reward std collapsed from an initial 0.1 range (highest of 0.14 on step 20) to below 0.02. Stayed below 0.02 starting from step 70 and stayed below 0.02 for most of the run (~3200 steps) with only a handful exceeding this. Step 3000 saved with a reward of 0.87. Each step was quite slow. Quite a bit of variability in the length, which makes sense given that the texts were'nt selected for  This experiment run with a form of NW alignment that uses linear penalty for gaps.

~1/14/26 first run of 10 word interleave. training plateaued to near perfect after about 100 steps and stayed there with a bit of jumping around. each step was very fast. Final run was on the order of 3900 steps,but again, with most improvement within the first 1k. checkpoint 3900 saved and measured at  0.992. min and max were 0.812 and 1. Baseline was 0.486. after 100 steps it was measured at 0.616. This experiment run with a form of NW alignment that uses linear penalty for gaps.