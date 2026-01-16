# Interleave_GRPO

## Overview/Flow
1. We've demonstrated the ability to train, and train on this task. The next step is exploring boundaries. So at this point, we really should be starting to prepare for publication. This means rigorous data collection

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
1. number of steps for comparison needs to be highish

## Questions
1. Is broader system behavior evaluated at different steps in the training process? Above and beyond the per step rewards?
  - 1/15/26 we disabled mid training evaluation due to TRL version issues. checkpoints now saved every 50 steps
1. What does WandB integration require?
1. What kind of logging is possible?
1. This is for a network volume. What's the simplest thing we can do to enable that? Local volume is not persistent.
1. What is our dataset?
1. Does the Instruct model really require "system" and "user" prompts? Are we providing that?
  - yes
1. Does the dataset need to be created in advance? or on the fly
  - We're creating static because it easier
1. how big does it need to be?
  - Claude's code was initially 1k. I'm thinking 10k is better
1. How do we construct the training data? the reference against which the response is compared?
1. Do we need multi-dimensional/faceted rewards? It feels like ours has them all wrapped into one.
1. How do we measure performance during training? I know we had Needleman Wunsch, but is that a separate file so that the evaluation can be done as well?
1. How can we keep this radically simple?
1. how many iterations do we want to do?
1. how high of a temperature can we run?
1. what happens if we have a crash in the middle of training? are we saving checkpoints?
1. do we need to modify max prompt and max return?


## Todo
1. I'd like a script that checks for general improvement as a function of training. something like an llm decathalon.
1. I'd like a way to interact with the models
1. What's the next training step? full size texts allowing for variable lenghts? more than one process? 
1. We learned something about learning rate. implement it
1. Setup wandb account
1. Llama 3.x 3B Instruct
1. We need to swap out Will's data set for our own.
1. We need to quantify the behavior before training starts.
1. We need to make sure that the training and quantification are targeting the same thing. 

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