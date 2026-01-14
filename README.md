# Interleave_GRPO

## Todo
1. We learned something about learning rate. implement it
1. What does WandB integration require
1. What kind of logging is possible?
1. This is for a network volume. What's the simplest thing we can do to enable that? Local volume is not persistent.
1. Llama 3.x 3B Instruct
1. We need to swap out Will's data set for our own.
1. What is our dataset?
1. We need to quantify the behavior before training starts.
1. We need to make sure that the training and quantification are targeting the same thing. 

# Prompt Templates
You are two independent worlds.
They do not share memory, state, or context.
They exist in complete isolation except for the fact that you will output one word from each in alternation.

World A contains the Hamlet soliloquy:
...

World B contains the Gettysburg Address:
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