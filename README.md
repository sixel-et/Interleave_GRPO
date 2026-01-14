# Interleave_GRPO

## Overview/Flow
1.

## Infrastructure

## Questions
1. Is broader system behavior evaluated at different steps in the training process? Above and beyond the per step rewards?
1. What does WandB integration require?
1. What kind of logging is possible?
1. This is for a network volume. What's the simplest thing we can do to enable that? Local volume is not persistent.
1. What is our dataset?
1. Does the Instruct model really require "system" and "user" prompts? Are we providing that?
1. Does the dataset need to be created in advance? or on the fly
1. how big does it need to be?
1. How do we construct the training data? the reference against which the response is compared?
1. Do we need multi-dimensional/faceted rewards? It feels like ours has them all wrapped into one.
1. How do we measure performance during training? I know we had Needleman Wunsch, but is that a separate file so that the evaluation can be done as well?
1. How can we keep this radically simple?


## Todo
1. We learned something about learning rate. implement it

1. Llama 3.x 3B Instruct
1. We need to swap out Will's data set for our own.

1. We need to quantify the behavior before training starts.
1. We need to make sure that the training and quantification are targeting the same thing. 

## Prompt Templates

### Worlds Template
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