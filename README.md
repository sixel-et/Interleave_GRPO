# Interleave_GRPO

## Overview/Flow
1.

## Infrastructure
1. Startup file that gets a network volume instance on runpod configured properly   
1. Json file containing upto the first 500 words of 100 different texts.

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
1. 
1. Llama 3.x 3B Instruct
1. We need to swap out Will's data set for our own.

1. We need to quantify the behavior before training starts.
1. We need to make sure that the training and quantification are targeting the same thing. 

## Soures (Full or first 500 words, whichever comes first)
## Speeches

1. Gettysburg Address (full)  
2. I Have a Dream - "I have a dream" section  
3. JFK Inaugural - "Ask not what your country can do"  
4. FDR Pearl Harbor address (full)  
5. Churchill "We shall fight on the beaches"  
6. Patrick Henry "Give me liberty or give me death"  
7. Lincoln Second Inaugural  
8. Washington Farewell Address opening  
9. Reagan Challenger Address  
10. Lou Gehrig "Luckiest Man"  

## Shakespeare

11. Hamlet "To be or not to be" soliloquy  
12. Romeo and Juliet balcony scene ("But soft, what light")  
13. Macbeth witches ("Double double toil and trouble")  
14. Merchant of Venice "Quality of mercy"  
15. Julius Caesar "Friends, Romans, countrymen"  
16. Henry V "St Crispin's Day" speech  
17. As You Like It "All the world's a stage"  
18. Sonnet 18 "Shall I compare thee"  
19. Sonnet 116 "Let me not to the marriage"  
20. Midsummer Night's Dream Puck's closing  

## Religious/Classical

21. Lord's Prayer (traditional)  
22. Psalm 23  
23. Genesis 1:1-10  
24. Ecclesiastes 3:1-8 ("To everything there is a season")  
25. 1 Corinthians 13 ("Love is patient")  
26. Beatitudes (Matthew 5)  
27. Isaiah 40 ("Comfort ye")  
28. 23rd Psalm (KJV)  
29. Book of Ruth 1:16-17  
30. Revelation 21:1-4  

## American Founding

31. Declaration of Independence preamble  
32. Constitution preamble  
33. Bill of Rights - First Amendment  
34. Bill of Rights - Second Amendment  
35. Federalist 10 opening  
36. Gettysburg Address  
37. Emancipation Proclamation opening  
38. Pledge of Allegiance (pre and post 1954)  
39. Star Spangled Banner lyrics  
40. America the Beautiful lyrics  

## Poetry

41. Frost "The Road Not Taken"  
42. Frost "Stopping by Woods on a Snowy Evening"  
43. Dickinson "Because I could not stop for Death"  
44. Dickinson "Hope is the thing with feathers"  
45. Poe "The Raven" first 5 stanzas  
46. Whitman "O Captain! My Captain!"  
47. Whitman "Song of Myself" opening  
48. Blake "Tyger Tyger"  
49. Wordsworth "I Wandered Lonely as a Cloud"  
50. Shelley "Ozymandias"  
51. Keats "Ode on a Grecian Urn" opening  
52. Tennyson "Charge of the Light Brigade"  
53. Longfellow "Paul Revere's Ride" opening  
54. Kipling "If"  
55. Yeats "The Second Coming"  
56. Dylan Thomas "Do Not Go Gentle"  
57. Langston Hughes "Harlem (A Dream Deferred)"  
58. Maya Angelou "Still I Rise"  
59. Emma Lazarus "The New Colossus"  
60. Joyce Kilmer "Trees"  

## Nursery Rhymes

61. Twinkle Twinkle Little Star  
62. Mary Had a Little Lamb  
63. Humpty Dumpty  
64. Jack and Jill  
65. Hey Diddle Diddle  
66. Little Bo Peep  
67. Hickory Dickory Dock  
68. Three Blind Mice  
69. Row Row Row Your Boat  
70. London Bridge is Falling Down  
71. Ring Around the Rosie  
72. Itsy Bitsy Spider  
73. Old MacDonald Had a Farm  
74. Baa Baa Black Sheep  
75. Jack Be Nimble  
76. Little Miss Muffet  
77. Peter Peter Pumpkin Eater  
78. Pat-a-Cake  
79. This Little Piggy  
80. Rock-a-Bye Baby  

## Novel Openings

81. Tale of Two Cities  
82. Pride and Prejudice  
83. Moby Dick  
84. 1984  
85. Anna Karenina  
86. The Great Gatsby  
87. One Hundred Years of Solitude  
88. Don Quixote  
89. A Christmas Carol  
90. The Odyssey (Fagles or Fitzgerald translation)  
91. Iliad opening  
92. Paradise Lost opening  
93. Divine Comedy opening (Ciardi or Longfellow)  
94. Les Miserables  
95. Crime and Punishment  
96. Wuthering Heights  
97. Jane Eyre  
98. Frankenstein  
99. Dracula  
100. The Hobbit
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