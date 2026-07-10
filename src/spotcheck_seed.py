"""Hand-verified seed sets for the LLM spot-check gate.

Fallback only: once data/gold.csv exists, spotcheck samples from it instead
(larger, stratified). Every entry here is a sound root (no weak radicals, no
hamza) so misrendering can never contaminate the truth labels.

REAL: attested MSA Form I verbs, each with its exact vocalization.
FAKE: two difficulty tiers — unattested vocalizations of real roots (hard
negatives: the root exists, that pattern does not) and phonotactically odd
invented roots (easy negatives). A model that answers "true" to everything
scores 50% here and fails the specificity gate.
"""

# (verb, root, pattern)
SEED_REAL = [
    ('كَتَبَ', 'كتب', 'faʿala'),   # to write
    ('جَلَسَ', 'جلس', 'faʿala'),   # to sit
    ('فَتَحَ', 'فتح', 'faʿala'),   # to open
    ('دَخَلَ', 'دخل', 'faʿala'),   # to enter
    ('خَرَجَ', 'خرج', 'faʿala'),   # to exit
    ('قَتَلَ', 'قتل', 'faʿala'),   # to kill
    ('طَبَخَ', 'طبخ', 'faʿala'),   # to cook
    ('نَصَرَ', 'نصر', 'faʿala'),   # to aid
    ('شَرِبَ', 'شرب', 'faʿila'),   # to drink
    ('عَلِمَ', 'علم', 'faʿila'),   # to know
    ('سَمِعَ', 'سمع', 'faʿila'),   # to hear
    ('حَفِظَ', 'حفظ', 'faʿila'),   # to memorize
    ('رَكِبَ', 'ركب', 'faʿila'),   # to ride
    ('لَعِبَ', 'لعب', 'faʿila'),   # to play
    ('ضَحِكَ', 'ضحك', 'faʿila'),   # to laugh
    ('فَرِحَ', 'فرح', 'faʿila'),   # to rejoice
    ('كَبُرَ', 'كبر', 'faʿula'),   # to grow big
    ('حَسُنَ', 'حسن', 'faʿula'),   # to be good
    ('صَغُرَ', 'صغر', 'faʿula'),   # to be small
    ('كَرُمَ', 'كرم', 'faʿula'),   # to be generous
]

SEED_FAKE = [
    # Hard negatives: real roots, unattested vocalization
    ('كَتِبَ', 'كتب', 'faʿila'),
    ('كَتُبَ', 'كتب', 'faʿula'),
    ('جَلُسَ', 'جلس', 'faʿula'),
    ('دَخِلَ', 'دخل', 'faʿila'),
    ('ضَحُكَ', 'ضحك', 'faʿula'),
    ('فَتُحَ', 'فتح', 'faʿula'),
    ('خَرُجَ', 'خرج', 'faʿula'),
    ('سَمُعَ', 'سمع', 'faʿula'),
    ('نَصِرَ', 'نصر', 'faʿila'),
    ('لَعُبَ', 'لعب', 'faʿula'),
    # Easy negatives: invented roots
    ('ثَذَغَ', 'ثذغ', 'faʿala'),
    ('ظَثَكَ', 'ظثك', 'faʿala'),
    ('غَذَثَ', 'غذث', 'faʿala'),
    ('ضَظَغَ', 'ضظغ', 'faʿala'),
    ('ذَثَبَ', 'ذثب', 'faʿala'),
    ('ثَضِقَ', 'ثضق', 'faʿila'),
    ('ظَغِدَ', 'ظغد', 'faʿila'),
    ('خَثُظَ', 'خثظ', 'faʿula'),
    ('ذَظُفَ', 'ذظف', 'faʿula'),
    ('غَظِثَ', 'غظث', 'faʿila'),
]
